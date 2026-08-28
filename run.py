#!/usr/bin/env python3
"""
RL-SDN Interactive Terminal User Interface (TUI).
Allows one-click execution to choose algorithms, toggle network chaos,
and view rich tabular performance results directly in the terminal.
"""
import sys
import os
import copy
import yaml
import logging
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table

from rl_sdn_controller.control_plane.controller import SDNController
from rl_sdn_controller.network.topology import NetworkTopology
from rl_sdn_controller.sdn_api.routing_table_api import RoutingTableAPI
from rl_sdn_controller.data_plane.simulator import DataPlaneSimulator
from rl_sdn_controller.network.routing_engine import OSPFRoutingEngine, RoundRobinRoutingEngine
from rl_sdn_controller.cli.visualizer import TerminalVisualizer
from rl_sdn_controller.sdn_api.stats_provider import LinkStats

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
console = Console()
viz = TerminalVisualizer()


def packet_weighted_latency(telem):
    """Compute packet-count-weighted average latency (physically correct metric)."""
    total_lat_w = sum(st.avg_latency_ms * st.tx_packets for st in telem.values() if st.avg_latency_ms > 0)
    total_pkt = sum(st.tx_packets for st in telem.values() if st.avg_latency_ms > 0)
    return total_lat_w / total_pkt if total_pkt > 0 else 0.0


def load_configs(chaos_enabled: bool = True):
    top_path = "configs/topology.yaml"
    traffic_path = "configs/traffic_profiles.yaml"
    rl_path = "configs/rl_config.yaml"
    chaos_path = "configs/chaos_config.yaml" if chaos_enabled else None

    with open(traffic_path, "r") as f:
        traffic_configs = yaml.safe_load(f).get("flows", [])
    with open(rl_path, "r") as f:
        rl_config = yaml.safe_load(f)
    chaos_config = None
    if chaos_path and os.path.exists(chaos_path):
        with open(chaos_path, "r") as f:
            chaos_config = yaml.safe_load(f)

    return top_path, traffic_configs, rl_config, chaos_config


def aggregate_telemetry(history):
    if not history:
        return {}
    aggregated = {}
    link_keys = history[0].keys()
    for key in link_keys:
        sample = history[0][key]
        total_tx_bytes = sum(h[key].tx_bytes for h in history if key in h)
        total_tx_p = sum(h[key].tx_packets for h in history if key in h)
        total_drop_p = sum(h[key].dropped_packets for h in history if key in h)
        total_arrived = total_tx_p + total_drop_p
        drop_rate = (total_drop_p / total_arrived * 100.0) if total_arrived > 0 else 0.0

        avg_util = float(np.mean([h[key].utilization_pct for h in history if key in h]))
        valid_lat = [h[key].avg_latency_ms for h in history if key in h and h[key].avg_latency_ms > 0]
        avg_lat = float(np.mean(valid_lat)) if valid_lat else 0.0
        avg_q = int(np.mean([h[key].queue_depth for h in history if key in h]))

        aggregated[key] = LinkStats(
            src=sample.src,
            dst=sample.dst,
            capacity_mbps=sample.capacity_mbps,
            tx_bytes=total_tx_bytes,
            tx_packets=total_tx_p,
            dropped_bytes=total_drop_p * 1000,
            dropped_packets=total_drop_p,
            queue_depth=avg_q,
            max_queue_packets=sample.max_queue_packets,
            utilization_pct=avg_util,
            drop_rate_pct=drop_rate,
            avg_latency_ms=avg_lat
        )
    return aggregated


def evaluate_rl_agent(use_dueling: bool, episodes: int, chaos: bool):
    top_path, traffic_configs, rl_config, chaos_config = load_configs(chaos)
    model_config = copy.deepcopy(rl_config)
    model_config["agent"]["use_dueling"] = use_dueling

    model_name = "Dueling DQN" if use_dueling else "Standard DQN"
    console.print(f"\n[bold green]⚙️  Training {model_name} for {episodes} episodes...[/bold green]")
    controller = SDNController(top_path, traffic_configs, model_config, chaos_config=chaos_config)
    controller.train_episodes(num_episodes=episodes, verbose=False)

    console.print(f"[bold green]📊 Evaluating {model_name} policy...[/bold green]")
    controller.env.max_steps = 600
    state, _ = controller.env.reset()
    tp_list, drop_list, lat_list = [], [], []
    telemetry_history = []
    done = False

    while not done:
        action = controller.agent.select_action(state, evaluate=True)
        state, reward, terminated, truncated, info = controller.env.step(action)
        done = terminated or truncated
        tp_list.append(info["total_throughput_mbps"])
        drop_list.append(info["avg_drop_pct"])
        lat_list.append(info["avg_latency_ms"])
        if "telemetry" in info:
            telemetry_history.append(info["telemetry"])

    valid_lat = [l for l in lat_list if not np.isnan(l) and l > 0]
    metrics = {
        "throughput_mbps": float(np.nanmean(tp_list)) if tp_list else 0.0,
        "drop_rate_pct": float(np.nanmean(drop_list)) if drop_list else 0.0,
        "avg_latency_ms": float(np.mean(valid_lat)) if valid_lat else 0.0,
        "p99_latency_ms": float(np.percentile(valid_lat, 99)) if valid_lat else 0.0
    }
    aggregated_telemetry = aggregate_telemetry(telemetry_history)
    return metrics, aggregated_telemetry


def evaluate_ospf(chaos: bool):
    top_path, traffic_configs, rl_config, chaos_config = load_configs(chaos)
    topo = NetworkTopology(top_path)
    rt_api = RoutingTableAPI()
    sim = DataPlaneSimulator(topo, rt_api, traffic_configs, chaos_config=chaos_config)
    flow_ids = list(sim.generators.keys())
    engine = OSPFRoutingEngine(topo, rt_api)
    engine.update_routes(flow_ids, sim.flow_src_dst)

    tp_list, drop_list, lat_list = [], [], []
    for _ in range(600):
        for lq in sim.links.values():
            lq.reset_window_stats()
        for _ in range(10):
            sim.step(0.01)
        telem = sim.stats_provider.collect_window_telemetry(0.1)
        tp = sum((st.tx_bytes * 8.0 / 1_000_000.0) / 0.1 for st in telem.values())
        dr = float(np.mean([st.drop_rate_pct for st in telem.values()])) if telem else 0.0
        lat = packet_weighted_latency(telem)
        tp_list.append(tp)
        drop_list.append(dr)
        lat_list.append(lat)

    valid_lat = [l for l in lat_list if not np.isnan(l) and l > 0]
    return {
        "throughput_mbps": float(np.nanmean(tp_list)) if tp_list else 0.0,
        "drop_rate_pct": float(np.nanmean(drop_list)) if drop_list else 0.0,
        "avg_latency_ms": float(np.mean(valid_lat)) if valid_lat else 0.0,
        "p99_latency_ms": float(np.percentile(valid_lat, 99)) if valid_lat else 0.0
    }


def evaluate_round_robin(chaos: bool):
    top_path, traffic_configs, rl_config, chaos_config = load_configs(chaos)
    topo = NetworkTopology(top_path)
    rt_api = RoutingTableAPI()
    sim = DataPlaneSimulator(topo, rt_api, traffic_configs, chaos_config=chaos_config)
    flow_ids = list(sim.generators.keys())
    engine = RoundRobinRoutingEngine(topo, rt_api)

    tp_list, drop_list, lat_list = [], [], []
    for _ in range(600):
        engine.update_routes(flow_ids, sim.flow_src_dst)
        for lq in sim.links.values():
            lq.reset_window_stats()
        for _ in range(10):
            sim.step(0.01)
        telem = sim.stats_provider.collect_window_telemetry(0.1)
        tp = sum((st.tx_bytes * 8.0 / 1_000_000.0) / 0.1 for st in telem.values())
        dr = float(np.mean([st.drop_rate_pct for st in telem.values()])) if telem else 0.0
        lat = packet_weighted_latency(telem)
        tp_list.append(tp)
        drop_list.append(dr)
        lat_list.append(lat)

    valid_lat = [l for l in lat_list if not np.isnan(l) and l > 0]
    return {
        "throughput_mbps": float(np.nanmean(tp_list)) if tp_list else 0.0,
        "drop_rate_pct": float(np.nanmean(drop_list)) if drop_list else 0.0,
        "avg_latency_ms": float(np.mean(valid_lat)) if valid_lat else 0.0,
        "p99_latency_ms": float(np.percentile(valid_lat, 99)) if valid_lat else 0.0
    }


def run_full_comparison(episodes: int, chaos: bool):
    console.print("\n[bold cyan]🚀 RUNNING FULL MULTI-ALGORITHM BENCHMARK...[/bold cyan]")
    dueling_m, dueling_telem = evaluate_rl_agent(use_dueling=True, episodes=episodes, chaos=chaos)
    standard_m, _ = evaluate_rl_agent(use_dueling=False, episodes=episodes, chaos=chaos)
    ospf_m = evaluate_ospf(chaos=chaos)
    rr_m = evaluate_round_robin(chaos=chaos)

    table = Table(title=f"📊 BENCHMARK COMPARISON ({'CHAOS ACTIVE' if chaos else 'NORMAL NETWORK'})")
    table.add_column("Metric", style="bold yellow")
    table.add_column("Dueling DQN (Proposed)", style="bold green")
    table.add_column("Standard DQN", style="bold cyan")
    table.add_column("Static OSPF", style="bold magenta")
    table.add_column("Round-Robin", style="bold blue")

    table.add_row(
        "Throughput (Mbps)",
        f"{dueling_m['throughput_mbps']:.1f}",
        f"{standard_m['throughput_mbps']:.1f}",
        f"{ospf_m['throughput_mbps']:.1f}",
        f"{rr_m['throughput_mbps']:.1f}"
    )
    table.add_row(
        "Packet Drop Rate (%)",
        f"{dueling_m['drop_rate_pct']:.2f}%",
        f"{standard_m['drop_rate_pct']:.2f}%",
        f"{ospf_m['drop_rate_pct']:.2f}%",
        f"{rr_m['drop_rate_pct']:.2f}%"
    )
    table.add_row(
        "Average Latency (ms)",
        f"{dueling_m['avg_latency_ms']:.2f} ms",
        f"{standard_m['avg_latency_ms']:.2f} ms",
        f"{ospf_m['avg_latency_ms']:.2f} ms",
        f"{rr_m['avg_latency_ms']:.2f} ms"
    )
    table.add_row(
        "Tail Latency P99 (ms)",
        f"{dueling_m['p99_latency_ms']:.2f} ms",
        f"{standard_m['p99_latency_ms']:.2f} ms",
        f"{ospf_m['p99_latency_ms']:.2f} ms",
        f"{rr_m['p99_latency_ms']:.2f} ms"
    )

    console.print(table)
    if dueling_telem:
        viz.print_link_telemetry_table(dueling_telem)


def main():
    os.system("clear" if os.name == "posix" else "cls")
    
    banner = Panel.fit(
        "[bold cyan]⚡ RL-SDN AUTOMATED CONTROL PLANE & PACKET SIMULATOR TUI ⚡[/bold cyan]\n"
        "[dim]Three-Layer RL Model Architectures & Baseline Comparison Interface[/dim]",
        border_style="bright_blue"
    )
    console.print(banner)

    console.print("\n[bold yellow]Select Algorithm / Mode to Run:[/bold yellow]")
    console.print("  [1] [bold green]Full Comparison[/bold green] (Dueling DQN vs Standard DQN vs OSPF vs Round-Robin)")
    console.print("  [2] [bold cyan]Dueling DQN Agent Only[/bold cyan]")
    console.print("  [3] [bold magenta]Standard DQN Agent Only[/bold magenta]")
    console.print("  [4] [bold blue]Static OSPF (Dijkstra Shortest Path)[/bold blue]")
    console.print("  [5] [bold yellow]Round-Robin (ECMP Load Balancer)[/bold yellow]")
    console.print("  [6] [bold red]Train Model & Export ONNX Policy[/bold red]")
    console.print("  [7] [bold white]Run Automated PyTest Suite[/bold white]")
    console.print("  [8] Exit\n")

    choice = Prompt.ask("Choose option", choices=["1", "2", "3", "4", "5", "6", "7", "8"], default="1")
    if choice == "8":
        console.print("[yellow]Exiting RL-SDN TUI. Goodbye![/yellow]")
        return

    if choice == "7":
        console.print("\n[bold cyan]🧪 Running Automated Test Suite...[/bold cyan]\n")
        pytest.main(["tests/", "-v"])
        return

    chaos = Confirm.ask("Enable Network Chaos Engine (flapping links, BER drops, delay jitter)?", default=True)
    episodes_str = Prompt.ask("Number of training / evaluation episodes", default="15")
    episodes = int(episodes_str)

    if choice == "1":
        run_full_comparison(episodes=episodes, chaos=chaos)

    elif choice == "2":
        m, telem = evaluate_rl_agent(use_dueling=True, episodes=episodes, chaos=chaos)
        viz.print_benchmark_table(m, {"throughput_mbps": 0, "drop_rate_pct": 0, "avg_latency_ms": 0, "p99_latency_ms": 0}, {"throughput_mbps": 0, "drop_rate_pct": 0, "avg_latency_ms": 0, "p99_latency_ms": 0})
        if telem:
            viz.print_link_telemetry_table(telem)

    elif choice == "3":
        m, telem = evaluate_rl_agent(use_dueling=False, episodes=episodes, chaos=chaos)
        viz.print_benchmark_table(m, {"throughput_mbps": 0, "drop_rate_pct": 0, "avg_latency_ms": 0, "p99_latency_ms": 0}, {"throughput_mbps": 0, "drop_rate_pct": 0, "avg_latency_ms": 0, "p99_latency_ms": 0})
        if telem:
            viz.print_link_telemetry_table(telem)

    elif choice == "4":
        m = evaluate_ospf(chaos=chaos)
        viz.print_benchmark_table({"throughput_mbps": 0, "drop_rate_pct": 0, "avg_latency_ms": 0, "p99_latency_ms": 0}, m, {"throughput_mbps": 0, "drop_rate_pct": 0, "avg_latency_ms": 0, "p99_latency_ms": 0})

    elif choice == "5":
        m = evaluate_round_robin(chaos=chaos)
        viz.print_benchmark_table({"throughput_mbps": 0, "drop_rate_pct": 0, "avg_latency_ms": 0, "p99_latency_ms": 0}, {"throughput_mbps": 0, "drop_rate_pct": 0, "avg_latency_ms": 0, "p99_latency_ms": 0}, m)

    elif choice == "6":
        onnx_file = Prompt.ask("Enter output ONNX filename", default="model.onnx")
        top_path, traffic_configs, rl_config, chaos_config = load_configs(chaos)
        controller = SDNController(top_path, traffic_configs, rl_config, chaos_config=chaos_config)
        controller.train_episodes(num_episodes=episodes, verbose=True)
        from rl_sdn_controller.ai.policy_exporter import export_policy_to_onnx
        export_policy_to_onnx(controller.agent.policy_net, controller.state_dim, onnx_file)
        console.print(f"[bold green]✅ Model exported to {onnx_file}[/bold green]")


if __name__ == "__main__":
    main()
