import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple

from rl_sdn_controller.network.topology import NetworkTopology
from rl_sdn_controller.network.state_manager import StateManager
from rl_sdn_controller.network.routing_engine import RLRoutingEngine
from rl_sdn_controller.sdn_api.routing_table_api import RoutingTableAPI
from rl_sdn_controller.data_plane.simulator import DataPlaneSimulator


class SDNEnv(gym.Env):
    """
    Gymnasium Custom Environment for SDN Routing Policy Optimization.
    State: Aggregated telemetry vector (link utilization %, queue depth, drop rate, latency).
    Action: Discrete index representing path assignment policy for flows.
    Reward: R = alpha * throughput - beta * drop_rate - gamma * latency.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, topology_path: str, traffic_configs: list, rl_config: Dict[str, Any], chaos_config: Dict[str, Any] = None):
        super(SDNEnv, self).__init__()

        self.topology = NetworkTopology(topology_path)
        self.routing_table_api = RoutingTableAPI()
        self.simulator = DataPlaneSimulator(self.topology, self.routing_table_api, traffic_configs, chaos_config=chaos_config)
        
        self.flow_ids = list(self.simulator.generators.keys())
        self.flow_src_dst = self.simulator.flow_src_dst
        
        self.rl_routing_engine = RLRoutingEngine(
            self.topology,
            self.routing_table_api,
            self.flow_ids,
            self.flow_src_dst
        )

        self.state_manager = StateManager(self.topology.get_link_keys())
        self.state_dim = self.state_manager.state_dim

        # Calculate discrete action space dimension based on combinations of flow candidate paths
        num_actions = 1
        for f_id in self.flow_ids:
            num_actions *= len(self.rl_routing_engine.candidate_paths[f_id])

        self.action_dim = num_actions
        self.action_space = spaces.Discrete(self.action_dim)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.state_dim,), dtype=np.float32)

        # Config parameters
        ctrl_cfg = rl_config.get("control_plane", {})
        self.update_interval_sec = ctrl_cfg.get("update_interval_ms", 100) / 1000.0
        self.max_steps = ctrl_cfg.get("max_simulation_steps", 600)

        rw_cfg = rl_config.get("reward_function", {})
        self.w_throughput = rw_cfg.get("throughput_weight", 1.0)
        self.w_drop = rw_cfg.get("drop_rate_weight", -100.0)
        self.w_latency = rw_cfg.get("latency_weight", -0.1)

        self.current_step = 0

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        self.routing_table_api.clear_rules()
        self.simulator.reset()

        # Apply default initial routing rules
        self.rl_routing_engine.apply_action(0)
        # Warmup simulation step (100ms)
        self.simulator.step(self.update_interval_sec)

        telemetry = self.simulator.stats_provider.collect_window_telemetry(self.update_interval_sec)
        obs = self.state_manager.get_observation_vector(telemetry)
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step += 1

        # 1. Update Layer 2 Routing Tables based on RL Action
        self.rl_routing_engine.apply_action(action)

        # Reset telemetry window statistics before running interval
        for link_q in self.simulator.links.values():
            link_q.reset_window_stats()

        # 2. Run Data Plane Simulator for update_interval_sec
        # Execute in micro-substeps (e.g., 10 sub-steps of 10ms) for realistic queueing
        substeps = 10
        sub_dt = self.update_interval_sec / float(substeps)
        for _ in range(substeps):
            self.simulator.step(sub_dt)

        # 3. Collect window telemetry
        telemetry = self.simulator.stats_provider.collect_window_telemetry(self.update_interval_sec)
        obs = self.state_manager.get_observation_vector(telemetry)

        # 4. Compute Reward R
        total_tx_mbps = sum((st.tx_bytes * 8.0 / 1_000_000.0) / self.update_interval_sec for st in telemetry.values())
        avg_drop_pct = np.mean([st.drop_rate_pct for st in telemetry.values()]) if telemetry else 0.0
        avg_lat_ms = np.mean([st.avg_latency_ms for st in telemetry.values() if st.avg_latency_ms > 0]) if telemetry else 0.0

        reward = (self.w_throughput * total_tx_mbps) + (self.w_drop * (avg_drop_pct / 100.0)) + (self.w_latency * avg_lat_ms)

        terminated = self.current_step >= self.max_steps
        truncated = False

        info = {
            "step": self.current_step,
            "total_throughput_mbps": total_tx_mbps,
            "avg_drop_pct": avg_drop_pct,
            "avg_latency_ms": avg_lat_ms,
            "telemetry": telemetry
        }

        return obs, float(reward), terminated, truncated, info
