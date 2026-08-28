import random
import math
from typing import List, Dict, Any
from rl_sdn_controller.data_plane.packet import Packet


class TrafficFlowGenerator:
    """
    Generates synthetic network packets for a single flow using Poisson
    or Bursty ON/OFF stochastic arrival processes.
    """
    def __init__(self, flow_config: Dict[str, Any]):
        self.flow_id = flow_config["id"]
        self.src = flow_config["src"]
        self.dst = flow_config["dst"]
        self.pattern = flow_config.get("pattern", "poisson")
        self.rate_pps = flow_config.get("rate_pps", 100)
        self.packet_size_bytes = flow_config.get("packet_size_bytes", 1000)
        self.duration_sec = flow_config.get("duration_sec", 60.0)

        # Bursty config
        self.burst_on_sec = flow_config.get("burst_on_ms", 500) / 1000.0
        self.burst_off_sec = flow_config.get("burst_off_ms", 1000) / 1000.0
        self.is_burst_on = True
        self.last_burst_switch_time = 0.0

        self.seq_counter = 0
        self.next_arrival_time = 0.0

    def reset(self):
        self.seq_counter = 0
        self.next_arrival_time = 0.0
        self.is_burst_on = True
        self.last_burst_switch_time = 0.0

    def _sample_poisson_interval(self) -> float:
        if self.rate_pps <= 0:
            return 1.0
        # Exponential distribution for Poisson process arrivals
        return random.expovariate(self.rate_pps)

    def _sample_pareto_interval(self, alpha: float = 1.5) -> float:
        if self.rate_pps <= 0:
            return 1.0
        # Pareto distribution for heavy-tailed bursty arrivals
        mean_interval = 1.0 / self.rate_pps
        scale = mean_interval * (alpha - 1.0) / alpha
        return (random.paretovariate(alpha) - 1.0) * scale + (scale / 2.0)

    def generate_packets(self, current_time: float, delta_time: float) -> List[Packet]:
        """Generates packets arriving in the time range [current_time, current_time + delta_time]."""
        packets: List[Packet] = []
        if current_time > self.duration_sec:
            return packets

        if self.pattern == "bursty":
            # Update ON/OFF burst state
            time_in_state = current_time - self.last_burst_switch_time
            if self.is_burst_on and time_in_state >= self.burst_on_sec:
                self.is_burst_on = False
                self.last_burst_switch_time = current_time
            elif not self.is_burst_on and time_in_state >= self.burst_off_sec:
                self.is_burst_on = True
                self.last_burst_switch_time = current_time

            if not self.is_burst_on:
                return packets

        sample_func = self._sample_pareto_interval if self.pattern == "pareto" else self._sample_poisson_interval

        # Arrival generation
        end_time = current_time + delta_time
        if self.next_arrival_time < current_time:
            self.next_arrival_time = current_time + sample_func()

        while self.next_arrival_time < end_time:
            self.seq_counter += 1
            pkt = Packet(
                flow_id=self.flow_id,
                src=self.src,
                dst=self.dst,
                size_bytes=self.packet_size_bytes,
                creation_time=self.next_arrival_time,
                seq_id=self.seq_counter
            )
            packets.append(pkt)
            self.next_arrival_time += sample_func()

        return packets
