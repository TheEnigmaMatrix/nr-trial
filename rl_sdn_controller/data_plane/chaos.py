import random
import logging
from typing import Dict, Tuple, Any, List

logger = logging.getLogger(__name__)


class ChaosEngine:
    """
    Stochastic Network Chaos Engine.
    Simulates physical network disruptions: link flapping, latency jitter,
    random packet loss, and latency spikes.
    """
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config.get("chaos", {}) if config else {}
        self.enabled = self.config.get("enabled", True)

        self.link_failure_prob = self.config.get("link_failure_probability", 0.05)
        self.repair_time_sec = self.config.get("link_repair_time_sec", 2.0)
        self.jitter_std_sec = self.config.get("jitter_std_us", 20.0) / 1_000_000.0
        self.spike_prob = self.config.get("latency_spike_probability", 0.02)
        self.spike_mult = self.config.get("latency_spike_multiplier", 3.0)
        self.random_drop_rate = self.config.get("random_drop_rate", 0.01)

        # Operational state of links: (src, dst) -> {"is_up": bool, "down_until": float}
        self.link_states: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def register_links(self, link_keys: List[Tuple[str, str]]):
        for key in link_keys:
            self.link_states[key] = {"is_up": True, "down_until": 0.0}

    def reset(self):
        """Resets link states back to operational at simulation t=0."""
        for state in self.link_states.values():
            state["is_up"] = True
            state["down_until"] = 0.0

    def is_link_up(self, src: str, dst: str) -> bool:
        if not self.enabled:
            return True
        state = self.link_states.get((src, dst))
        return state["is_up"] if state else True

    def update(self, current_time: float, delta_time: float):
        """Updates link outage and recovery states."""
        if not self.enabled:
            return

        for key, state in self.link_states.items():
            # Only trigger failure outages on inter-router core links (which have candidate backup paths)
            if key[0].startswith('h') or key[1].startswith('h'):
                continue

            if not state["is_up"]:
                if current_time >= state["down_until"]:
                    state["is_up"] = True
                    logger.info(f"⚡ [CHAOS RECOVERY] Link {key[0]}->{key[1]} recovered at t={current_time:.2f}s")
            else:
                # Random failure check
                if random.random() < (self.link_failure_prob * delta_time):
                    state["is_up"] = False
                    state["down_until"] = current_time + self.repair_time_sec
                    logger.warning(f"💥 [CHAOS FAILURE] Link {key[0]}->{key[1]} FAILED at t={current_time:.2f}s for {self.repair_time_sec}s")

    def sample_delay(self, base_latency_sec: float) -> float:
        """Applies delay jitter and latency spikes."""
        if not self.enabled:
            return base_latency_sec

        # Apply Gaussian delay jitter
        jitter = random.gauss(0.0, self.jitter_std_sec)
        delay = max(0.000001, base_latency_sec + jitter)

        # Apply latency spike
        if random.random() < self.spike_prob:
            delay *= self.spike_mult

        return delay

    def should_drop_packet(self) -> bool:
        """Determines if a packet is lost due to random physical noise/BER."""
        if not self.enabled:
            return False
        return random.random() < self.random_drop_rate
