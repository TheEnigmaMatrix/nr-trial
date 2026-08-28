from typing import Dict, Tuple, List
import numpy as np
from rl_sdn_controller.sdn_api.stats_provider import LinkStats


class StateManager:
    """
    Transforms SDN LinkStats telemetry into compact, normalized state observation vectors
    for Reinforcement Learning input.
    """
    def __init__(self, ordered_link_keys: List[Tuple[str, str]]):
        self.link_keys = ordered_link_keys
        # Each link produces 4 features: [utilization_norm, queue_depth_norm, drop_rate_norm, avg_latency_norm]
        self.num_features_per_link = 4
        self.state_dim = len(self.link_keys) * self.num_features_per_link

    def get_observation_vector(self, telemetry: Dict[Tuple[str, str], LinkStats]) -> np.ndarray:
        obs = []
        for key in self.link_keys:
            if key in telemetry:
                stats = telemetry[key]
                util_norm = stats.utilization_pct / 100.0
                q_depth_norm = stats.queue_depth / float(stats.max_queue_packets) if stats.max_queue_packets > 0 else 0.0
                drop_rate_norm = stats.drop_rate_pct / 100.0
                lat_norm = min(1.0, stats.avg_latency_ms / 50.0) # Normalized up to 50ms cap
            else:
                util_norm, q_depth_norm, drop_rate_norm, lat_norm = 0.0, 0.0, 0.0, 0.0

            obs.extend([util_norm, q_depth_norm, drop_rate_norm, lat_norm])

        return np.array(obs, dtype=np.float32)
