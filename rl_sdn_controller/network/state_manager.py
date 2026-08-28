from typing import Dict, Tuple, List
import numpy as np
from rl_sdn_controller.sdn_api.stats_provider import LinkStats


class StateManager:
    """
    Transforms SDN LinkStats telemetry into compact, normalized state observation vectors
    for Reinforcement Learning input.
    Features per link (6): [utilization_norm, queue_depth_norm, drop_rate_norm, avg_latency_norm, link_up, queue_trend]
    """
    def __init__(self, ordered_link_keys: List[Tuple[str, str]]):
        self.link_keys = ordered_link_keys
        self.num_features_per_link = 6
        self.state_dim = len(self.link_keys) * self.num_features_per_link
        self.prev_q_depths: Dict[Tuple[str, str], float] = {}

    def get_observation_vector(self, telemetry: Dict[Tuple[str, str], LinkStats]) -> np.ndarray:
        obs = []
        for key in self.link_keys:
            if key in telemetry:
                stats = telemetry[key]
                util_norm = stats.utilization_pct / 100.0
                q_depth_norm = stats.queue_depth / float(stats.max_queue_packets) if stats.max_queue_packets > 0 else 0.0
                drop_rate_norm = stats.drop_rate_pct / 100.0
                lat_norm = min(1.0, stats.avg_latency_ms / 50.0) # Normalized up to 50ms cap
                link_up = 1.0 if getattr(stats, 'is_up', True) else 0.0
                
                prev_q = self.prev_q_depths.get(key, q_depth_norm)
                q_trend = np.clip(q_depth_norm - prev_q, -1.0, 1.0)
                self.prev_q_depths[key] = q_depth_norm
            else:
                util_norm, q_depth_norm, drop_rate_norm, lat_norm, link_up, q_trend = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

            obs.extend([util_norm, q_depth_norm, drop_rate_norm, lat_norm, link_up, float(q_trend)])

        return np.array(obs, dtype=np.float32)
