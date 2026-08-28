import time
import logging
from typing import Dict, Any, List

from rl_sdn_controller.ai.env import SDNEnv
from rl_sdn_controller.ai.agent import DQNAgent

logger = logging.getLogger(__name__)


class SDNController:
    """
    Main Orchestration Controller for RL-SDN System.
    Executes RL evaluation and training cycle every 10-100ms:
    1. Collect Telemetry
    2. RL Agent Forward Pass (Select Action)
    3. Update Routing Tables
    4. Compute Reward & Store Experience
    5. Perform Replay Optimization
    """
    def __init__(self, topology_path: str, traffic_configs: list, rl_config: Dict[str, Any], chaos_config: Dict[str, Any] = None):
        self.rl_config = rl_config
        self.env = SDNEnv(topology_path, traffic_configs, rl_config, chaos_config=chaos_config)
        self.agent = DQNAgent(self.env.state_dim, self.env.action_dim, rl_config.get("agent", {}))

    def train_episodes(self, num_episodes: int = 10, verbose: bool = True) -> List[Dict[str, Any]]:
        """
        Trains RL Agent over multiple simulation episodes.
        Returns metrics history for each episode.
        """
        history = []

        for ep in range(1, num_episodes + 1):
            state, _ = self.env.reset()
            total_reward = 0.0
            episode_losses = []
            episode_throughput = []
            episode_drops = []
            episode_latencies = []

            done = False
            while not done:
                # 1. Action selection
                action = self.agent.select_action(state, evaluate=False)

                # 2. Environment step (updates Layer 2 rules, advances simulation, collects telemetry)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # 3. Store experience tuple in replay memory
                self.agent.memory.push(state, action, reward, next_state, done)

                # 4. Perform Q-learning optimization step
                loss = self.agent.update()
                if loss > 0:
                    episode_losses.append(loss)

                state = next_state
                total_reward += reward

                episode_throughput.append(info["total_throughput_mbps"])
                episode_drops.append(info["avg_drop_pct"])
                episode_latencies.append(info["avg_latency_ms"])

            avg_loss = float(sum(episode_losses) / len(episode_losses)) if episode_losses else 0.0
            avg_tp = float(sum(episode_throughput) / len(episode_throughput)) if episode_throughput else 0.0
            avg_drop = float(sum(episode_drops) / len(episode_drops)) if episode_drops else 0.0
            avg_lat = float(sum(episode_latencies) / len(episode_latencies)) if episode_latencies else 0.0

            ep_stats = {
                "episode": ep,
                "total_reward": total_reward,
                "avg_loss": avg_loss,
                "avg_throughput_mbps": avg_tp,
                "avg_drop_pct": avg_drop,
                "avg_latency_ms": avg_lat,
                "epsilon": self.agent.epsilon
            }
            history.append(ep_stats)

            if verbose and (ep % 1 == 0 or ep == num_episodes):
                logger.info(
                    f"Episode {ep}/{num_episodes} | Reward: {total_reward:.2f} | "
                    f"Throughput: {avg_tp:.1f} Mbps | Drop: {avg_drop:.2f}% | Latency: {avg_lat:.2f} ms | Epsilon: {self.agent.epsilon:.3f}"
                )

        return history
