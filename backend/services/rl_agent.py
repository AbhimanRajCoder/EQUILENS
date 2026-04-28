import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from typing import Dict, Any


# ---------------------------------------------------------------------------
# Gymnasium environment for bias-mitigation strategy selection
# ---------------------------------------------------------------------------
class BiasEnv(gym.Env):
    def __init__(self, initial_bias: float, initial_accuracy: float):
        super(BiasEnv, self).__init__()
        # Observation space: [bias_score, accuracy]
        self.observation_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        # Action space: 0: Remove Sensitive, 1: Reweight, 2: Threshold, 3: Fairness Constraint
        self.action_space = spaces.Discrete(4)
        
        self.initial_bias = initial_bias
        self.initial_accuracy = initial_accuracy
        self.state = np.array([initial_bias, initial_accuracy], dtype=np.float32)
        self.steps = 0
        self.max_steps = 10
        
        # Strategy names mapping
        self.strategies = [
            "Remove Sensitive Attribute",
            "Reweight Dataset",
            "Threshold Adjustment",
            "Fairness Constraint"
        ]
        
        # Simulated outcomes for training
        self.simulated_outcomes = {
            0: {"fairness_gain": 0.10, "accuracy_drop": 0.05},
            1: {"fairness_gain": 0.08, "accuracy_drop": 0.02},
            2: {"fairness_gain": 0.15, "accuracy_drop": 0.01},
            3: {"fairness_gain": 0.05, "accuracy_drop": 0.03}
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.state = np.array([self.initial_bias, self.initial_accuracy], dtype=np.float32)
        return self.state, {}

    def step(self, action):
        self.steps += 1
        outcome = self.simulated_outcomes[action]
        
        fairness_gain = outcome["fairness_gain"]
        accuracy_drop = outcome["accuracy_drop"]
        
        # Update state (clamped between 0 and 1)
        new_bias = max(0, min(1, self.state[0] - fairness_gain))  # Reducing bias
        new_accuracy = max(0, min(1, self.state[1] - accuracy_drop))
        self.state = np.array([new_bias, new_accuracy], dtype=np.float32)
        
        # Reward = fairness_gain - 0.5 * accuracy_drop
        reward = fairness_gain - 0.5 * accuracy_drop
        
        # Fairness score is 1 - bias
        fairness_score = 1 - self.state[0]
        
        terminated = bool(fairness_score > 0.85 or self.steps >= self.max_steps)
        truncated = False
        
        return self.state, reward, terminated, truncated, outcome


# ---------------------------------------------------------------------------
# Resolve model path relative to this file (backend/ directory)
# ---------------------------------------------------------------------------
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_PATH = os.path.join(_BACKEND_DIR, "ppo_bias_agent")


# ---------------------------------------------------------------------------
# Module-level model loading  —  runs once when the service is first imported
# ---------------------------------------------------------------------------

# Run scripts/pretrain_agent.py overnight before demo
if os.path.exists(f"{_MODEL_PATH}.zip"):
    print(f"[rl_agent] Loading pre-trained PPO from {_MODEL_PATH}.zip")
    _model = PPO.load(_MODEL_PATH)
else:
    print("[rl_agent] Pre-trained model not found — falling back to quick 500-step training.")
    _fallback_env = BiasEnv(initial_bias=0.2, initial_accuracy=0.8)
    _model = PPO("MlpPolicy", _fallback_env, verbose=0)
    _model.learn(total_timesteps=500)


# ---------------------------------------------------------------------------
# Public service class consumed by the /api/recommend router
# ---------------------------------------------------------------------------
class RLAgent:
    def __init__(self):
        self.strategies = [
            "Remove Sensitive Attribute",
            "Reweight Dataset",
            "Threshold Adjustment",
            "Fairness Constraint"
        ]
        self.reasons = [
            "Effective when the sensitive attribute is redundant or strongly correlated with other features.",
            "Best for correcting historical representation bias in the training data.",
            "Optimal for balancing outcomes without needing to retrain the underlying model.",
            "Strongest theoretical guarantee for fairness by explicitly penalizing disparity during training."
        ]
        # Re-use the module-level model — no training happens here
        self.model = _model

    def get_recommendation(self, bias_score: float, accuracy: float) -> Dict[str, Any]:
        """Return the best mitigation strategy for the given bias / accuracy state.
        
        No training occurs here — the model was loaded (or fallback-trained)
        at module import time.
        """
        obs = np.array([bias_score, accuracy], dtype=np.float32)
        action, _states = self.model.predict(obs, deterministic=True)
        action = int(action)
        
        # Calculate scores for all strategies based on Reward = Fairness Gain - 0.5 * Accuracy Drop
        dummy_env = BiasEnv(initial_bias=bias_score, initial_accuracy=accuracy)
        all_scores = []
        for i, strategy_name in enumerate(self.strategies):
            outcome = dummy_env.simulated_outcomes[i]
            score = outcome["fairness_gain"] - 0.5 * outcome["accuracy_drop"]
            all_scores.append({
                "strategy_name": strategy_name,
                "score": round(score, 4)
            })
            
        # Sort scores to find runner-up
        sorted_scores = sorted(all_scores, key=lambda x: x["score"], reverse=True)
        
        # The recommended strategy by the RL model might not always be the highest score 
        # (though it should be if trained well), but we'll follow the model's recommendation 
        # for the 'winner' and take the next best for 'runner-up'.
        recommended_strategy_name = self.strategies[action]
        
        # Filter out the recommended one to find the runner up from the rest
        remaining = [s for s in sorted_scores if s["strategy_name"] != recommended_strategy_name]
        runner_up_data = remaining[0]
        
        # Determine why the runner up lost compared to the recommended strategy
        rec_outcome = dummy_env.simulated_outcomes[action]
        ru_action = self.strategies.index(runner_up_data["strategy_name"])
        ru_outcome = dummy_env.simulated_outcomes[ru_action]
        
        if ru_outcome["accuracy_drop"] > rec_outcome["accuracy_drop"]:
            lost_reason = "higher accuracy cost"
        elif ru_outcome["fairness_gain"] < rec_outcome["fairness_gain"]:
            lost_reason = "lower fairness gain"
        else:
            lost_reason = "lower penalty-adjusted score"

        outcome = dummy_env.simulated_outcomes[action]
        
        return {
            "recommended_strategy": recommended_strategy_name,
            "expected_fairness_gain": outcome["fairness_gain"],
            "expected_accuracy_drop": outcome["accuracy_drop"],
            "reason": self.reasons[action],
            "all_scores": all_scores,
            "runner_up": {
                "strategy_name": runner_up_data["strategy_name"],
                "score": runner_up_data["score"],
                "reason": lost_reason
            }
        }
