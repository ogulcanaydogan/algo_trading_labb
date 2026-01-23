"""
Policy Network for RL Trading Agent.

Neural network architecture for PPO-based trading policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# Try to import PyTorch, but work without it
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    F = None


@dataclass
class PolicyConfig:
    """Configuration for the policy network."""

    state_dim: int = 100
    action_dim: int = 3  # SHORT, FLAT, LONG
    hidden_dim: int = 128
    num_layers: int = 2
    use_lstm: bool = True
    lstm_hidden: int = 64
    dropout: float = 0.1
    activation: str = "relu"


if HAS_TORCH:

    class TradingPolicyNetwork(nn.Module):
        """
        PPO-style Actor-Critic policy network for trading.

        Architecture:
        - Shared encoder (LSTM or MLP)
        - Policy head (actor) - outputs action probabilities
        - Value head (critic) - outputs state value

        Features:
        - Supports LSTM for sequential state processing
        - Dropout for regularization
        - Separate heads for policy and value

        Usage:
            policy = TradingPolicyNetwork(config)
            action_probs, value = policy(state)
        """

        def __init__(self, config: Optional[PolicyConfig] = None):
            """
            Initialize the policy network.

            Args:
                config: Network configuration
            """
            super().__init__()
            self.config = config or PolicyConfig()

            # Activation function
            self.activation = {
                "relu": nn.ReLU,
                "tanh": nn.Tanh,
                "elu": nn.ELU,
            }.get(self.config.activation, nn.ReLU)

            # Build encoder
            if self.config.use_lstm:
                self.encoder = self._build_lstm_encoder()
                encoder_out_dim = self.config.lstm_hidden * 2  # Bidirectional
            else:
                self.encoder = self._build_mlp_encoder()
                encoder_out_dim = self.config.hidden_dim

            # Policy head (actor)
            self.policy_head = nn.Sequential(
                nn.Linear(encoder_out_dim, self.config.hidden_dim // 2),
                self.activation(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.hidden_dim // 2, self.config.action_dim),
            )

            # Value head (critic)
            self.value_head = nn.Sequential(
                nn.Linear(encoder_out_dim, self.config.hidden_dim // 2),
                self.activation(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.hidden_dim // 2, 1),
            )

            # Initialize weights
            self._init_weights()

        def _build_lstm_encoder(self) -> nn.Module:
            """Build LSTM encoder."""
            return nn.LSTM(
                input_size=self.config.state_dim,
                hidden_size=self.config.lstm_hidden,
                num_layers=self.config.num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=self.config.dropout if self.config.num_layers > 1 else 0,
            )

        def _build_mlp_encoder(self) -> nn.Module:
            """Build MLP encoder."""
            layers = []
            in_dim = self.config.state_dim

            for i in range(self.config.num_layers):
                out_dim = self.config.hidden_dim
                layers.extend(
                    [
                        nn.Linear(in_dim, out_dim),
                        self.activation(),
                        nn.Dropout(self.config.dropout),
                    ]
                )
                in_dim = out_dim

            return nn.Sequential(*layers)

        def _init_weights(self) -> None:
            """Initialize network weights."""
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=0.01)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.LSTM):
                    for name, param in module.named_parameters():
                        if "weight" in name:
                            nn.init.orthogonal_(param)
                        elif "bias" in name:
                            nn.init.zeros_(param)

        def forward(
            self,
            state: torch.Tensor,
            hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple]]:
            """
            Forward pass through the network.

            Args:
                state: Input state tensor (batch, state_dim) or (batch, seq, state_dim)
                hidden: Optional LSTM hidden state

            Returns:
                Tuple of (action_logits, value, new_hidden)
            """
            # Encode state
            if self.config.use_lstm:
                # Reshape for LSTM if needed
                if state.dim() == 2:
                    state = state.unsqueeze(1)  # (batch, 1, state_dim)

                encoded, new_hidden = self.encoder(state, hidden)
                # Use last output
                encoded = encoded[:, -1, :]  # (batch, hidden*2)
            else:
                encoded = self.encoder(state)
                new_hidden = None

            # Get policy (action logits)
            action_logits = self.policy_head(encoded)

            # Get value estimate
            value = self.value_head(encoded).squeeze(-1)

            return action_logits, value, new_hidden

        def get_action(
            self,
            state,
            deterministic: bool = False,
        ) -> Tuple[int, float, float]:
            """
            Get action from policy.

            Args:
                state: State tensor or numpy array
                deterministic: If True, return argmax action

            Returns:
                Tuple of (action, log_prob, value)
            """
            # Convert numpy to tensor if needed
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state)
            if state.dim() == 1:
                state = state.unsqueeze(0)

            with torch.no_grad():
                action_logits, value, _ = self.forward(state)

                # Get action probabilities
                action_probs = F.softmax(action_logits, dim=-1)

                if deterministic:
                    action = action_probs.argmax(dim=-1)
                else:
                    # Sample from distribution
                    dist = torch.distributions.Categorical(action_probs)
                    action = dist.sample()

                log_prob = F.log_softmax(action_logits, dim=-1)
                action_log_prob = log_prob.gather(-1, action.unsqueeze(-1)).squeeze(-1)

                return action.item(), action_log_prob.item(), value.item()

        def evaluate_actions(
            self,
            states,
            actions,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Evaluate actions for PPO update.

            Args:
                states: Batch of states (tensor or numpy)
                actions: Batch of actions taken (tensor or numpy)

            Returns:
                Tuple of (log_probs, values, entropy)
            """
            # Convert numpy to tensor if needed
            if isinstance(states, np.ndarray):
                states = torch.FloatTensor(states)
            if isinstance(actions, np.ndarray):
                actions = torch.LongTensor(actions)

            action_logits, values, _ = self.forward(states)

            # Action log probabilities
            log_probs = F.log_softmax(action_logits, dim=-1)
            action_log_probs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

            # Entropy for exploration bonus
            probs = F.softmax(action_logits, dim=-1)
            entropy = -torch.sum(probs * log_probs, dim=-1)

            return action_log_probs, values, entropy

        def save(self, path: str) -> None:
            """Save model weights."""
            torch.save(
                {
                    "config": self.config,
                    "state_dict": self.state_dict(),
                },
                path,
            )

        @classmethod
        def load(cls, path: str) -> "TradingPolicyNetwork":
            """Load model from file."""
            checkpoint = torch.load(path)
            model = cls(checkpoint["config"])
            model.load_state_dict(checkpoint["state_dict"])
            return model

else:
    # Fallback when PyTorch is not available
    class TradingPolicyNetwork:
        """Placeholder for TradingPolicyNetwork when PyTorch is unavailable."""

        def __init__(self, config: Optional[PolicyConfig] = None):
            self.config = config or PolicyConfig()
            self._weights = None

        def forward(self, state: np.ndarray) -> Tuple[np.ndarray, float, None]:
            """Simple forward pass without neural network."""
            # Random policy fallback
            action_logits = np.random.randn(3)
            value = 0.0
            return action_logits, value, None

        def get_action(
            self,
            state: np.ndarray,
            deterministic: bool = False,
        ) -> Tuple[int, float, float]:
            """Get random action."""
            if deterministic:
                action = 1  # FLAT
            else:
                action = np.random.randint(0, 3)
            return action, 0.0, 0.0

        def evaluate_actions(
            self,
            states: np.ndarray,
            actions: np.ndarray,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Return dummy values."""
            n = len(states)
            return np.zeros(n), np.zeros(n), np.zeros(n)

        def save(self, path: str) -> None:
            """Save placeholder."""
            import json

            with open(path, "w") as f:
                json.dump({"type": "placeholder"}, f)

        @classmethod
        def load(cls, path: str) -> "TradingPolicyNetwork":
            """Load placeholder."""
            return cls()
