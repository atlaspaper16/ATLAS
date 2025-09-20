import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPWithCommunityEmbedding(nn.Module):
    """MLP with optional community embeddings.
    - forward(x): returns log-probabilities (log_softmax) for single-label tasks (use with NLLLoss).
    - forward_logits(x): returns raw logits for multi-label tasks (apply sigmoid + BCELoss).
    """
    def __init__(self, input_dim, hidden_dim, out_dim, num_layers, dropout, batch_size):
        super().__init__()
        self.lins = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = dropout
        self.NormLayer = nn.LayerNorm
        self.lins.append(nn.Linear(input_dim, hidden_dim))
        self.norms.append(self.NormLayer(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.lins.append(nn.Linear(hidden_dim, hidden_dim))
            self.norms.append(self.NormLayer(hidden_dim))

        # Output layer (no norm here)
        self.lins.append(nn.Linear(hidden_dim, out_dim))

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, 'reset_parameters'):
                norm.reset_parameters()

    def _trunk(self, x: torch.Tensor) -> torch.Tensor:
        # Shared trunk for both paths
        for lin, norm in zip(self.lins[:-1], self.norms):
            x = lin(x)
            x = norm(x)
            x = F.gelu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Raw logits (before sigmoid). Use for multi-label with BCELoss."""
        x = self._trunk(x)
        return self.lins[-1](x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Log-probabilities (log_softmax). Use for single-label with NLLLoss."""
        x = self._trunk(x)
        return F.log_softmax(self.lins[-1](x), dim=-1)

