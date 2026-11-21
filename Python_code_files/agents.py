from __future__ import annotations

import torch
from torch import nn


class EvidenceAgent(nn.Module):
    """A modality agent that predicts fake probability and an evidence reliability."""
    def __init__(self, input_dim: int = 256, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(hidden_dim, 1)
        self.reliability_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, sequence):
        # sequence: [batch, time, 256]
        encoded = self.encoder(sequence)
        # Mean + max pooling preserve both broad and salient evidence.
        pooled = 0.5 * encoded.mean(dim=1) + 0.5 * encoded.max(dim=1).values
        logit = self.classifier(pooled).squeeze(-1)
        reliability = self.reliability_head(pooled).squeeze(-1)
        return logit, reliability, pooled


class JudgeAgent(nn.Module):
    """Reliability-aware fusion and three-way decision (real/fake/uncertain)."""
    def __init__(self, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.judge = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 6, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, agent_outputs):
        logits = [x[0] for x in agent_outputs]
        reliabilities = [x[1] for x in agent_outputs]
        embeddings = [x[2] for x in agent_outputs]
        rel = torch.stack(reliabilities, dim=1).clamp(0.05, 0.95)
        probs = torch.stack([torch.sigmoid(x) for x in logits], dim=1)
        weighted_prob = (probs * rel).sum(dim=1) / rel.sum(dim=1).clamp_min(1e-6)
        judge_input = torch.cat(embeddings + [rel, probs], dim=1)
        judge_logit = self.judge(judge_input).squeeze(-1)
        # Residual fusion uses the learned judge plus reliability-weighted evidence.
        fused_logit = 0.5 * judge_logit + 0.5 * torch.logit(weighted_prob.clamp(1e-5, 1 - 1e-5))
        return fused_logit, rel, probs


class ProposedAVModel(nn.Module):
    def __init__(self, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.vv_agent = EvidenceAgent(hidden_dim=hidden_dim, dropout=dropout)
        self.aa_agent = EvidenceAgent(hidden_dim=hidden_dim, dropout=dropout)
        self.av_agent = EvidenceAgent(hidden_dim=hidden_dim, dropout=dropout)
        self.judge_agent = JudgeAgent(hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, batch):
        outputs = [
            self.vv_agent(batch["v"]),
            self.aa_agent(batch["a"]),
            self.av_agent(batch["av"]),
        ]
        fused_logit, reliability, agent_probs = self.judge_agent(outputs)
        return {
            "logit": fused_logit,
            "reliability": reliability,
            "agent_probs": agent_probs,
            "agent_logits": torch.stack([o[0] for o in outputs], dim=1),
        }
