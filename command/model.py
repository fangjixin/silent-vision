from __future__ import annotations


class AttentivePooling:
    def __init__(self, feature_dim: int) -> None:
        import torch
        from torch import nn

        self.module = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.Tanh(), nn.Linear(feature_dim, 1))
        self.softmax = torch.nn.Softmax(dim=1)

    def __call__(self, features):
        weights = self.softmax(self.module(features))
        return (features * weights).sum(dim=1)


class ConformerBlock:
    def __init__(self, feature_dim: int, num_heads: int, dropout: float) -> None:
        from torch import nn

        self.module = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=num_heads,
                dim_feedforward=feature_dim * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            ),
        )

    def __call__(self, features):
        return self.module(features)


def build_command_model(
    *,
    feature_dim: int,
    num_classes: int,
    num_layers: int = 4,
    num_heads: int = 4,
    dropout: float = 0.1,
):
    from torch import nn

    class _ConformerBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ffn1 = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, feature_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(feature_dim * 4, feature_dim),
            )
            self.self_attn_norm = nn.LayerNorm(feature_dim)
            self.self_attn = nn.MultiheadAttention(feature_dim, num_heads, dropout=dropout, batch_first=True)
            self.conv_norm = nn.LayerNorm(feature_dim)
            self.depthwise_conv = nn.Conv1d(feature_dim, feature_dim, kernel_size=7, padding=3, groups=feature_dim)
            self.pointwise_conv = nn.Conv1d(feature_dim, feature_dim, kernel_size=1)
            self.ffn2 = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, feature_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(feature_dim * 4, feature_dim),
            )
            self.final_norm = nn.LayerNorm(feature_dim)

        def forward(self, features):
            x = features + 0.5 * self.ffn1(features)
            attn_input = self.self_attn_norm(x)
            attn_output, _ = self.self_attn(attn_input, attn_input, attn_input, need_weights=False)
            x = x + attn_output
            conv_input = self.conv_norm(x).transpose(1, 2)
            conv_output = self.pointwise_conv(self.depthwise_conv(conv_input)).transpose(1, 2)
            x = x + conv_output
            x = x + 0.5 * self.ffn2(x)
            return self.final_norm(x)

    class _ConformerWrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_norm = nn.LayerNorm(feature_dim)
            self.layers = nn.ModuleList([_ConformerBlock() for _ in range(num_layers)])
            self.pool_score = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.Tanh(), nn.Linear(feature_dim, 1))
            self.classifier = nn.Linear(feature_dim, num_classes)

        def forward(self, features):
            x = self.input_norm(features)
            for layer in self.layers:
                x = layer(x)
            weights = torch.softmax(self.pool_score(x), dim=1)
            pooled = (x * weights).sum(dim=1)
            return self.classifier(pooled)

    return _ConformerWrapper()


class CommandConformerClassifier:
    def __init__(self, feature_dim: int, num_classes: int, num_layers: int = 4) -> None:
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.model = build_command_model(feature_dim=feature_dim, num_classes=num_classes, num_layers=num_layers)

    def load_checkpoint(self, checkpoint_path: str, device: str):
        import torch

        payload = torch.load(checkpoint_path, map_location=device)
        state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()
        return self
