"""Style classifier that learns to distinguish Marcelo's writing from generic text.

Uses DeBERTa-v3-small as the backbone with a binary classification head.
This model serves as the reward signal for GRPO training.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, PreTrainedModel


class StyleClassifier(nn.Module):
    """Binary text classifier for writing style detection.

    Architecture:
      DeBERTa-v3-small encoder → mean pooling → dropout → linear → sigmoid

    The output probability represents how likely a text is to be
    in Marcelo's writing style (1.0 = definitely Marcelo, 0.0 = generic).
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-small",
        dropout: float = 0.1,
        freeze_encoder_layers: int = 0,
        head_norm: bool = True,
    ):
        super().__init__()
        self.encoder: PreTrainedModel = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model_name = model_name
        self.dropout = dropout
        self.freeze_encoder_layers = freeze_encoder_layers
        self.head_norm = head_norm

        hidden_size = self.encoder.config.hidden_size
        # Mean-pooled hidden states come out with uneven per-dimension scale.
        # Feeding them straight into a linear layer makes the head's effective
        # learning rate depend on the backbone, which is what stalled training
        # when only the head was trainable. LayerNorm removes that coupling.
        head_layers: list[nn.Module] = [nn.LayerNorm(hidden_size)] if head_norm else []
        head_layers += [nn.Dropout(dropout), nn.Linear(hidden_size, 1)]
        self.classifier = nn.Sequential(*head_layers)

        # LayerNorm normalises each sample across its own dimensions, which is
        # not the same thing as putting the dimensions on a common scale: a
        # dimension that varies wildly across the corpus still dominates the
        # linear layer afterwards. Logistic regression on these very features
        # standardises per dimension and reached AUC 0.903 where the trained
        # head reached 0.827 on the same split. These buffers close that gap.
        # They stay at 0/1 until `set_feature_stats` fills them from the
        # training set, so an unfitted model behaves exactly as before.
        self.register_buffer("feature_mean", torch.zeros(hidden_size))
        self.register_buffer("feature_std", torch.ones(hidden_size))

        # DeBERTa-v3 shares the embedding with the MLM head; updating it during
        # downstream fine-tuning destabilises training and produces NaN weights
        # after a single AdamW step. Freeze it as the DeBERTa-v3 paper recommends.
        embeddings = getattr(self.encoder, "embeddings", None)
        if hasattr(embeddings, "word_embeddings"):
            embeddings.word_embeddings.weight.requires_grad_(False)

        # optionally freeze early encoder layers for faster training
        if freeze_encoder_layers > 0:
            layers = self.encoder.encoder.layer
            for layer in layers[:freeze_encoder_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

            # Freezing every transformer layer means the encoder is meant to be
            # a fixed feature extractor, but DeBERTa keeps parameters outside
            # them (relative position embeddings, LayerNorms). Leaving those
            # trainable defeats the point and blocks feature caching.
            if freeze_encoder_layers >= len(layers):
                for param in self.encoder.parameters():
                    param.requires_grad = False

    def mean_pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor):
        """Mean pooling over token embeddings, respecting the attention mask."""
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / sum_mask

    def set_feature_stats(self, features: torch.Tensor) -> None:
        """Fit the per-dimension standardiser on pooled training features."""
        self.feature_mean.copy_(features.mean(dim=0))
        # a dimension that never varies carries no information; clamping stops
        # it from being amplified into noise rather than dropped
        self.feature_std.copy_(features.std(dim=0).clamp(min=1e-6))

    def head(self, pooled: torch.Tensor) -> torch.Tensor:
        """Standardise pooled features, then score them. Returns logits."""
        standardised = (pooled - self.feature_mean) / self.feature_std
        return self.classifier(standardised).squeeze(-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        pos_weight: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Returns dict with 'logits', 'probs', and optionally 'loss'.
        pos_weight: passed to BCEWithLogitsLoss to correct for class imbalance.
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.mean_pool(outputs.last_hidden_state, attention_mask)
        logits = self.head(pooled)
        probs = torch.sigmoid(logits)

        result = {"logits": logits, "probs": probs}

        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            result["loss"] = loss_fn(logits, labels.float())

        return result

    def predict(self, texts: list[str], batch_size: int = 16) -> list[float]:
        """Score a list of texts. Returns probability of being Marcelo's style."""
        self.eval()
        all_probs = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {k: v.to(next(self.parameters()).device) for k, v in encoded.items()}

            with torch.no_grad():
                output = self.forward(**encoded)

            all_probs.extend(output["probs"].cpu().tolist())

        return all_probs

    @classmethod
    def from_pretrained(cls, path: str) -> StyleClassifier:
        """Load a trained classifier from disk."""
        load_path = Path(path)
        config = json.loads((load_path / "config.json").read_text(encoding="utf-8"))
        model = cls(
            model_name=config.get("model_name", "microsoft/deberta-v3-small"),
            dropout=config.get("dropout", 0.1),
            freeze_encoder_layers=config.get("freeze_encoder_layers", 0),
            # checkpoints written before the head LayerNorm existed have no flag
            head_norm=config.get("head_norm", False),
        )
        state_dict = torch.load(load_path / "model.pt", map_location="cpu", weights_only=True)
        # checkpoints written before per-dimension standardisation existed have
        # no such buffers; the identity values reproduce their old behaviour
        hidden_size = model.encoder.config.hidden_size
        state_dict.setdefault("feature_mean", torch.zeros(hidden_size))
        state_dict.setdefault("feature_std", torch.ones(hidden_size))
        model.load_state_dict(state_dict)
        return model

    def save_pretrained(self, path: str):
        """Save the trained classifier to disk."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        torch.save(self.state_dict(), save_path / "model.pt")
        config = {
            "model_name": self.model_name,
            "hidden_size": self.encoder.config.hidden_size,
            "dropout": self.dropout,
            "freeze_encoder_layers": self.freeze_encoder_layers,
            "head_norm": self.head_norm,
        }
        (save_path / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        self.tokenizer.save_pretrained(save_path)
