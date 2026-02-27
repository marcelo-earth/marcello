"""Tests for the style classifier."""

import torch
import pytest


class TestStyleClassifier:
    """Test classifier architecture and forward pass.

    Note: these tests use a tiny random model to avoid downloading
    the full DeBERTa weights in CI. Integration tests with real
    weights should be run separately.
    """

    @pytest.fixture
    def mock_classifier(self):
        """Create a classifier with random weights for testing."""
        from marcello.classifier.model import StyleClassifier

        model = StyleClassifier.__new__(StyleClassifier)
        torch.nn.Module.__init__(model)

        # tiny encoder mock
        hidden_size = 32
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_size, 1),
        )
        return model, hidden_size

    def test_mean_pool(self, mock_classifier):
        model, hidden_size = mock_classifier

        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        attention_mask = torch.ones(batch_size, seq_len)
        # mask out last 3 tokens for first sample
        attention_mask[0, 7:] = 0

        pooled = model.mean_pool(hidden_states, attention_mask)

        assert pooled.shape == (batch_size, hidden_size)
        # first sample should only average over first 7 tokens
        expected_0 = hidden_states[0, :7].mean(dim=0)
        assert torch.allclose(pooled[0], expected_0, atol=1e-5)

    def test_classifier_output_shape(self, mock_classifier):
        model, hidden_size = mock_classifier
        pooled = torch.randn(4, hidden_size)
        logits = model.classifier(pooled).squeeze(-1)
        assert logits.shape == (4,)
