"""Tests for the data collection and processing pipeline."""

import json
import tempfile
from pathlib import Path

import pytest

from marcello.data.collector import WritingSampleCollector
from marcello.data.processor import TextProcessor
from marcello.data.negative_sampler import NegativeSampler, NegativeStrategy


class TestWritingSampleCollector:
    def test_collect_from_txt(self, tmp_path):
        txt_file = tmp_path / "sample.txt"
        txt_file.write_text(
            "This is the first paragraph with enough text to pass the minimum length filter.\n\n"
            "This is the second paragraph, also long enough to be considered a valid sample."
        )

        collector = WritingSampleCollector(min_length=20, max_length=500)
        count = collector.collect_from_txt(txt_file)

        assert count == 2
        assert len(collector) == 2
        assert all(s.source_type == "txt" for s in collector.samples)

    def test_collect_from_txt_filters_short(self, tmp_path):
        txt_file = tmp_path / "short.txt"
        txt_file.write_text("Too short.\n\nAlso short.")

        collector = WritingSampleCollector(min_length=50)
        count = collector.collect_from_txt(txt_file)

        assert count == 0

    def test_collect_from_jsonl(self, tmp_path):
        jsonl_file = tmp_path / "messages.jsonl"
        lines = [
            json.dumps({"text": "A long enough message that should be collected by the system.", "ts": 123}),
            json.dumps({"text": "Another message with sufficient length for the minimum threshold.", "ts": 456}),
        ]
        jsonl_file.write_text("\n".join(lines))

        collector = WritingSampleCollector(min_length=20)
        count = collector.collect_from_jsonl(jsonl_file)

        assert count == 2
        assert collector.samples[0].source_type == "jsonl"

    def test_collect_from_directory(self, tmp_path):
        (tmp_path / "a.txt").write_text("First file with enough content to pass the length filter easily.")
        (tmp_path / "b.txt").write_text("Second file also long enough to be a valid writing sample here.")

        collector = WritingSampleCollector(min_length=20)
        count = collector.collect_from_directory(tmp_path)

        assert count == 2

    def test_to_dataset(self, tmp_path):
        txt_file = tmp_path / "sample.txt"
        txt_file.write_text("A paragraph long enough to be collected as a valid writing sample by the system.")

        collector = WritingSampleCollector(min_length=20)
        collector.collect_from_txt(txt_file)
        dataset = collector.to_dataset()

        assert len(dataset) == 1
        assert "text" in dataset.column_names
        assert "label" in dataset.column_names
        assert dataset[0]["label"] == 1


class TestTextProcessor:
    def test_removes_urls(self):
        processor = TextProcessor(remove_urls=True)
        result = processor.clean("Check out https://example.com for more info.")
        assert "https://example.com" not in result
        assert "Check out" in result

    def test_removes_emails(self):
        processor = TextProcessor(remove_emails=True)
        result = processor.clean("Contact me at user@example.com for details.")
        assert "user@example.com" not in result

    def test_normalizes_whitespace(self):
        processor = TextProcessor(normalize_whitespace=True)
        result = processor.clean("Too   many    spaces   here")
        assert "  " not in result

    def test_deduplication(self):
        processor = TextProcessor(deduplicate=True)
        assert not processor.is_duplicate("First unique text")
        assert processor.is_duplicate("First unique text")
        assert not processor.is_duplicate("Second unique text")


class TestNegativeSampler:
    def test_shuffle_sentences_strategy(self):
        sampler = NegativeSampler(
            strategy=NegativeStrategy.SHUFFLE_SENTENCES,
            num_negatives_per_positive=1,
            seed=42,
        )

        texts = ["First sentence. Second sentence. Third sentence."]
        negatives = sampler.generate_negatives(texts)

        assert len(negatives) == 1
        assert negatives[0] != texts[0]  # shuffled should differ
