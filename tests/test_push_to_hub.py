"""Tests for push_to_hub.py — exercises dry-run paths without network calls."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import push_to_hub as pth  # noqa: E402


class TestBuildParser:
    def test_defaults(self):
        parser = pth.build_parser()
        args = parser.parse_args([])
        assert args.org == "marcelo-earth"
        assert args.classifier_path == "outputs/classifier/best"
        assert args.model_path == "outputs/grpo/final"
        assert not args.dry_run
        assert not args.merge_weights

    def test_all_flag_enables_all_pushes(self):
        parser = pth.build_parser()
        args = parser.parse_args(["--all"])
        assert args.all is True


class TestDryRunClassifier:
    def test_dry_run_prints_without_uploading(self, capsys):
        args = pth.build_parser().parse_args(
            ["--classifier", "--dry-run", "--classifier-path", "outputs/classifier/best"]
        )
        api = MagicMock()

        pth.push_classifier_artifact(api, args, token=None)

        api.create_repo.assert_not_called()
        api.upload_folder.assert_not_called()
        out = capsys.readouterr().out
        assert "would upload" in out


class TestDryRunModel:
    def test_dry_run_prints_without_uploading(self, capsys):
        args = pth.build_parser().parse_args(
            ["--model", "--dry-run", "--model-path", "outputs/grpo/final"]
        )
        api = MagicMock()

        pth.push_model_artifact(api, args, token=None)

        api.create_repo.assert_not_called()
        api.upload_folder.assert_not_called()

    def test_dry_run_merge_weights_flag(self, capsys):
        args = pth.build_parser().parse_args(["--model", "--dry-run", "--merge-weights"])
        api = MagicMock()

        pth.push_model_artifact(api, args, token=None)

        out = capsys.readouterr().out
        assert "merge" in out.lower()


class TestDryRunDataset:
    def test_dry_run_counts_samples(self, tmp_path, capsys):
        for i in range(3):
            (tmp_path / f"poem_{i}.txt").write_text(f"Sample poem number {i}.")

        args = pth.build_parser().parse_args(
            [
                "--dataset",
                "--dry-run",
                "--samples-path",
                str(tmp_path),
                "--samples-blog-path",
                str(tmp_path / "nonexistent"),
            ]
        )
        api = MagicMock()

        pth.push_dataset_artifact(api, args, token=None)

        out = capsys.readouterr().out
        assert "3" in out


class TestCheckAuth:
    def test_exits_when_not_logged_in(self):
        import pytest

        with patch("push_to_hub.whoami", side_effect=Exception("not logged in")):
            with pytest.raises(SystemExit):
                pth.check_auth(token=None)
