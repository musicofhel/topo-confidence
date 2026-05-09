"""Tests for pipeline bug fixes and diagnostic guards.

Covers:
- FE-ID exact matching (Bugs 2, 3, 9)
- Claude CLI flag isolation (Bug 1)
- Git subprocess return code checks (Bug 4)
- Log routing (Bug 5)
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

import pytest


class TestParseFeNum:
    def test_standard(self):
        from pipeline._match import parse_fe_num
        assert parse_fe_num("P11-FE101") == "fe101"

    def test_bare(self):
        from pipeline._match import parse_fe_num
        assert parse_fe_num("FE42") == "fe42"

    def test_already_lowercase(self):
        from pipeline._match import parse_fe_num
        assert parse_fe_num("fe26841a") == "fe26841a"

    def test_numeric_only(self):
        from pipeline._match import parse_fe_num
        assert parse_fe_num("P11-101") == "fe101"


class TestMatchResultDir:
    def test_exact(self):
        from pipeline._match import match_result_dir
        assert match_result_dir("P11-FE101", "fe101") is True

    def test_with_suffix(self):
        from pipeline._match import match_result_dir
        assert match_result_dir("P11-FE101", "fe101_leace_erasure") is True

    def test_no_substring_fe42_fe421(self):
        from pipeline._match import match_result_dir
        assert match_result_dir("P11-FE42", "fe421_regularized") is False

    def test_no_substring_fe1_fe101(self):
        from pipeline._match import match_result_dir
        assert match_result_dir("P11-FE1", "fe101_leace") is False

    def test_no_substring_fe1_fe115(self):
        from pipeline._match import match_result_dir
        assert match_result_dir("P11-FE1", "fe115_song_zhong") is False

    def test_no_substring_fe15_fe115(self):
        from pipeline._match import match_result_dir
        assert match_result_dir("P11-FE15", "fe115_song_zhong") is False

    def test_no_substring_fe15_fe145(self):
        from pipeline._match import match_result_dir
        assert match_result_dir("P11-FE15", "fe145_per_layer") is False

    def test_fe15_matches_fe15_prefix(self):
        from pipeline._match import match_result_dir
        assert match_result_dir("P11-FE15", "fe15_length_band") is True

    def test_no_match_unrelated(self):
        from pipeline._match import match_result_dir
        assert match_result_dir("P11-FE101", "leace_erasure") is False

    def test_exact_num_only(self):
        from pipeline._match import match_result_dir
        assert match_result_dir("P11-FE101", "101") is True

    def test_no_num_substring_in_larger(self):
        from pipeline._match import match_result_dir
        assert match_result_dir("P11-FE42", "421") is False

    def test_mid_position_match(self):
        from pipeline._match import match_result_dir
        assert match_result_dir("P11-FE42", "foo_fe42_bar") is True

    def test_mid_position_no_substring(self):
        from pipeline._match import match_result_dir
        assert match_result_dir("P11-FE42", "foo_fe421_bar") is False


class TestClaudeCLIFlags:
    def test_generate_recompute_uses_bare(self):
        import pipeline.generate_recompute as gr
        source = inspect.getsource(gr.generate_for_fe)
        assert "--bare" in source
        assert '"--tools"' in source or "'--tools'" in source

    def test_call_claude_uses_bare(self):
        import pipeline.nodes as nodes
        source = inspect.getsource(nodes._call_claude)
        assert "--bare" in source
        assert '"--tools"' in source or "'--tools'" in source


class TestGitReturnCodes:
    @patch("pipeline.autopilot.subprocess.run")
    def test_auto_commit_returns_false_on_commit_failure(self, mock_run):
        from pipeline.autopilot import _auto_commit_results
        mock_run.side_effect = [
            MagicMock(stdout="M pathway11_h100/foo/results.json\n", returncode=0),
            MagicMock(returncode=0, stderr=""),
            MagicMock(returncode=1, stderr="error: commit failed"),
        ]
        assert _auto_commit_results("P11-FE101") is False

    @patch("pipeline.autopilot.subprocess.run")
    def test_auto_commit_returns_false_on_add_failure(self, mock_run):
        from pipeline.autopilot import _auto_commit_results
        mock_run.side_effect = [
            MagicMock(stdout="M pathway11_h100/foo/results.json\n", returncode=0),
            MagicMock(returncode=128, stderr="error: git add failed"),
        ]
        assert _auto_commit_results("P11-FE101") is False

    @patch("pipeline.autopilot.subprocess.run")
    def test_auto_commit_returns_true_on_nothing_to_commit(self, mock_run):
        from pipeline.autopilot import _auto_commit_results
        mock_run.return_value = MagicMock(stdout="", returncode=0)
        assert _auto_commit_results("P11-FE101") is True

    @patch("pipeline.autopilot.subprocess.run")
    def test_auto_commit_returns_true_on_success(self, mock_run):
        from pipeline.autopilot import _auto_commit_results
        mock_run.side_effect = [
            MagicMock(stdout="M pathway11_h100/foo/results.json\n", returncode=0),
            MagicMock(returncode=0, stderr=""),
            MagicMock(returncode=0, stderr=""),
            MagicMock(stdout="abc123def456\n", returncode=0),
        ]
        assert _auto_commit_results("P11-FE101") is True


class TestLogRouting:
    def test_generate_recompute_logger_name(self):
        import pipeline.generate_recompute as gr
        assert gr.log.name == "pipeline.generate_recompute"

    def test_match_logger_name(self):
        import pipeline._match as m
        assert m.log.name == "pipeline._match"
