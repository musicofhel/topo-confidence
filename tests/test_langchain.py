"""Tests for TopoEvaluator (no model loading — mock-based)."""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np


class TestTopoEvaluatorImport:
    def test_import_without_langchain(self):
        """Should import without langchain installed."""
        from topo_confidence.integrations.langchain import TopoEvaluator

        assert TopoEvaluator is not None

    def test_evaluation_name(self):
        from topo_confidence.integrations.langchain import TopoEvaluator

        evaluator = TopoEvaluator.__new__(TopoEvaluator)
        assert evaluator.evaluation_name == "topo_confidence"

    def test_requires_input(self):
        from topo_confidence.integrations.langchain import TopoEvaluator

        evaluator = TopoEvaluator.__new__(TopoEvaluator)
        assert evaluator.requires_input is True

    def test_requires_reference(self):
        from topo_confidence.integrations.langchain import TopoEvaluator

        evaluator = TopoEvaluator.__new__(TopoEvaluator)
        assert evaluator.requires_reference is False
