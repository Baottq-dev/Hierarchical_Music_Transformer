"""
Evaluation utilities for AMT.
"""

from .evaluator import ModelEvaluator
try:
    from .evaluator import evaluate_batch
except ImportError:
    def evaluate_batch(*args, **kwargs):
        print("evaluate_batch function is not available")

from .metrics import EvaluationMetrics
from .tester import ModelTester
try:
    from .tester import test_model
except ImportError:
    def test_model(*args, **kwargs):
        print("test_model function is not available") 