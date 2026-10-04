from typing import Any

import sys
import pytest
import shutil
import random
import numpy as np
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="function")
def temp_cleanup_on_success(request: Any) -> Path:
    """Per-test temp directory with cleanup on success only"""
    temp_name = f"temp_{request.node.name}"
    temp_path = Path("tests") / temp_name
    temp_path.mkdir(exist_ok=True)
    yield temp_path

    # Clean up ONLY if test passed (no timeouts)
    if hasattr(request.node, "failed") and not request.node.failed:
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture(scope="function")
def fixed_seed() -> int:
    """Ensure reproducible tests with fixed seeds"""
    original_seed = 42
    np.random.seed(original_seed)
    random.seed(original_seed)
    # Only set torch seed if torch is available
    try:
        import torch

        torch.manual_seed(original_seed)
    except ImportError:
        pass  # torch not available in test environment
    yield original_seed


@pytest.fixture(scope="function")
def zero_temperature() -> float:
    """Use zero temperature for deterministic outputs"""
    return 0.0


@pytest.fixture(scope="function")
def sample_engchi_data() -> dict[str, Any]:
    """Small subset of real engchi prompts for testing"""
    return {
        "sample_prompt": "<|im_start|>user\nEnglish: Hello world\nChinese:<|im_end|>",
        "overrides": {"temperature": 0.0, "max_tokens": 400, "seed": 42},
    }


@pytest.fixture(scope="function")
def engchi_fixture_path() -> Path:
    """Tracked engchi YAML fixture for parser tests"""
    return Path("tests/fixtures/engchi_sample.yml")


@pytest.fixture(scope="function")
def qwen_model_path() -> Path:
    """Path to existing Qwen3-0.6B model"""
    return Path("Qwen/Qwen3-0.6B")


def compare_floats(actual: float, expected: float, rel_tol: float = 1e-5, abs_tol: float = 1e-8) -> bool:
    """Compare float values with appropriate tolerance"""
    if abs(expected) < abs_tol:
        return abs(actual - expected) < abs_tol
    else:
        return abs(actual - expected) / abs(expected) < rel_tol
