# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Callable

import pytest
import torch
from trt_test_utils import build_tiny


@pytest.fixture
def tiny_detector() -> Callable[..., torch.nn.Module]:
    """Factory fixture: ``tiny_detector(name, device='cpu')``."""
    return build_tiny
