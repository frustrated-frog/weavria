from collections.abc import Callable

import pytest

from weavria.core import ModelResponse
from weavria.models import FakeModelProvider, ModelProvider


@pytest.fixture
def provider_factory() -> Callable[[tuple[ModelResponse, ...]], ModelProvider]:
    """Future providers reuse the suite by overriding this scripted factory."""
    return FakeModelProvider
