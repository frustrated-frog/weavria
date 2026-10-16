import pytest

from apps.finscope.config import FinScopeSettings


def test_parent_origin_rejects_paths() -> None:
    with pytest.raises(ValueError, match="exact origin"):
        FinScopeSettings(
            model_base_url="https://model.example/v1",
            model_name="model",
            model_api_key="",
            allowed_parent_origin="http://localhost:5173/widget",
        )
