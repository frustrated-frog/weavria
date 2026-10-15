"""External FinScope adapter; Weavria packages never import this application."""

from .server import FinScopeSettings, create_app

__all__ = ["FinScopeSettings", "create_app"]
