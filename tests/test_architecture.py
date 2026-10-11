"""Subsystems use public Core contracts and cannot import applications or labs."""

import ast
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("package", ["core", "models", "context"])
def test_subsystem_dependencies_stay_within_their_layer(package: str) -> None:
    directory = Path(__file__).parents[1] / "src" / "weavria" / package
    for source in directory.rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    assert node.level == 1, f"{source}: relative import escapes subsystem"
                    continue
                modules = [node.module or ""]
            else:
                continue
            for module in modules:
                root = module.partition(".")[0]
                allowed = (
                    root in sys.stdlib_module_names
                    or root == "pydantic"
                    or module == "weavria.core"
                    or module == f"weavria.{package}"
                    or module.startswith(f"weavria.{package}.")
                )
                assert allowed, f"{source}: forbidden subsystem dependency {module}"
