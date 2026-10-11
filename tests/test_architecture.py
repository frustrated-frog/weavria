"""Core may import only the standard library, Pydantic and itself."""

import ast
import sys
from pathlib import Path


def test_core_dependencies_stay_within_the_contract_layer() -> None:
    core_directory = Path(__file__).parents[1] / "src" / "weavria" / "core"
    for source in core_directory.rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    assert node.level == 1, f"{source}: relative import escapes core"
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
                    or module.startswith("weavria.core.")
                )
                assert allowed, f"{source}: forbidden core dependency {module}"
