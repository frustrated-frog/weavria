"""Run the adapter without importing an application at module import time."""

import uvicorn


def main() -> None:
    uvicorn.run(
        "apps.finscope.server:create_app",
        factory=True,
        host="127.0.0.1",
        port=8765,
    )


if __name__ == "__main__":
    main()
