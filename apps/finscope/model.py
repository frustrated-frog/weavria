"""HTTP transport for the generic OpenAI-compatible model normalizer."""

import httpx


class HTTPChatCompletionTransport:
    """Own HTTP authentication, endpoint addressing and client lifecycle."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str = "",
        timeout_seconds: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        value = base_url.rstrip("/")
        self._endpoint = (
            value if value.endswith("/chat/completions") else value + "/chat/completions"
        )
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._client = client or httpx.AsyncClient()
        self._owns_client = client is None

    async def complete(self, body: dict[str, object]) -> object:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = "Bearer " + self._api_key
        response = await self._client.post(
            self._endpoint,
            headers=headers,
            json=body,
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
