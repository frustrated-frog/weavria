import httpx

from apps.finscope.model import HTTPChatCompletionTransport


async def test_http_model_transport_owns_endpoint_and_authentication() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://model.example/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer secret"
        return httpx.Response(200, json={"choices": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        transport = HTTPChatCompletionTransport(
            "https://model.example/v1",
            api_key="secret",
            client=client,
        )
        payload = await transport.complete({"model": "test", "messages": []})

    assert payload == {"choices": []}
