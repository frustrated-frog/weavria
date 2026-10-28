"""Optional, explicitly selected byte BPE; no import-time model/network discovery."""


class BPETokenizer:
    """Load a named tiktoken encoding when explicitly configured by the host."""

    def __init__(self, encoding: str) -> None:
        try:
            import tiktoken
        except ImportError as exc:
            raise ImportError("Install weavria[tokenization] to enable BPE accounting") from exc
        self._encoding = tiktoken.get_encoding(encoding)
        self.version = f"tiktoken/{tiktoken.__version__}/{encoding}"

    def count(self, text: str) -> int:
        # Source text resembling a control token must remain ordinary untrusted text.
        return len(self._encoding.encode_ordinary(text))

    def prefix_token_bound(self, text: str) -> int:
        """Bound a fixed prefix followed by arbitrary bytes without assuming BPE additivity.

        Only stable tokens survive arbitrary continuations. Charge each remaining
        byte as one token; control-token-looking source text remains ordinary data.
        """
        stable, _ = self._encoding.encode_with_unstable(text, disallowed_special=())
        prefix = self._encoding.decode_bytes(stable)
        encoded = text.encode()
        if not encoded.startswith(prefix):
            raise ValueError("tokenizer returned a non-prefix stable encoding")
        return len(stable) + len(encoded) - len(prefix)
