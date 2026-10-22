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
