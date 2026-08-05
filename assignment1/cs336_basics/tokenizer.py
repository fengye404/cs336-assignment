import regex
from collections import Counter

PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""



class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

    def encode(self, text: str) -> list[int]:
        regex.findall(PATTERN, text)
        
        pass
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pass
    def decode(self, ids: list[int]) -> str:
        pass