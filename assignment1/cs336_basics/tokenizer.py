import regex
from collections.abc import Iterable, Iterator

PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        if special_tokens is None:
            special_tokens = []
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        # special token 最好校验一下，如果不在 vocab 里面，我们就给他手动加一下然后分配一个 id
        for sp_token in special_tokens:
            sp_token_bytes = sp_token.encode("utf-8")
            if(sp_token_bytes not in self.vocab.values()):
                self.vocab[max(self.vocab.keys()) + 1] = sp_token_bytes
                
        # 一个字典推导式，实际上就是把 vocab 的 kv 调转一下，因为它是 int2bytes 的    
        self.vocab_b2i = {value: key for key, value in self.vocab.items()}

    def encode(self, text: str) -> list[int]:
        token_int_list = []
        
        # 和 train_bpe 一样，还是先按照 special tokens 切分一下
        if len(self.special_tokens) == 0:
            split_list = [text]
        else:
            special_token_pattern = "|".join(
                regex.escape(special)
                for special in sorted(self.special_tokens, key=len, reverse=True)
            )
            split_list = regex.split(f"({special_token_pattern})", text)
            
        # 切分后的 split_list: ["Hi", "<|endoftext|>", " there"]
        # 对齐进行遍历，对每一个单独的结果进行 encode
        for part in split_list:
            if(part == ""):
                continue
            
            # 如果是 special token，就直接从 vocab 里面找到 id，然后 append 到最终结果里面
            if(part in self.special_tokens):
                token_int_list.append(self.vocab_b2i[part.encode("utf-8")])
            else:
                # 否则就是普通的正常文本
                # 先预分词，Tokenizer 的 encode 也是不能跨 pre-token 的
                # pre_tokens_bytes_list 的内容：[[b"a", b"b", b"d"],[b"a", b"b", b"d"],....]
                pre_tokens = regex.findall(PATTERN, part)
                pre_tokens_bytes_list = []
                for pre_token in pre_tokens:
                    raw_bytes = pre_token.encode("utf-8")
                    pre_tokens_bytes = []
                    for pre_token_byte_int in raw_bytes:
                        pre_tokens_bytes.append(bytes([pre_token_byte_int]))
                    pre_tokens_bytes_list.append(pre_tokens_bytes)
                
                # 这里三层循环
                current_list_index = 0
                while current_list_index < len(pre_tokens_bytes_list):
                    # 1. 先依次取出预分词的结果，取出来的内容 pre_token_bytes：[b"a", b"b", b"d"]
                    pre_token_bytes = pre_tokens_bytes_list[current_list_index]
                    for merge in self.merges:
                        # 2. 遍历 merges，一条 merge 规则的输入是一条 token 序列，输出也是一条新的 token 序列
                        new_bytes = []
                        i = 0
                        while i < len(pre_token_bytes) - 1: 
                            # 3. 从左到右扫描最新 token 序列，根据 merge 关系，拼装新的 token 序列
                            if merge == (pre_token_bytes[i], pre_token_bytes[i+1]):
                                new_bytes.append(pre_token_bytes[i]+pre_token_bytes[i+1])
                                i += 2
                            else:
                                new_bytes.append(pre_token_bytes[i])
                                i += 1
                        # 从左到右扫描 完 token 序列后，边界要处理下
                        # 例如 pre_token_bytes = [b"a", b"b", b"c"] merge = (b"a", b"b")
                        # merge 完了之后还剩最后一个 b"c" 就退出了，要手动添加到最新的 token 序列里面
                        if(i == len(pre_token_bytes) - 1):
                            new_bytes.append(pre_token_bytes[i])
                        # 新的 token 序列需要赋值，进入下一个 merge 规则的循环
                        pre_token_bytes = new_bytes
                        
                    # 2、3循环跑完后，pre_token_bytes 就是一个已经完整应用过所有 merges 的新的 token 序列了，此时就可以查词表转为 int 了
                    for pre_token in pre_token_bytes:
                        token_int_list.append(self.vocab_b2i[pre_token])
                        
                    current_list_index += 1
            
        return token_int_list
        
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for part in iterable:
            for token_id in self.encode(part):
                yield token_id
                
    def decode(self, ids: list[int]) -> str:
        bytes_list = []
        for token_id in ids:
            bytes_list.append(self.vocab[token_id])
        return b"".join(bytes_list).decode("utf-8", errors = "replace")