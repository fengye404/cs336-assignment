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
        # 一个字典推导式，实际上就是把 vocab 的 kv 调转一下，因为它是 int2bytes 的    
        self.vocab_b2i = {value: key for key, value in self.vocab.items()}

    def encode(self, text: str) -> list[int]:
        # 和 train_bpe 一样，还是先按照 special tokens 切分一下
        
        # 先预分词，Tokenizer 的 encode 也是不能跨 pre-token 的
        # pre_tokens_bytes_list 的内容：[[b"a", b"b", b"d"],[b"a", b"b", b"d"],....]
        pre_tokens = regex.findall(PATTERN, text)
        pre_tokens_bytes_list = []
        for pre_token in pre_tokens:
            raw_bytes = pre_token.encode("utf-8")
            pre_tokens_bytes = []
            for pre_token_byte_int in raw_bytes:
                pre_tokens_bytes.append(bytes([pre_token_byte_int]))
            pre_tokens_bytes_list.append(pre_tokens_bytes)
        
        # 这里三层循环
        current_list_index = 0
        token_int_list = []
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
                token_int_list.append(self.vocab_i2b[pre_token])
                
            current_list_index += 1
            
        return token_int_list
        
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pass
    def decode(self, ids: list[int]) -> str:
        pass