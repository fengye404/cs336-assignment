import regex
from collections import Counter

PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def train_bpe(input_path, vocab_size, special_tokens):
    # 先读取数据
    with open(input_path, "rb") as f:
        corpus = f.read()        
    text = corpus.decode("utf-8")
    
    # 数据按照课程中给出的正则切块，然后统计切块后的出现次数
    # 需要排除掉special tokens
    spilt_list = []
    if len(special_tokens) == 0:
        spilt_list.append(text)
    else:
        special_token_pattern = "|".join([regex.escape(special) for special in special_tokens])
        spilt_list = regex.split(special_token_pattern,text)
    all_pretokens = []
    for spilt in spilt_list:
        all_pretokens.extend(regex.findall(PATTERN,spilt))
    pretoken_text_counts = Counter(all_pretokens)
    
    # 数据切分为字符
    token_sequence_counts = {}
    for key, value in pretoken_text_counts.items():
        raw_bytes = key.encode("utf-8")
        bytes_tokens = []
        for byte in raw_bytes:
            bytes_tokens.append(bytes([byte]))
        token_sequence_counts[tuple(bytes_tokens)] = value
    
    # 构建初始词表
    # 注意要把 special token 加进去
    merges = []
    vocab = {}
    for token_id in range(256):
        vocab[token_id] = bytes([token_id])
    for token_id, special_token in enumerate(special_tokens, start=256):
        vocab[token_id] = special_token.encode("utf-8")
        
    # 开始 merge
    current_vocab_size = len(vocab)
    while current_vocab_size < vocab_size:
        # 统计字符 pair 出现次数
        pair_counts = Counter()
        for seq, count in token_sequence_counts.items():
            for pair in zip(seq, seq[1:]):
                pair_counts[pair] += count
                
        # 找到出现次数最多的 pair
        if not pair_counts:
            break
        pair, count = find_max(pair_counts)
        
        # 合并 pair，更新词表
        target = pair[0] + pair[1]
        vocab[current_vocab_size] = target
        # merges 表也要更新
        merges.append(pair)
        
        # 还需要把 token_sequence_counts 也更新掉
        new_token_sequence_counts = {}
        for seq, count in token_sequence_counts.items():
            new_seq = []
            index = 0
            while(index < len(seq)):
                if(index + 1 < len(seq) and (seq[index] , seq[index+1]) == pair):
                    new_seq.append(target)
                    index += 2
                else:
                    new_seq.append(seq[index])
                    index += 1
            new_token_sequence_counts[tuple(new_seq)] = count
        token_sequence_counts = new_token_sequence_counts
        
        # 词表大小加一
        current_vocab_size += 1
    
    return vocab, merges
    
def find_max(pair_counts):
    return max(
        pair_counts.items(),
        key = lambda item: (item[1], item[0])
    )