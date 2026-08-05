import regex
from collections import Counter

PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def train_bpe(input_path, vocab_size, special_tokens):
    # 以 bytes 读取语料，之后显式按 UTF-8 解码，避免默认编码影响结果。
    with open(input_path, "rb") as f:
        corpus = f.read()
    text = corpus.decode("utf-8")

    # special token 是文档边界：不把它的内部字符交给预分词或 BPE 统计。
    # 这里先按照 special 把原始文本分开成 spilt list
    spilt_list = []
    if len(special_tokens) == 0:
        spilt_list.append(text)
    else:
        special_token_pattern = "|".join([regex.escape(special) for special in special_tokens])
        spilt_list = regex.split(special_token_pattern, text)
    # 再对拆开来的每一项，按照作业里面的预分词器（实际上就是一段正则）进行预分词
    # 为什么要预分词，可以看这个：https://zhuanlan.zhihu.com/p/692508797
    all_pretokens = []
    for spilt in spilt_list:
        all_pretokens.extend(regex.findall(PATTERN, spilt))
    # 先统计下预分词后，每个词的频率，例如 "low" -> 5。
    pretoken_text_counts = Counter(all_pretokens)

    # 将每个 pre-token 从字符串转为 utf-8 的 bytes，例如 "low"-> b"low"
    # 然后再拆成独立的
    # token_sequence_counts 里面的内容：(b"l", b"o", b"w") -> 5
    token_sequence_counts = {}
    for key, value in pretoken_text_counts.items():
        raw_bytes = key.encode("utf-8")
        bytes_tokens = []
        # bytes 实际上是 int 的数组，所以这里取出来的是 int，还需要手动创建一个 bytes
        for byte_int in raw_bytes:
            bytes_tokens.append(bytes([byte_int]))
        token_sequence_counts[tuple(bytes_tokens)] = value

    # 初始词表包含全部 256 个可能的单 byte 值。
    # special token 作为一个完整 token 加入 vocab，不拆成内部的 byte token。
    merges = []
    vocab = {}
    for token_id in range(256):
        vocab[token_id] = bytes([token_id])
    for token_id, special_token in enumerate(special_tokens, start=256):
        vocab[token_id] = special_token.encode("utf-8")

    # 每轮只学习一种最高频 pair；新 token 会让 vocab 的大小增加 1。
    current_vocab_size = len(vocab)
    while current_vocab_size < vocab_size:
        # 基于「当前」token 序列统计相邻 pair 的加权次数。
        # 例如某序列出现 5 次，其中的每个相邻 pair 都贡献 5 次，而不是 1 次。
        pair_counts = Counter()
        for seq, count in token_sequence_counts.items():
            for pair in zip(seq, seq[1:]):
                pair_counts[pair] += count

        # 没有相邻 pair 时，说明无法继续学习新的 merge。
        if not pair_counts:
            break
        # 找最高频 pair；同频时由 find_max 按字典序选择更大的 pair。
        pair, count = find_max(pair_counts)

        # 两个 bytes token 拼接成一个新 bytes token，例如 b"l" + b"o" -> b"lo"。
        target = pair[0] + pair[1]
        vocab[current_vocab_size] = target
        # merges 必须保留创建顺序；编码时会按这个顺序应用规则。
        merges.append(pair)

        # 不能边遍历边修改旧表：读取旧 token_sequence_counts，
        # 将 merge 后的结果写入新表，全部完成后再整体替换。
        new_token_sequence_counts = {}
        for seq, count in token_sequence_counts.items():
            new_seq = []
            index = 0
            while index < len(seq):
                if index + 1 < len(seq) and (seq[index], seq[index + 1]) == pair:
                    # 命中时消费两个旧 token，写入一个新 token；因此不会重叠合并。
                    new_seq.append(target)
                    index += 2
                else:
                    # 未命中时保留当前 token，继续检查下一个位置。
                    new_seq.append(seq[index])
                    index += 1
            new_token_sequence_counts[tuple(new_seq)] = count
        token_sequence_counts = new_token_sequence_counts

        # 本轮新增了一个 vocab token，下一轮将基于更新后的序列重新统计 pair。
        current_vocab_size += 1

    return vocab, merges


def find_max(pair_counts):
    return max(pair_counts.items(), key=lambda item: (item[1], item[0]))
