import bpe

import regex

PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def main():
    bpe.train_bpe("../data/TinyStoriesV2-GPT4-valid.txt",100,"<end>")
    
    pass
    
    
if __name__ == "__main__":
    main()