class Tokenizer:

    def __init__(self, vocab_size=2048, special_tokens={
        '<pad>': 256, '<unk>': 257, '<eos>': 258,
    }):

        self.vocab_size = vocab_size
        self.special_tokens = special_tokens if special_tokens else {}  # str -> int
        self.merges = {}  # (int, int) -> int

        self.vocab = {i: bytes([i]) for i in range(256)}  # int -> bytes

        for special_token, idx in self.special_tokens.items():
            if idx < 256:
                raise ValueError(f"Special token index {idx} must be >= 256")
            if idx in self.vocab:
                raise ValueError(
                    f"Special token index {idx} conflicts with existing vocab index")
            self.vocab[idx] = special_token.encode('utf-8')

        print(self.vocab)

    def train(self, texts):
        pass

    def encode(self, text):
        pass

    def decode(self, ids):
        pass
