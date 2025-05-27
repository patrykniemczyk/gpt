def get_freqs(ids_list):
    freqs = {}
    for ids in ids_list:
        for pair in zip(ids[:-1], ids[1:]):
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1
    return freqs


def max_pair(freqs):
    max_freq = 0
    max_pair = None
    for pair, freq in freqs.items():
        if freq > max_freq:
            max_freq = freq
            max_pair = pair
    return max_pair, max_freq


def merge_pair(ids_list, pair, idx):
    new_ids_list = []
    for ids in ids_list:
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        new_ids_list.append(new_ids)
    return new_ids_list


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

    def train(self, texts):
        texts_bytes = [text.encode('utf-8') for text in texts]
        text_ids = [list(text) for text in texts_bytes]

        current_idx = max(self.vocab.keys()) + 1

        while current_idx < self.vocab_size:
            freqs = get_freqs(text_ids)
            if not freqs:
                break
            (pair, freq) = max_pair(freqs)
            if freq < 2:
                break
            text_ids = merge_pair(text_ids, pair, current_idx)
            self.merges[pair] = current_idx
            self.vocab[current_idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            current_idx += 1

        print(
            f"Training complete. Merges: {len(self.merges)}, Vocab size: {len(self.vocab)}")
        print("Special tokens:", self.special_tokens)
        print(self.merges)
        print(self.vocab)

    def encode(self, text):
        pass

    def decode(self, ids_list):
        pass


tokenizer = Tokenizer()
texts = ["Hello, world!", "This is a test.", "GPT is great!",
         "Let's tokenize this text.", "Special tokens are important."]
tokenizer.train(texts)
