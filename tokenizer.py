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
    if not freqs:
        return None, 0
    return max(freqs.items(), key=lambda x: x[1])


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

    def __init__(self, vocab_size=2048):

        self.vocab_size = vocab_size
        self.merges = {}  # (int, int) -> int
        self.vocab = {i: bytes([i]) for i in range(256)}  # int -> bytes

    def train(self, texts):
        texts_bytes = [text.encode('utf-8') for text in texts]
        text_ids = [list(text) for text in texts_bytes]

        current_idx = max(self.vocab.keys()) + 1

        while current_idx < self.vocab_size:
            freqs = get_freqs(text_ids)
            if not freqs:
                break
            pair, freq = max_pair(freqs)
            if pair is None or freq < 2:
                break
            text_ids = merge_pair(text_ids, pair, current_idx)
            self.merges[pair] = current_idx
            self.vocab[current_idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            current_idx += 1

    def encode(self, text):
        if not text:
            return []
        text_bytes = text.encode('utf-8')
        ids = list(text_bytes)

        while len(ids) > 1:
            freqs = get_freqs([ids])
            if not freqs:
                break
            pair = min(
                freqs.keys(), key=lambda x: self.merges.get(x, float('inf')))
            if pair not in self.merges:
                break
            ids = merge_pair([ids], pair, self.merges[pair])[0]

        return ids

    def decode(self, ids_list):
        text_bytes = b"".join(self.vocab[id] for id in ids_list)
        return text_bytes.decode('utf-8', errors='replace')
