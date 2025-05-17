import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from model import GPT

text = open("data.txt", "r", encoding="utf-8").read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocab size: {vocab_size}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s]


def decode(l):
    return "".join([itos[i] for i in l])


class CharDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        import math
        return math.ceil((len(self.data) - 1) / self.block_size)

    def __getitem__(self, idx):
        start = idx * self.block_size
        end = start + self.block_size
        x = torch.tensor(self.data[start:end], dtype=torch.long)
        y = torch.tensor(self.data[start + 1:end + 1], dtype=torch.long)
        return x, y


block_size = 128
data = encode(text)
dataset = CharDataset(data, block_size)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
