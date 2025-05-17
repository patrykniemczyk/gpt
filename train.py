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
        return (len(self.data) - 1) // self.block_size

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


@torch.no_grad()
def sample(model, idx, max_new_tokens, temperature=1.0):
    model.eval()
    x = torch.tensor(idx, dtype=torch.long).unsqueeze(0).to(device)  # [1, T]

    for _ in range(max_new_tokens):
        logits = model(x)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]
        x = torch.cat((x, next_token), dim=1)

    return x[0].tolist()


model = GPT(
    vocab_size=vocab_size,
    embed_dim=128,
    ff_dim=512,
    num_layers=4,
    heads=8,
    dropout=0.1,
    max_seq_len=block_size
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

start_epoch = 0
checkpoint_path = 'checkpoint.pth'

if os.path.exists(checkpoint_path):
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resuming from epoch {start_epoch}")

print("Sampling before training...")
print(decode(sample(model, [0], max_new_tokens=100)))

epochs = 10

for epoch in range(start_epoch, epochs):
    pbar = tqdm(dataloader)
    total_loss = 0

    for x, y in pbar:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_description(
            f"Epoch {epoch+1} | Loss {total_loss / (pbar.n + 1):.4f}")

    print(f"Sampling after epoch {epoch + 1}...")
    print(decode(sample(model, [0], max_new_tokens=100)))

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)

print("Training complete.")
