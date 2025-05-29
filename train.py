import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
import json

from model import GPT
from tokenizer import Tokenizer

num_training_samples = 1000

if os.path.exists('data.json'):
    print("Loading existing dataset...")
    with open('data.json', 'r', encoding='utf-8') as f:
        texts = json.load(f)
else:
    ds = load_dataset("HuggingFaceFW/fineweb", "CC-MAIN-2014-10",
                      split="train", streaming=True)

    texts = []

    for i, sample in enumerate(ds):
        texts.append(sample['text'])
        if i >= num_training_samples - 1:
            break

    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(texts, f)

print(f"Loaded {len(texts)} samples from dataset.")
max_block_size = 128
vocab_size = 300 + 3

SOS_TOKEN = vocab_size - 3
EOS_TOKEN = vocab_size - 2
PAD_TOKEN = vocab_size - 1

tokenizer = Tokenizer(vocab_size - 3)
tokenizer.train(texts[:10])
encode = tokenizer.encode


def decode_with_special_tokens(ids_list):
    filtered_ids = [id for id in ids_list if id not in [
        SOS_TOKEN, EOS_TOKEN, PAD_TOKEN]]
    if not filtered_ids:
        return ""
    return tokenizer.decode(filtered_ids)


decode = decode_with_special_tokens

texts_encoded = [encode(text)[:max_block_size-2] for text in texts]


class TextDataset(Dataset):
    def __init__(self, texts_encoded):
        self.texts = texts_encoded

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        if len(text) < max_block_size - 1:
            text += [PAD_TOKEN] * (max_block_size - len(text) - 1)
        x = [SOS_TOKEN] + text
        y = text + [EOS_TOKEN]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


dataset = TextDataset(texts_encoded)


def collate_fn(batch):
    x_batch, y_batch = zip(*batch)
    x_batch = torch.stack(x_batch)
    y_batch = torch.stack(y_batch)
    return x_batch, y_batch


dataloader = DataLoader(dataset, batch_size=8,
                        shuffle=True, collate_fn=collate_fn)


@torch.no_grad()
def sample(model, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
    model.eval()
    x = torch.tensor(idx, dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(max_new_tokens):
        if x.size(1) >= max_block_size:
            x = x[:, -max_block_size:]

        logits = model(x)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            top_k = min(top_k, logits.size(-1))
            top_k_logits, _ = torch.topk(logits, top_k)
            min_top_k = top_k_logits[:, -1].unsqueeze(-1)
            logits = torch.where(logits < min_top_k, torch.full_like(
                logits, float('-inf')), logits)

        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:,
                                     1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        x = torch.cat((x, next_token), dim=1)

        if next_token.item() == EOS_TOKEN:
            break

    return x[0].tolist()


model = GPT(
    vocab_size=vocab_size,
    embed_dim=128,
    ff_dim=512,
    num_layers=4,
    heads=8,
    dropout=0.1,
    max_seq_len=max_block_size,
    pad_token_id=PAD_TOKEN
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=100, eta_min=1e-6)

start_epoch = 0
checkpoint_path = 'checkpoint.pth'

if os.path.exists(checkpoint_path):
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resuming from epoch {start_epoch}")

print("Sampling before training...")
sample_output = decode(
    sample(model, [SOS_TOKEN], max_new_tokens=512, temperature=1.0))
print(f"Initial sample: {sample_output[:200]}...")

epochs = 100
best_loss = float('inf')

for epoch in range(start_epoch, epochs):
    model.train()
    pbar = tqdm(dataloader)
    total_loss = 0

    for batch_idx, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            y.view(-1),
            ignore_index=PAD_TOKEN
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        pbar.set_description(
            f"Epoch {epoch+1} | Loss {avg_loss:.4f} | LR {scheduler.get_last_lr()[0]:.2e}")

    scheduler.step()

    print(
        f"\nEpoch {epoch + 1}/{epochs} completed - Average Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        print(f"New best loss: {best_loss:.4f} - Saving best model...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_loss': best_loss,
        }, 'best_model.pth')

    print(f"Sampling after epoch {epoch + 1}...")
    print("Temperature 1.0:")
    print(decode(sample(model, [SOS_TOKEN],
          max_new_tokens=512, temperature=1.0)))
    print("\nTemperature 0.8:")
    print(decode(sample(model, [SOS_TOKEN],
          max_new_tokens=512, temperature=0.8)))
    print("\nTop-k 50:")
    print(decode(sample(model, [SOS_TOKEN],
          max_new_tokens=512, temperature=1.0, top_k=50)))

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, checkpoint_path)

print("\n" + "="*50)
print("Training complete!")
print(f"Best loss achieved: {best_loss:.4f}")
print(f"Final checkpoint saved to: {checkpoint_path}")
if best_loss < float('inf'):
    print(f"Best model saved to: best_model.pth")
print("="*50)
