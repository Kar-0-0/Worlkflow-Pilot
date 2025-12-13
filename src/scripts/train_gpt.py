import torch
import torch.nn as nn 
import torch.nn.functional as F
import sys
from pathlib import Path
import tiktoken

THIS_DIR = Path(__file__).resolve().parent           
ROOT = THIS_DIR.parents[1]                           
SRC_DIR = ROOT / "src"                               

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from workflow_pilot.gpt import GPT


with open('data/tiny_shakespeare.txt') as f:
    data = f.read()


enc = tiktoken.get_encoding("cl100k_base")
vocab_size = enc.n_vocab

def encode(x): return enc.encode(x)
def decode(x): return enc.decode(x)


context_length = 256
emb_dim = 256
n_head = 4
n_layers = 6
dropout = 0.2
epochs = 10_000
batch_size = 8
lr = 3e-4
betas = (0.9, 0.95)
weight_decay = 0.1
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

data = encode(data)
data = torch.tensor(data, device=device)
dat_len = len(data)
split_len = int(0.9 * dat_len)
x_train = data[:split_len]
x_test = data[split_len:]

def get_batch(batch_size, split='train'):
    x = x_train if split == 'train' else x_test
    ix = torch.randint(len(x)-context_length-1, (batch_size,), device=device)
    xb = torch.stack([x[i:i+context_length] for i in ix], dim=0)
    yb = torch.stack([x[i+1:i+1+context_length] for i in ix], dim=0)

    return xb, yb # (B, context_length)



model = GPT(n_layers, emb_dim, n_head, context_length, vocab_size)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

print()
for epoch in range(epochs):
    xb, yb = get_batch(batch_size, 'train') # (B, T)
    logits, loss = model(xb, yb) # (B, T, vocab_size)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 500 == 0 or epoch == 0:
        e = epoch+1
        print(f"Epoch {e} Loss: {loss.item()}")
        print("Generating...")
        print(decode(model.generate(torch.zeros((1, 1), dtype=torch.long, device=device), max_char=100)[0].tolist()))

torch.save(model.state_dict(), "gpt_v1.pth")

print("Final Generation")
print(decode(model.generate(torch.tensor([[enc.encode('.')[0]]], dtype=torch.long, device=device), max_char=200)[0].tolist()))
