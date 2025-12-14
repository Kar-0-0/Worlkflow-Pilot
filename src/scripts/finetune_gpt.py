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

enc = tiktoken.get_encoding("cl100k_base")
vocab_size = enc.n_vocab

def encode(x): return enc.encode(x)
def decode(x): return enc.decode(x)

context_length = 256
emb_dim = 896
n_head = 14
n_layers = 16
dropout = 0.2
epochs = 5
batch_size = 4
lr = 1e-4
betas = (0.9, 0.95)
weight_decay = 0.1
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

IGNORE_INDEX = -100

def encode_example(ex):
    instr = ex["instruction"].strip()
    inp = ex.get("input", "")
    out = ex["output"].strip()

    prompt = f"Instruction: {instr}\nInput:\n{inp}\nOutput:\n"
    full = prompt + out

    input_ids = encode(full)
    prompt_len = len(encode(prompt))

    labels = input_ids.copy()
    labels[:prompt_len] = [IGNORE_INDEX] * prompt_len  # loss only on output tokens
    return input_ids, labels



from torch.utils.data import Dataset, DataLoader  # torch provides this API [web:153]
import json


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
    

class JsonlSFTDataset(Dataset):
    def __init__(self, path):
        self.examples = load_jsonl(path)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return encode_example(self.examples[idx])  # returns (input_ids_list, labels_list)

def collate(batch):
    B = len(batch)
    Tfull = context_length + 1

    x_full = torch.full((B, Tfull), fill_value=0, dtype=torch.long)
    y_full = torch.full((B, Tfull), fill_value=IGNORE_INDEX, dtype=torch.long)

    for i, (inp, lab) in enumerate(batch):
        inp = inp[:Tfull]
        lab = lab[:Tfull]
        L = len(inp)
        x_full[i, :L] = torch.tensor(inp, dtype=torch.long)
        y_full[i, :L] = torch.tensor(lab, dtype=torch.long)

    xb = x_full[:, :-1]   # (B, context_length)
    yb = y_full[:, 1:]    # (B, context_length)
    return xb.to(device), yb.to(device)

def make_prompt(instruction: str, inp: str) -> str:
    # Matches your training format in encode_example()
    return f"Instruction: {instruction}\nInput:\n{inp}\nOutput:\n"

# --- Epoch test prompts (match training format + newlines) ---
test_prompt1 = make_prompt(
    "Summarize conditional and biconditional rules for exam prep.",
    "Conditional: $p → q$ false only if p true, q false\n\n"
    "Biconditional: $p ↔ q$ true if p, q same truth value\n\n"
    "Contrapositive: $¬q → ¬p$"
)

test_prompt2 = make_prompt(
    "Turn predicate logic examples into flashcards.",
    "$P(x)$: x > 3\n\n"
    "$Q(x, y)$: x = y + 3\n\n"
    "$Dog(x)$, $Loves(x, y)$"
)

test_prompt3 = make_prompt(
    "Explain universal quantification with an example.",
    ""  # empty input test
)

def run_test(prompt: str, max_new_tokens: int = 200):
    ids = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=device)  # (1, prompt_len)
    out_ids = model.generate(ids, max_char=max_new_tokens)[0].tolist()
    print(decode(out_ids))

train_loader = DataLoader(JsonlSFTDataset("data/dataset.jsonl"), batch_size=batch_size, shuffle=True, collate_fn=collate)


model = GPT(n_layers, emb_dim, n_head, context_length, vocab_size)
state = torch.load("gpt_300m_pretrain.pth", map_location=device)
model.load_state_dict(state)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total_params}")
for epoch in range(epochs):
    losses = []
    for i, (xb, yb) in enumerate(train_loader):
        if i % 100 == 0:
            print(f"{i}/{len(train_loader)}")
        logits, loss = model(xb, yb) # (B, T, vocab_size)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        e = epoch+1
        print(f"Epoch {e} Average Loss: {sum(losses)/len(losses)}")
        print("Generating...")
        print("------------------------------------- Test 1 -------------------------------------")
        run_test(test_prompt1)

        print("------------------------------------- Test 2 -------------------------------------")
        run_test(test_prompt2)

        print("------------------------------------- Test 3 -------------------------------------")
        run_test(test_prompt3)

    model.train()


torch.save(model.state_dict(), "gpt_300m_finetune.pth")

print("Final Generation")

print("------------------------------------- Final Test 1 -------------------------------------")
run_test(test_prompt1)

print("------------------------------------- Final Test 2 -------------------------------------")
run_test(test_prompt2)

print("------------------------------------- Final Test 3 -------------------------------------")
run_test(test_prompt3)
