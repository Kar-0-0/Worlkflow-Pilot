import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent           
ROOT = THIS_DIR.parents[1]                           
SRC_DIR = ROOT / "src"                               

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from workflow_pilot.gpt import GPT

enc = tiktoken.get_encoding("cl100k_base")


context_length = 256
emb_dim = 256
n_head = 4
n_layers = 6
vocab_size = enc.n_vocab
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


model = GPT(n_layers, emb_dim, n_head, context_length, vocab_size)
state = torch.load("gpt_v1.pth", map_location=device)
model.load_state_dict(state)
model.to(device)
model.eval()

conversation_ids = []
while True:
    prompt = input("What would you like to say?: ")
    if prompt == 'quit' or prompt == 'exit':
        break

    user_text = f"User: {prompt}\nAssistant: "
    user_ids = enc.encode(user_text)

    conversation_ids.extend(user_ids) 

    input_ids = conversation_ids[-context_length:]
    input_tensor = torch.tensor(input_ids, device=device)[None, :]

    out_ids = model.generate(input_tensor, max_char=200)[0].tolist()

    new_ids = out_ids[len(input_ids):]

    reply = enc.decode(new_ids)
    
    print(reply)

    conversation_ids.extend(new_ids)





