import torch 
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, emb_dim, n_head, context_length, dropout=0.2):
        super().__init__()
        self.qkv = nn.Linear(emb_dim, emb_dim*3)
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)).view(1, 1, context_length, context_length))
        self.dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.head_size = emb_dim // n_head
        self.proj_out = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv(x) # (B, T, C*3)
        q, k, v = qkv.split(C, dim=2) # (B, T, C)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2) # (B, nh, T, hs)

        scale = 1/self.head_size**0.5
        scores = (q @ k.transpose(-2, -1)) * scale # (B, nh, T, T)
        scores = scores.masked_fill(self.tril[:, :, :T, :T] == 0, float('-inf'))
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)

        out = scores @ v # (B, nh, T, hs)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj_out(out)

        return out # (B, T, C)


class FeedForward(nn.Module):
    def __init__(self, emb_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, emb_dim*4),
            nn.GELU(),
            nn.Linear(emb_dim*4, emb_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        out = self.net(x)

        return out # (B, T, C)


class Block(nn.Module):
    def __init__(self, emb_dim, n_head, context_length):
        super().__init__()
        self.attn = CausalSelfAttention(emb_dim, n_head, context_length)
        self.l1 = nn.LayerNorm(emb_dim)
        self.ffwd = FeedForward(emb_dim)
        self.l2 = nn.LayerNorm(emb_dim)

    def forward(self, x):
        h = x + self.attn(self.l1(x))
        h = h + self.ffwd(self.l2(h))

        return h # (B, T, C)


class GPT(nn.Module):
    def __init__(self, n_layers, emb_dim, n_head, context_length, vocab_size):
        super().__init__()
        self.context_length = context_length
        self.tok_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(context_length, emb_dim)
        self.transformer = nn.Sequential(*[Block(emb_dim, n_head, context_length) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(emb_dim)
        self.linear = nn.Linear(emb_dim, vocab_size)
    
    def forward(self, x, targets=None):
        tok_embs = self.tok_emb(x) # (B, T, C)
        pos = torch.arange(x.size(1), device=x.device)
        pos_embs = self.pos_emb(pos)[None, :, :]

        x = pos_embs + tok_embs # (B, T, C)
        x = self.transformer(x)
        x = self.ln(x)

        logits = self.linear(x)

        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets, ignore_index=-100)
        else:
            loss = None

        return logits, loss # (B, T, vocab_size)

    def generate(self, ids, max_char, temperature=0.7, top_k=50):
        _, T = ids.shape
        for _ in range(max_char):
            logits, _ = self(ids[:, -self.context_length:]) # (B, T, vocab_size)
            logits = logits[:, -1, :]
            logits = logits / temperature
            v, _ = torch.topk(logits, k=top_k, dim=-1)
            cutoff = v[:, -1].unsqueeze(-1)
            logits = logits.masked_fill(logits < cutoff, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            new_id = torch.multinomial(probs, dim=-1, keepdim=True)
            ids = torch.cat([ids, new_id], dim=1)
        
        return ids

