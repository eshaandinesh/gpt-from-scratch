import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# config
block_size = 8
batch_size = 4
torch.manual_seed(1306)

# data
with open('data/input.txt', 'r', encoding='utf-8') as file:
    content = file.read()

chars = sorted(set(content))
vocab_size = len(chars)

stoi = {ch : i for i,ch in enumerate(chars)}
itos = {i : ch for i,ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join(itos[i] for i in l)

data = torch.tensor(encode(content), dtype = torch.long)
n = int(0.9 * len(content))
train_data = data[: n]
val_data = data[n:]

# testing to see the context and target
# x = train_data[:block_size]
# y = train_data[1:block_size + 1]
# for t in range(block_size):
#     context = x[:t+1]
#     target = y[t]
#     print(f"For input {context} target is {target}")

# batching
def get_batch(split):
    data = train_data if split == 'train' else val_data
    indx = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i: i + block_size] for i in indx])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in indx])
    return x, y

# testing batch output
# x, y = get_batch('train')
# print(decode(x[0].tolist()))
# print(decode(y[0].tolist()))

# model
class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)

        if targets is None:
            return logits, None
        
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[: , -1, :]
            probs = F.softmax(logits, dim = -1)
            idx_next = torch.multinomial(probs, num_samples = 1)
            idx = torch.cat((idx, idx_next), dim = 1)
        return idx

# evaluation
@torch.no_grad()
def estimate_loss(eval_iters=200):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# training    
model = BigramModel(vocab_size)
optimizer = optim.AdamW(model.parameters(), lr=1e-3) 
epochs = 10000

for epoch in range(epochs):
    if epoch % 1000 == 0:
        stats = estimate_loss()
        print(f"step {epoch}: train loss {stats['train']:.4f}, val loss {stats['val']:.4f}")
    
    x, y = get_batch('train')
    logits, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# generation
context = torch.zeros((1, 1), dtype=torch.long)
output = model.generate(context, max_new_tokens=200)
print(decode(output[0].tolist()))