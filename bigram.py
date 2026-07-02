import torch

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
x, y = get_batch('train')
print(decode(x[0].tolist()))
print(decode(y[0].tolist()))