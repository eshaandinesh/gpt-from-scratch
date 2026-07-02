import torch

with open(r'C:\Users\eshaa\Downloads\gpt-from-scratch\data\input.txt', 'r', encoding='utf-8') as file:
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

def get_batch(block_size):
    x = train_data[:block_size]
    y = train_data[1:block_size + 1]

    return (x,y)

block_size = 8
x,y = get_batch(block_size)
# print(decode(x))
print()