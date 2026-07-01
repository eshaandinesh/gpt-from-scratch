import urllib.request
import os

os.makedirs("data", exist_ok=True)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "data/input.txt"
)