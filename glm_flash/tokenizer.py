import torch

VOCAB_SIZE = 256

def encode(text):
    return torch.tensor(list(text.encode("utf-8")), dtype=torch.long)
def decode(tokens):
    return bytes(tokens.tolist()).decode("utf-8")
if __name__ == "__main__":
    text = "Hello, world!"
    tokens = encode(text)
    print("Tokens:", tokens)
    print("Shape:", tokens.shape)
    decoded = decode(tokens)
    print("Decoded:", decoded)