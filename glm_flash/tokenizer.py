
from pathlib import Path

DATASET_PATH = Path(__file__).with_name("input.txt")
UNKNOWN_TOKEN = "\0"

with open(DATASET_PATH, "r", encoding="utf-8") as file:
    TEXT = file.read()

CHARS = sorted(set(TEXT) | {UNKNOWN_TOKEN})

CHAR_TO_ID = {
    char: index
    for index, char in enumerate(CHARS)
}

ID_TO_CHAR = {
    index: char
    for char, index in CHAR_TO_ID.items()
}

VOCAB_SIZE = len(CHARS)


def encode(text):
   
    return [
        CHAR_TO_ID.get(char, CHAR_TO_ID[UNKNOWN_TOKEN])
        for char in text
    ]


def decode(tokens):

    return "".join(
        ID_TO_CHAR[int(token)]
        for token in tokens
        if ID_TO_CHAR[int(token)] != UNKNOWN_TOKEN
    )


if __name__ == "__main__":
    sample = "The quick brown fox"
    tokens = encode(sample)

    print("Vocabulary size:", VOCAB_SIZE)
    print("Sample text:", sample)
    print("Encoded tokens:", tokens)
    print("Decoded text:", decode(tokens))
    print("Round-trip successful:", decode(tokens) == sample)