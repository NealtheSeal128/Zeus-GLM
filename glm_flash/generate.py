import torch

from model import LanguageModel
from tokenizer import encode, decode


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = LanguageModel().to(DEVICE)

model.load_state_dict(
    torch.load(
        "model.pt",
        map_location=DEVICE
    )
)

model.eval()


def generate(prompt, max_new_tokens=100):
    tokens = encode(prompt).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        for _ in range(max_new_tokens):

            logits = model(tokens)

            next_token_logits = logits[:, -1, :]

            next_token = torch.argmax(
                next_token_logits,
                dim=-1,
                keepdim=True
            )

            tokens = torch.cat(
                [tokens, next_token],
                dim=1
            )

    return decode(tokens[0].cpu())


prompt = "The quick brown fox"

print("Prompt:", prompt)
print()
print("Generated text:")
print(generate(prompt))