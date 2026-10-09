
import torch

from model import LanguageModel
from tokenizer import encode, decode

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = LanguageModel().to(DEVICE)

model.load_state_dict(
    torch.load("model.pt", map_location=DEVICE)
)

model.eval()


def generate(
    prompt,
    max_new_tokens=100,
    temperature=0.8,
    top_k=20
):
    tokens = encode(prompt).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Keep the input within the model's context limit.
            context = tokens[:, -1024:]

            logits = model(context)
            logits = logits[:, -1, :]

            # Temperature controls randomness.
            logits = logits / temperature

            # Keep only the top-k candidate tokens.
            k = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, k)
            threshold = values[:, -1].unsqueeze(-1)

            logits[logits < threshold] = float("-inf")

            probabilities = torch.softmax(logits, dim=-1)

            next_token = torch.multinomial(
                probabilities, num_samples=1
            )

            tokens = torch.cat(
                [tokens, next_token], dim=1
            )

    return decode(tokens[0].cpu())


prompt = "The quick brown fox"

print("Prompt:", prompt)
print()
print("Generated text:")
print(generate(prompt))