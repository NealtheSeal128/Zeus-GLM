import torch
import torch.nn as nn
import torch.optim as optim

from model import LanguageModel


BATCH_SIZE = 4
SEQ_LEN = 64
LEARNING_RATE = 3e-4
STEPS = 500

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


TEXT = """
The quick brown fox jumps over the lazy dog.
Language models learn by predicting the next token.
A transformer reads a sequence of tokens and learns
patterns between them. Training improves the model
by comparing its predictions with the correct answers.
""" * 100


data = torch.tensor(
    list(TEXT.encode("utf-8")),
    dtype=torch.long
)

def get_batch():
    positions = torch.randint(
        0,
        len(data) - SEQ_LEN - 1,
        (BATCH_SIZE,)
    )

    inputs = torch.stack([
        data[i:i + SEQ_LEN]
        for i in positions
    ])

    targets = torch.stack([
        data[i + 1:i + SEQ_LEN + 1]
        for i in positions
    ])

    return inputs.to(DEVICE), targets.to(DEVICE)

model = LanguageModel().to(DEVICE)

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

loss_function = nn.CrossEntropyLoss()


print("Device:", DEVICE)
print("Starting training...")


model.train()

for step in range(STEPS):

    inputs, targets = get_batch()

    optimizer.zero_grad()

    logits = model(inputs)

    loss = loss_function(
        logits.reshape(-1, 256),
        targets.reshape(-1)
    )

    loss.backward()

    optimizer.step()

    if step % 50 == 0:
        print(
            f"Step {step:4d} | Loss: {loss.item():.4f}"
        )


print("Training finished!")

torch.save(
    model.state_dict(),
    "model.pt"
)

print("Model saved to model.pt")