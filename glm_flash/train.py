
import torch
import torch.nn as nn
import torch.optim as optim

from model import LanguageModel

BATCH_SIZE = 4
SEQ_LEN = 64
LEARNING_RATE = 3e-4
STEPS = 2000
EVAL_INTERVAL = 100
EVAL_BATCHES = 20

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the dataset.
with open("input.txt", "r", encoding="utf-8") as file:
    text = file.read()

data = torch.tensor(
    list(text.encode("utf-8")),
    dtype=torch.long
)

# Reserve the final 5% for validation.
split_index = int(len(data) * 0.95)
train_data = data[:split_index]
val_data = data[split_index:]

def get_batch(source):
    positions = torch.randint(
        0,
        len(source) - SEQ_LEN - 1,
        (BATCH_SIZE,)
    )

    inputs = torch.stack([
        source[i:i + SEQ_LEN]
        for i in positions
    ])

    targets = torch.stack([
        source[i + 1:i + SEQ_LEN + 1]
        for i in positions
    ])

    return inputs.to(DEVICE), targets.to(DEVICE)

model = LanguageModel().to(DEVICE)

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

loss_function = nn.CrossEntropyLoss()

@torch.no_grad()
def estimate_loss():
    model.eval()
    results = {}

    for name, source in (
        ("train", train_data),
        ("validation", val_data)
    ):
        losses = []

        for _ in range(EVAL_BATCHES):
            inputs, targets = get_batch(source)
            logits = model(inputs)

            loss = loss_function(
                logits.reshape(-1, 256),
                targets.reshape(-1)
            )

            losses.append(loss.item())

        results[name] = sum(losses) / len(losses)

    model.train()
    return results

print("Device:", DEVICE)
print("Dataset: Tiny Shakespeare")
print("Training bytes:", len(train_data))
print("Validation bytes:", len(val_data))
print("Starting training...")

best_validation_loss = float("inf")

model.train()

for step in range(STEPS):
    inputs, targets = get_batch(train_data)

    optimizer.zero_grad()

    logits = model(inputs)

    loss = loss_function(
        logits.reshape(-1, 256),
        targets.reshape(-1)
    )

    loss.backward()
    optimizer.step()

    if step % EVAL_INTERVAL == 0 or step == STEPS - 1:
        losses = estimate_loss()

        print(
            f"Step {step:4d} | "
            f"Train loss: {losses['train']:.4f} | "
            f"Validation loss: {losses['validation']:.4f}"
        )

        if losses["validation"] < best_validation_loss:
            best_validation_loss = losses["validation"]

            torch.save(
                model.state_dict(),
                "model.pt"
            )

            print("  Saved new best model.")

print("Training finished!")
print("Best validation loss:", round(best_validation_loss, 4))
print("Best model saved to model.pt")