import json
import numpy as np
import torch
from pathlib import Path

from modelo.Aurora_lstm import AuroraLSTM

# =====================
# CONFIG
# =====================

SEQ_LEN = 128
BATCH_SIZE = 16
EMBED_DIM = 512
HIDDEN_SIZE = 1024
NUM_LAYERS = 4
LEARNING_RATE = 0.001
SAVE_EVERY = 1000

DEVICE = "cpu"

torch.set_num_threads(6)
torch.set_num_interop_threads(6)

# =====================
# VOCAB
# =====================

with open(
    "datasets/vocab.json",
    "r",
    encoding="utf8"
) as f:

    vocab = json.load(f)

VOCAB_SIZE = len(vocab)

# =====================
# TOKENS
# =====================

print("Carregando tokens.bin...")

tokens = np.fromfile(
    "datasets/tokens.bin",
    dtype=np.uint32
)

print(
    f"{len(tokens):,} tokens carregados"
)

# =====================
# MODELO
# =====================

model = AuroraLSTM(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS
)

model.to(DEVICE)

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# =====================
# CHECKPOINT
# =====================

checkpoint_path = Path(
    "AuroraV3/AuroraV3.pt"
)

step = 0

if checkpoint_path.exists():

    print(
        "Carregando checkpoint..."
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer"]
    )

    step = checkpoint["step"]

# =====================
# BATCH
# =====================

def get_batch():

    posicoes = np.random.randint(
        0,
        len(tokens) - SEQ_LEN - 1,
        size=BATCH_SIZE
    )

    x_batch = []
    y_batch = []

    for pos in posicoes:

        x = tokens[
            pos:
            pos + SEQ_LEN
        ]

        y = tokens[
            pos + 1:
            pos + SEQ_LEN + 1
        ]

        x_batch.append(x)
        y_batch.append(y)

    x_batch = np.array(
        x_batch,
        dtype=np.int64
    )

    y_batch = np.array(
        y_batch,
        dtype=np.int64
    )

    return (
        torch.tensor(
            x_batch,
            dtype=torch.long,
            device=DEVICE
        ),
        torch.tensor(
            y_batch,
            dtype=torch.long,
            device=DEVICE
        )
    )

# =====================
# TREINO
# =====================

losses = []

print("\nTreino iniciado\n")

while True:

    x, y = get_batch()

    optimizer.zero_grad()

    output = model(x)

    loss = criterion(
        output.view(
            -1,
            VOCAB_SIZE
        ),
        y.view(-1)
    )

    loss.backward()

    optimizer.step()

    step += 1

    losses.append(
        loss.item()
    )

    if len(losses) > 100:
        losses.pop(0)

    if step % 100 == 0:

        avg_loss = (
            sum(losses)
            / len(losses)
        )

        print(
            f"Step {step}"
            f" AvgLoss {avg_loss:.4f}"
        )

    if step % SAVE_EVERY == 0:

        avg_loss = (
            sum(losses)
            / len(losses)
        )

        torch.save(
            {
                "step": step,
                "avg_loss": avg_loss,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            },
            checkpoint_path
        )

        print(
            f"Checkpoint salvo "
            f"(step {step})"
        )