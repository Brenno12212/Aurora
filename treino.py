import json
import re
import torch
from pathlib import Path

from modelo.Aurora_lstm import AuroraLSTM


# =====================
# CONFIG
# =====================

SEQ_LEN = 512
BATCH_SIZE = 1024
EMBED_DIM = 512
HIDDEN_SIZE = 1024
NUM_LAYERS = 4
LEARNING_RATE = 0.001
SAVE_EVERY = 1000
DEVICE = "cpu"

torch.set_num_interop_threads(8)
torch.set_num_threads(8)

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

UNK = vocab["<UNK>"]


# =====================
# MODEL
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

checkpoint_path = (
    Path("AuroraV2")
    / "AuroraV2.2.pt"
)

checkpoint_path.parent.mkdir(
    exist_ok=True
)

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


# =====================
# TREINO
# =====================

step = 0

buffer = []

while True:

    with open(
        "datasets/wikipedia.txt",
        "r",
        encoding="utf8",
        errors="ignore"
    ) as arquivo:

        for linha in arquivo:

            palavras = re.findall(
                r"\w+",
                linha.lower()
            )

            ids = [
                vocab.get(
                    p,
                    UNK
                )
                for p in palavras
            ]

            buffer.extend(ids)

            while len(buffer) >= SEQ_LEN + 1:

                entrada = buffer[:SEQ_LEN]

                alvo = buffer[
                    1:SEQ_LEN + 1
                ]

                x = torch.tensor(
                    [entrada],
                    dtype=torch.long
                )

                y = torch.tensor(
                    [alvo],
                    dtype=torch.long
                )

                x = x.to(DEVICE)

                y = y.to(DEVICE)

                optimizer.zero_grad()

                output = model(x)

                loss = criterion(
                    output.reshape(
                        -1,
                        VOCAB_SIZE
                    ),
                    y.reshape(-1)
                )

                loss.backward()

                optimizer.step()

                step += 1

                if step % 100 == 0:

                    print(
                        f"Step {step}"
                        f" Loss {loss.item():.4f}"
                    )

                if step % SAVE_EVERY == 0:

                    torch.save(
                        {
                            "model":
                            model.state_dict(),

                            "optimizer":
                            optimizer.state_dict()
                        },
                        checkpoint_path
                    )

                    print(
                        "Checkpoint salvo."
                    )

                buffer = buffer[
                    SEQ_LEN:
                ]