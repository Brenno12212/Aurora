import json
import re
import torch

from modelo.Aurora_lstm import AuroraLSTM


# ====================
# CONFIG
# ====================

DEVICE = "cpu"
EMBED_DIM = 512
HIDDEN_SIZE = 1024
NUM_LAYERS = 4
SEQ_LEN = 128
MAX_NOVAS_PALAVRAS = 50
TEMPERATURE = 0.8

torch.set_num_interop_threads(6)
torch.set_num_threads(6)

# ====================
# VOCAB
# ====================

with open(
    "datasets/vocab.json",
    "r",
    encoding="utf8"
) as f:

    vocab = json.load(f)

id_to_word = {
    v: k
    for k, v in vocab.items()
}

VOCAB_SIZE = len(vocab)

UNK = vocab["<UNK>"]
PAD = vocab["<PAD>"]


# ====================
# MODELO
# ====================

model = AuroraLSTM(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS
)

checkpoint = torch.load(
    "AuroraV3/AuroraV3.pt",
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model"]
)

model.eval()


# ====================
# GERAÇÃO
# ====================

def gerar_texto(prompt, max_palavras=30):

    palavras = re.findall(
        r"\w+",
        prompt.lower()
    )

    for _ in range(max_palavras):

        ids = [
            vocab.get(
                palavra,
                UNK
            )
            for palavra in palavras
        ]

        ids = ids[-SEQ_LEN:]

        while len(ids) < SEQ_LEN:
            ids.insert(0, PAD)

        x = torch.tensor(
            [ids],
            dtype=torch.long
        )

        with torch.no_grad():

            output = model(x)

            ultimo = output[0, -1]

            probs = torch.softmax(
                ultimo / TEMPERATURE,
                dim=0
            )

            top_probs, top_indices = torch.topk(
                probs,
                20
            )

            escolha = torch.multinomial(
                top_probs,
                1
            )

            token_id = top_indices[
                escolha
            ].item()

        palavra = id_to_word.get(
            token_id,
            "<UNK>"
        )

        if palavra in (
            "<PAD>",
            "<UNK>"
        ):
            continue

        palavras.append(
            palavra
        )

    return " ".join(
        palavras
    )


# ====================
# CHAT
# ====================

print()
print("Aurora carregada.")
print("Digite 'sair' para encerrar.")

while True:

    texto = input(
        "\nVocê: "
    )

    if texto.lower() == "sair":
        break

    resposta = gerar_texto(
        texto,
        MAX_NOVAS_PALAVRAS
    )

    print()
    print(
        "Aurora:",
        resposta
    )