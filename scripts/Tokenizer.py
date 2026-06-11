import json
import re
import numpy as np
from pathlib import Path

# =====================
# VOCAB
# =====================

with open(
    "datasets/vocab.json",
    "r",
    encoding="utf8"
) as f:
    vocab = json.load(f)

UNK = vocab["<UNK>"]

# =====================
# TOKENIZAÇÃO
# =====================

tokens = []

arquivo = Path(
    "datasets/wikipedia.txt"
)

print("Lendo wikipedia...")

with open(
    arquivo,
    "r",
    encoding="utf8",
    errors="ignore"
) as f:

    for i, linha in enumerate(f):

        palavras = re.findall(
            r"\w+",
            linha.lower()
        )

        ids = [
            vocab.get(
                palavra,
                UNK
            )
            for palavra in palavras
        ]

        tokens.extend(ids)

        if i % 10000 == 0:
            print(
                f"{len(tokens):,} tokens"
            )

# =====================
# SALVAR
# =====================

tokens = np.array(
    tokens,
    dtype=np.uint32
)

saida = Path(
    "datasets/tokens.bin"
)

tokens.tofile(saida)

print()
print("Finalizado")
print("Arquivo:", saida)
print("Tokens:", len(tokens))
print(
    "Tamanho:",
    round(
        saida.stat().st_size / 1024 / 1024,
        2
    ),
    "MB"
)