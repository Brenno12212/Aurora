from datasets import load_dataset

dataset = load_dataset(
    "wikimedia/wikipedia",
    "20231101.pt",
    split="train"
)

print(dataset)

with open(
    "../datasets/wikipedia.txt",
    "w",
    encoding="utf-8"
) as f:

    for artigo in dataset:

        texto = artigo["text"]

        f.write(texto)
        f.write("\n")