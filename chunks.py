import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(
    BASE_DIR,
    "documents.csv"
)

df = pd.read_csv(csv_path)

chunks = []

for _, row in df.iterrows():

    text = str(row["text"])

    chunk_size = 500

    for i in range(0, len(text), chunk_size):

        chunk = text[i:i + chunk_size]

        chunks.append({
            "source": row["source"],
            "page": row["page"],
            "chunk": chunk
        })

chunk_df = pd.DataFrame(chunks)

output_path = os.path.join(
    BASE_DIR,
    "chunks.csv"
)

chunk_df.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("chunks:", len(chunk_df))
print("saved:", output_path)