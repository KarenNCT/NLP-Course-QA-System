import os
import pickle
import pandas as pd
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

chunks_path = os.path.join(BASE_DIR, "chunks.csv")
index_path = os.path.join(BASE_DIR, "faiss.index")
metadata_path = os.path.join(BASE_DIR, "metadata.pkl")

print("BASE_DIR =", BASE_DIR)

if not os.path.exists(chunks_path):
    raise FileNotFoundError(
        f"找不到 chunks.csv: {chunks_path}"
    )

print("讀取 chunks...")
df = pd.read_csv(chunks_path)

texts = df["chunk"].astype(str).tolist()

print("chunk數量:", len(texts))

print("載入 BGE 模型...")
model = SentenceTransformer(
    "BAAI/bge-m3"
)

print("開始 embedding...")

embeddings = model.encode(
    texts,
    show_progress_bar=True,
    normalize_embeddings=True
)

embeddings = np.array(
    embeddings
).astype("float32")

print("建立 FAISS...")

dim = embeddings.shape[1]

index = faiss.IndexFlatIP(dim)

index.add(embeddings)

print("寫入 FAISS index...")

faiss.write_index(
    index,
    index_path
)

print("寫入 metadata...")

with open(metadata_path, "wb") as f:
    pickle.dump(
        df.to_dict("records"),
        f
    )

print("完成")
print("dimension =", dim)
print("vectors =", index.ntotal)
print("saved index =", index_path)
print("saved metadata =", metadata_path)