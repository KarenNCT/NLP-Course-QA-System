import os
import pickle
import pandas as pd
import numpy as np
import faiss
import torch
import chardet

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

input_path = os.path.join(BASE_DIR, "input.csv")
output_path = os.path.join(BASE_DIR, "output.csv")
index_path = os.path.join(BASE_DIR, "faiss.index")
metadata_path = os.path.join(BASE_DIR, "metadata.pkl")


def detect_encoding(file_path):
    with open(file_path, "rb") as f:
        raw_data = f.read()

    result = chardet.detect(raw_data)

    encoding = result["encoding"]
    confidence = result["confidence"]

    print("detected encoding =", encoding)
    print("confidence =", confidence)

    return encoding


def read_input_csv(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到 input.csv：{file_path}")

    encoding = detect_encoding(file_path)

    encodings_to_try = []

    if encoding:
        encodings_to_try.append(encoding)

    encodings_to_try += [
        "utf-8-sig",
        "utf-8",
        "utf-16",
        "cp950",
        "big5"
    ]

    tried = set()

    for enc in encodings_to_try:
        if enc in tried:
            continue

        tried.add(enc)

        try:
            df = pd.read_csv(
                file_path,
                header=None,
                encoding=enc,
                engine="python"
            )

            print("成功讀取 input.csv")
            print("encoding =", enc)

            return df

        except Exception as e:
            print(f"讀取失敗 encoding={enc}: {e}")

    raise Exception("無法讀取 input.csv，請確認檔案格式")


print("讀取 input.csv...")
input_df = read_input_csv(input_path)

questions = input_df[0].astype(str).tolist()

print("載入 BGE-M3...")
embed_model = SentenceTransformer("BAAI/bge-m3")


print("載入 FAISS...")
index = faiss.read_index(index_path)

with open(metadata_path, "rb") as f:
    metadata = pickle.load(f)


print("載入 Qwen...")
llm_name = "Qwen/Qwen2.5-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(llm_name)

llm = AutoModelForCausalLM.from_pretrained(
    llm_name,
    device_map="auto",
    torch_dtype="auto"
)


def answer_question(query, k=3):
    q_emb = embed_model.encode(
        [query],
        normalize_embeddings=True
    )

    q_emb = np.array(q_emb).astype("float32")

    D, I = index.search(q_emb, k=k)

    contexts = []

    for idx in I[0]:
        item = metadata[idx]

        contexts.append(
            f"[Source: {item['source']}, Page: {item['page']}]\n"
            f"{item['chunk']}"
        )

    context_text = "\n\n".join(contexts)

    prompt = f"""
You are a QA system for an NLP course.

Answer the question using ONLY the provided context.

Rules:
1. Use only the context.
2. Answer in one sentence only.
3. Do not copy the context directly.
4. If the answer is not in the context, say:
   "I don't know based on the course materials."
5. Chinese question -> Traditional Chinese answer.
6. English question -> English answer.
7. For date questions, answer in YYYY/MM/DD format if possible.

Context:
{context_text}

Question:
{query}

Answer:
"""

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(llm.device)

    outputs = llm.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None
    )

    answer = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )

    return answer.strip()


answers = []

for i, question in enumerate(questions, start=1):
    print(f"\n[{i}/{len(questions)}] {question}")

    answer = answer_question(question)

    print("答案：", answer)

    answers.append(answer)


output_df = pd.DataFrame({
    0: questions,
    1: answers
})

output_df.to_csv(
    output_path,
    index=False,
    header=False,
    encoding="utf-8-sig"
)

print("\n完成！")
print("輸出檔案：", output_path)