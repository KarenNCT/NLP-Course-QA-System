import os
import fitz
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = BASE_DIR
OUTPUT_CSV = "documents.csv"

rows = []

for filename in os.listdir(DATA_DIR):
    if filename.lower().endswith(".pdf"):

        pdf_path = os.path.join(DATA_DIR, filename)
        pdf = fitz.open(pdf_path)

        for page_num, page in enumerate(pdf, start=1):

            text = page.get_text().strip()

            if text:
                rows.append({
                    "source": filename,
                    "page": page_num,
                    "text": text
                })

df = pd.DataFrame(rows)

output_path = os.path.join(DATA_DIR, OUTPUT_CSV)

df.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("完成！")
print("輸出：", output_path)
print("總頁數：", len(df))