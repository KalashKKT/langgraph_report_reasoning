import fitz  # PyMuPDF
from pathlib import Path
import json
import fitz

def extract_pages(pdf_path: str) -> list[str]:
    """
    Extract text from each page of a PDF and return a list like:
    [
        "Page 0: <page 0 text>",
        "Page 1: <page 1 text>",
        ...
    ]
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages: list[str] = []

    doc = fitz.open(pdf_path)
    try:
        for page_index, page in enumerate(doc):
            # Extract plain text
            raw_text = page.get_text("text")

            # Normalize whitespace (remove extra newlines/tabs)
            normalized_text = " ".join(raw_text.split())

            pages.append(f"Page {page_index}: {normalized_text}")
    finally:
        doc.close()

    return pages


if __name__ == "__main__":
    # 🔹 Input PDF path
    pdf_file = r"D:\Piratech\data_agents\reports\aum_report.pdf"

    # 🔹 Extract text
    extracted_pages = extract_pages(pdf_file)

    # 🔹 Output directory (your required location)
    output_dir = Path(r"D:\Piratech\data_agents\reports\text_extraction")
    output_dir.mkdir(parents=True, exist_ok=True)  # create folder if missing

    # 🔹 Create output filenames based on PDF name
    pdf_name = Path(pdf_file).stem.replace("–", "-")

    output_txt = output_dir / f"{pdf_name}.txt"
    output_json = output_dir / f"{pdf_name}.json"

    # 🔹 Save TXT (one page per line)
    with output_txt.open("w", encoding="utf-8") as f:
        for page in extracted_pages:
            f.write(page + "\n")

    # 🔹 Save JSON (list format)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(extracted_pages, f, ensure_ascii=False, indent=2)

    print(f"Extraction completed successfully.")
    print(f"TXT saved to: {output_txt}")
    print(f"JSON saved to: {output_json}")