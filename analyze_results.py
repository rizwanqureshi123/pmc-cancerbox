"""
Caption quality analysis for the PMC labeled dataset.

Computes:
  - Average caption length (words, characters)
  - Average sentence length (words per sentence)
  - Vocabulary size (unique tokens)
  - Medical terminology coverage (two views: % of captions containing at
    least one medical term, and % of the vocabulary that is medical)
  - Duplicate caption percentage

Usage:
    python analyze_captions.py --json ..\\labeled-dataset-1783958094047.json ..\\labeled-dataset1-1778366141328.json ...
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pmc_labeled_dataset import is_relevant

# --- Medical vocabulary reference -------------------------------------------------

# Curated whole-word medical/anatomical/pathology/imaging terms.
MEDICAL_TERMS = {
    # Modalities / imaging
    "mri", "ct", "pet", "ultrasound", "sonography", "mammography", "mammogram",
    "radiograph", "endoscopy", "dermoscopy", "fundus", "microscopy", "histology",
    "biopsy", "scan", "contrast", "angiography", "fluoroscopy", "tomography",
    "flair", "t1-weighted", "t2-weighted", "doppler",
    # Anatomy
    "brain", "breast", "liver", "kidney", "lung", "heart", "ovary", "ovarian",
    "uterus", "uterine", "cervix", "cervical", "lymph", "node", "pelvis",
    "pelvic", "abdomen", "abdominal", "thorax", "thoracic", "spine", "spinal",
    "prostate", "bladder", "colon", "rectum", "pancreas", "spleen", "thyroid",
    "artery", "vein", "vessel", "muscle", "bone", "tissue", "endometrium",
    "endometrial", "myometrium", "adnexa", "fallopian",
    # Pathology / findings
    "tumor", "tumour", "carcinoma", "adenocarcinoma", "sarcoma", "lesion",
    "mass", "nodule", "cyst", "cystic", "malignant", "benign", "metastasis",
    "metastases", "metastatic", "neoplasm", "neoplastic", "polyp",
    "calcification", "hemorrhage", "haemorrhage", "infarct", "aneurysm",
    "abscess", "edema", "oedema", "necrosis", "fibrosis", "atrophy",
    "hyperplasia", "dysplasia", "stenosis", "occlusion", "thrombosis",
    "inflammation", "infection", "fracture", "hematoma", "haematoma",
    "effusion", "nodular", "hypoechoic", "hyperechoic", "isoechoic",
    "enhancing", "hypodense", "hyperdense", "hypointense", "hyperintense",
    # Clinical / diagnostic
    "diagnosis", "diagnostic", "pathology", "pathological", "histopathology",
    "immunohistochemistry", "staging", "grade", "differentiated", "invasive",
    "recurrence", "resection", "excision", "surgical", "treatment", "therapy",
    "chemotherapy", "radiotherapy", "patient", "clinical",
}

# Suffix patterns that reliably indicate medical/clinical terminology even
# for words not explicitly listed above (e.g. "nephrectomy", "cholangiography").
MEDICAL_SUFFIXES = (
    "oma", "itis", "osis", "ectomy", "otomy", "ostomy", "ography", "graphy",
    "scopy", "plasty", "pathy", "algia", "emia", "genic", "trophy", "plasia",
)


def is_medical_word(word: str) -> bool:
    word = word.lower()
    if word in MEDICAL_TERMS:
        return True
    return any(word.endswith(suffix) and len(word) > len(suffix) + 2 for suffix in MEDICAL_SUFFIXES)


# --- Text helpers ------------------------------------------------------------------

WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z\-]*")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def tokenize_words(text: str) -> list[str]:
    return [w.lower() for w in WORD_RE.findall(text)]


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    sentences = SENTENCE_SPLIT_RE.split(text)
    return [s for s in sentences if s.strip()]


# --- Main analysis -------------------------------------------------------------------

def collect_captions(json_paths: list[str]) -> list[str]:
    captions: list[str] = []
    for json_path in json_paths:
        with Path(json_path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        for entry in data:
            images = entry["images"] if "images" in entry else [entry]
            for image_item in images:
                if not is_relevant(image_item):
                    continue
                caption = (image_item.get("caption") or "").strip()
                if caption:
                    captions.append(caption)
    return captions


def analyze(captions: list[str], output_dir: Path) -> None:
    total = len(captions)
    if total == 0:
        print("No captions found.")
        return

    # Duplicate percentage
    counts = Counter(captions)
    unique = len(counts)
    duplicate_pct = 100 * (total - unique) / total

    # Caption length stats
    word_counts = [len(tokenize_words(c)) for c in captions]
    char_counts = [len(c) for c in captions]
    avg_words = sum(word_counts) / total
    avg_chars = sum(char_counts) / total

    # Sentence-level stats
    all_sentence_lengths = []
    for c in captions:
        for sentence in split_sentences(c):
            all_sentence_lengths.append(len(tokenize_words(sentence)))
    avg_sentence_len = (
        sum(all_sentence_lengths) / len(all_sentence_lengths) if all_sentence_lengths else 0.0
    )

    # Vocabulary
    vocab = Counter()
    for c in captions:
        vocab.update(tokenize_words(c))
    vocab_size = len(vocab)

    # Medical terminology coverage
    medical_vocab_words = [w for w in vocab if is_medical_word(w)]
    medical_vocab_pct = 100 * len(medical_vocab_words) / vocab_size if vocab_size else 0.0

    captions_with_medical_term = sum(
        1 for c in captions if any(is_medical_word(w) for w in tokenize_words(c))
    )
    captions_medical_pct = 100 * captions_with_medical_term / total

    print(f"Total captions:                    {total}")
    print(f"Unique captions:                   {unique}")
    print(f"Duplicate caption percentage:      {duplicate_pct:.2f}%")
    print()
    print(f"Average caption length (words):    {avg_words:.2f}")
    print(f"Average caption length (chars):    {avg_chars:.2f}")
    print(f"Average sentence length (words):   {avg_sentence_len:.2f}")
    print()
    print(f"Vocabulary size (unique tokens):   {vocab_size}")
    print(f"Medical terms in vocabulary:       {len(medical_vocab_words)} ({medical_vocab_pct:.2f}% of vocab)")
    print(f"Captions containing >=1 med term:  {captions_with_medical_term} ({captions_medical_pct:.2f}% of captions)")
    print()
    print("Top 15 most common words (sanity check, includes stopwords):")
    for word, count in vocab.most_common(15):
        print(f"  {word:20s} {count}")

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.hist(word_counts, bins=40, color="#4C72B0")
    plt.title("Caption Length Distribution (words)")
    plt.xlabel("Words per caption")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_dir / "caption_length_distribution.png", dpi=150)
    plt.close()
    print(f"\nChart saved to {output_dir / 'caption_length_distribution.png'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze caption/text quality across the dataset.")
    parser.add_argument("--json", nargs="+", required=True, help="One or more labeled JSON files.")
    parser.add_argument("--output-dir", type=str, default="eval_charts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    captions = collect_captions(args.json)
    analyze(captions, Path(args.output_dir))


if __name__ == "__main__":
    main()