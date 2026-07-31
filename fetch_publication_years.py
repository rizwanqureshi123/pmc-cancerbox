"""
Fetch publication years for the source PMC articles behind your dataset,
via NCBI's E-utilities (esummary), and plot the distribution.

Runs entirely against NCBI's public API -- no local checkpoint or GPU needed.
Uses batched requests (up to 150 PMC IDs per call) to stay well within NCBI's
public rate limits.

Usage:
    python fetch_publication_years.py --json ..\\labeled-dataset-1783958094047.json ... --output-dir eval_charts
"""

from __future__ import annotations

import argparse
import json
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pmc_labeled_dataset import is_relevant

ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
BATCH_SIZE = 150
DELAY_SECONDS = 0.4  # stay under NCBI's ~3 req/sec unauthenticated limit


def collect_pmcids(json_paths: list[str]) -> list[str]:
    pmcids: set[str] = set()
    for json_path in json_paths:
        with Path(json_path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        for entry in data:
            entry_pmcid = entry.get("pmcid")
            images = entry["images"] if "images" in entry else [entry]
            for image_item in images:
                if not is_relevant(image_item):
                    continue
                pmcid = image_item.get("pmcid") or entry_pmcid
                if pmcid:
                    pmcids.add(str(pmcid).replace("PMC", ""))
    return sorted(pmcids)


def fetch_years(pmcids: list[str]) -> dict[str, str]:
    years: dict[str, str] = {}
    for i in range(0, len(pmcids), BATCH_SIZE):
        batch = pmcids[i : i + BATCH_SIZE]
        params = {"db": "pmc", "id": ",".join(batch), "retmode": "xml"}
        url = f"{ESUMMARY_URL}?{urlencode(params)}"
        try:
            with urlopen(url, timeout=30) as resp:
                xml_data = resp.read()
            root = ET.fromstring(xml_data)
            for doc in root.findall(".//DocSum"):
                id_elem = doc.find("Id")
                pubdate_elem = next(
                    (item for item in doc.findall("Item") if item.get("Name") == "PubDate"), None
                )
                if id_elem is not None and pubdate_elem is not None and pubdate_elem.text:
                    year = pubdate_elem.text.strip()[:4]
                    if year.isdigit():
                        years[id_elem.text] = year
        except Exception as exc:
            print(f"  Batch {i}-{i + len(batch)} failed: {exc}")
        print(f"  Fetched {min(i + BATCH_SIZE, len(pmcids))}/{len(pmcids)}")
        time.sleep(DELAY_SECONDS)
    return years


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch PMC publication years and plot distribution.")
    parser.add_argument("--json", nargs="+", required=True)
    parser.add_argument("--output-dir", type=str, default="eval_charts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pmcids = collect_pmcids(args.json)
    print(f"Found {len(pmcids)} unique PMC IDs. Querying NCBI E-utilities...")
    years_by_id = fetch_years(pmcids)
    print(f"Resolved publication year for {len(years_by_id)}/{len(pmcids)} articles.")

    year_counts = Counter(years_by_id.values())
    sorted_years = sorted(year_counts.items())

    print("\nPublication year distribution:")
    for year, count in sorted_years:
        print(f"  {year}: {count}")

    if sorted_years:
        years, counts = zip(*sorted_years)
        plt.figure(figsize=(10, 5))
        plt.bar(years, counts, color="#8172B2")
        plt.title("Publication Year Distribution of Source PMC Articles")
        plt.xlabel("Year")
        plt.ylabel("Number of Articles")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / "publication_year_distribution.png", dpi=150)
        plt.close()
        print(f"\nChart saved to {out_dir / 'publication_year_distribution.png'}")


if __name__ == "__main__":
    main()