#!/usr/bin/env python3
"""
get_top_taxa.py

Examples:
    python xc_scripts/get_top_taxa.py --key YOUR_KEY
    python xc_scripts/get_top_taxa.py --key 7c6325af6cd6edd6c65f6eb3c68de849b5b16070 --clades frogs
    python xc_scripts/get_top_taxa.py --key YOUR_KEY --clades birds --top 20 --max-pages 50
    python xc_scripts/get_top_taxa.py --key 7c6325af6cd6edd6c65f6eb3c68de849b5b16070 --clades grasshoppers --out xc_top_taxa.csv --max-pages 50
    python xc_scripts/get_top_taxa.py --key 7c6325af6cd6edd6c65f6eb3c68de849b5b16070 --clades bats,frogs,grasshoppers --out xc_top_taxa.csv --max-pages 300

Requires:
    pip install requests
"""
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import requests

BASE_URL = "https://xeno-canto.org/api/3/recordings"

# Keep this very simple.
# One clade = one simple query whenever possible.
CLADES: Dict[str, str] = {
    "birds": "grp:birds",
    "frogs": "grp:frogs",
    "bats": "grp:bats",
    "land_mammals": "grp:land mammals",
    "sea_mammals": "grp:sea_mammals",
    "orthoptera": "ord:Orthoptera",
    "grasshoppers": "grp:grasshoppers",   # alias
    "cicadas": "fam:Cicadidae",
}

ALIASES: Dict[str, str] = {
    "grasshopper": "grasshoppers",
    "grasshoppers": "grasshoppers",
    "orthoptera": "orthoptera",
    "frog": "frogs",
    "bat": "bats",
    "bird": "birds",
    "landmammals": "land_mammals",
    "land-mammals": "land_mammals",
    "seamammals": "sea_mammals",
    "sea-mammals": "sea_mammals",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", required=True, help="Xeno-canto API key")
    parser.add_argument(
        "--clades",
        default="birds,frogs",
        help="Comma-separated clades, e.g. birds,frogs,grasshoppers",
    )
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--max-pages", type=int, default=10)
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--out", default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()

def normalize_clade_name(name: str) -> str:
    name = name.strip().lower()
    return ALIASES.get(name, name)

def request_page(api_key: str, query: str, page: int, timeout: int) -> dict:
    params = {
        "query": query,
        "page": page,
        "key": api_key,
    }
    r = requests.get(BASE_URL, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def get_first_nonempty(rec: dict, keys: List[str]) -> Optional[str]:
    for key in keys:
        value = rec.get(key)
        if value is not None:
            value = str(value).strip()
            if value:
                return value
    return None

def extract_names(rec: dict) -> Tuple[str, str]:
    gen = get_first_nonempty(rec, ["gen", "genus"])
    sp = get_first_nonempty(rec, ["sp", "species"])
    sci = get_first_nonempty(rec, ["scientific-name", "scientific_name", "sci", "sciName"])
    com = get_first_nonempty(rec, ["en", "english", "common-name", "common_name"])

    if gen and sp:
        scientific_name = f"{gen} {sp}"
    elif sci:
        scientific_name = sci
    else:
        scientific_name = "UNKNOWN_SCIENTIFIC_NAME"

    common_name = com if com else "UNKNOWN_COMMON_NAME"
    return scientific_name, common_name

def fetch_counts_for_clade(
    api_key: str,
    clade_name: str,
    timeout: int,
    sleep_s: float,
    max_pages: int,
    verbose: bool,
) -> Tuple[Counter, Dict[str, str], str, int]:
    query = CLADES[clade_name]

    counts = Counter()
    common_names: Dict[str, str] = {}
    total_records = 0

    for page in range(1, max_pages + 1):
        data = request_page(api_key, query, page, timeout)
        recordings = data.get("recordings", []) or []

        if not recordings:
            if verbose:
                print(f"    no recordings on page {page}, stopping")
            break

        for rec in recordings:
            sci, com = extract_names(rec)
            counts[sci] += 1
            common_names[sci] = com
            total_records += 1

        if verbose:
            num_pages = data.get("numPages", "?")
            print(f"    page {page}/{num_pages} | query={query!r} | total_records={total_records}")

        try:
            num_pages = int(data.get("numPages", page))
            if page >= num_pages:
                break
        except Exception:
            pass

        time.sleep(sleep_s)

    return counts, common_names, query, total_records

def print_results(results: Dict[str, dict], top_n: int) -> None:
    for clade_name, payload in results.items():
        counts = payload["counts"]
        common_names = payload["common_names"]
        used_query = payload["used_query"]
        total_records = payload["total_records"]

        print("\n" + "=" * 80)
        print(f"CLADE: {clade_name}")
        print(f"USED QUERY: {used_query}")
        print(f"TOTAL RECORDS FETCHED: {total_records}")
        print("-" * 80)

        if not counts:
            print("No records found.")
            continue

        print(f"{'Rank':<6}{'Common name':<35}{'Scientific name':<35}{'Count':>10}")
        print("-" * 80)
        for i, (sci, n) in enumerate(counts.most_common(top_n), start=1):
            com = common_names.get(sci, "UNKNOWN_COMMON_NAME")
            print(f"{i:<6}{com[:33]:<35}{sci[:33]:<35}{n:>10}")

def save_csv(path: str, results: Dict[str, dict], top_n: int) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["clade", "used_query", "rank", "common_name", "scientific_name", "count"])
        for clade_name, payload in results.items():
            counts = payload["counts"]
            common_names = payload["common_names"]
            used_query = payload["used_query"]

            for i, (sci, n) in enumerate(counts.most_common(top_n), start=1):
                writer.writerow([
                    clade_name,
                    used_query,
                    i,
                    common_names.get(sci, "UNKNOWN_COMMON_NAME"),
                    sci,
                    n,
                ])

def main() -> int:
    args = parse_args()

    requested = [normalize_clade_name(c) for c in args.clades.split(",") if c.strip()]
    unknown = [c for c in requested if c not in CLADES]
    if unknown:
        print(f"Unknown clade(s): {', '.join(unknown)}")
        print(f"Available clades: {', '.join(CLADES.keys())}")
        return 1

    print("Fetching Xeno-canto top taxa by clade...")
    results: Dict[str, dict] = {}

    for clade_name in requested:
        print(f"\nProcessing clade: {clade_name}")
        try:
            counts, common_names, used_query, total_records = fetch_counts_for_clade(
                api_key=args.key,
                clade_name=clade_name,
                timeout=args.timeout,
                sleep_s=args.sleep,
                max_pages=args.max_pages,
                verbose=args.verbose,
            )
        except requests.HTTPError as e:
            print(f"  HTTP error for {clade_name}: {e}")
            counts, common_names, used_query, total_records = Counter(), {}, CLADES[clade_name], 0

        results[clade_name] = {
            "counts": counts,
            "common_names": common_names,
            "used_query": used_query,
            "total_records": total_records,
        }

    print_results(results, args.top)

    if args.out:
        save_csv(args.out, results, args.top)
        print(f"\nSaved CSV to: {args.out}")

    return 0

if __name__ == "__main__":
    sys.exit(main())