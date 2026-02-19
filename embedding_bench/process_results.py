#!/usr/bin/env python3
"""
Aggregate embedding benchmark results and print a markdown summary table.

Usage:
    python process_results.py --results-dir ../results/ --hardware h100
    python process_results.py --results-dir ../results/  # discovers hardware from dir names
"""

import argparse
import json
from pathlib import Path


def load_results(results_dir: Path, hardware_override: str | None) -> list[dict]:
    """Walk results_dir, load all *.json files (skip server.log and summary.json)."""
    records = []
    for json_file in sorted(results_dir.rglob("*.json")):
        if json_file.name in ("summary.json",):
            continue
        try:
            with open(json_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: skipping {json_file}: {e}")
            continue

        # Infer hardware from directory name pattern: <model_slug>__<hardware>
        hardware = hardware_override
        if hardware is None:
            # Directory containing the json file may be named <slug>__<hw>
            parent = json_file.parent.name
            if "__" in parent:
                hardware = parent.split("__", 1)[-1]
            else:
                hardware = "unknown"

        data["hardware"] = hardware
        # Derived metric: per-text p99 latency
        batch_size = data.get("batch_size", 1)
        p99 = data.get("p99_latency_ms", 0.0)
        data["latency_per_text_ms"] = round(p99 / batch_size, 3) if batch_size else 0.0

        records.append(data)
    return records


def sort_key(r: dict) -> tuple:
    return (
        r.get("model", ""),
        r.get("hardware", ""),
        r.get("chunk_size", 0),
        r.get("batch_size", 0),
        r.get("concurrency", 0),
    )


def print_markdown_table(records: list[dict]) -> None:
    headers = [
        "Model",
        "Hardware",
        "Chunk",
        "Batch",
        "Conc",
        "p50 (ms)",
        "p99 (ms)",
        "Tput (emb/s)",
        "Tput/User",
        "Power (W)",
        "Emb/Joule",
    ]
    rows = []
    for r in records:
        model_short = r.get("model", "").split("/")[-1]
        power_w = r.get("power_avg_w")
        power_str = f"{power_w:.1f}" if power_w is not None else "-"
        emb_per_joule = r.get("emb_per_joule")
        emb_per_joule_str = f"{emb_per_joule:.2f}" if emb_per_joule is not None else "-"
        rows.append([
            model_short,
            r.get("hardware", ""),
            str(r.get("chunk_size", "")),
            str(r.get("batch_size", "")),
            str(r.get("concurrency", "")),
            f"{r.get('p50_latency_ms', 0):.1f}",
            f"{r.get('p99_latency_ms', 0):.1f}",
            f"{r.get('throughput_emb_per_sec', 0):.1f}",
            f"{r.get('throughput_per_user', 0):.1f}",
            power_str,
            emb_per_joule_str,
        ])

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def fmt_row(cells: list[str]) -> str:
        return "| " + " | ".join(c.ljust(col_widths[i]) for i, c in enumerate(cells)) + " |"

    sep = "|-" + "-|-".join("-" * w for w in col_widths) + "-|"

    print(fmt_row(headers))
    print(sep)
    for row in rows:
        print(fmt_row(row))


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate embedding benchmark results")
    parser.add_argument("--results-dir", default="../results", help="Root results directory")
    parser.add_argument("--hardware", default=None, help="Hardware label (overrides inferred value)")
    parser.add_argument("--output", default=None, help="Write summary JSON to this path (default: <results-dir>/summary.json)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: results directory not found: {results_dir}")
        raise SystemExit(1)

    records = load_results(results_dir, args.hardware)
    if not records:
        print("No result JSON files found.")
        raise SystemExit(1)

    records.sort(key=sort_key)

    out_path = Path(args.output) if args.output else results_dir / "summary.json"
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Wrote {len(records)} records to {out_path}\n")

    print_markdown_table(records)


if __name__ == "__main__":
    main()
