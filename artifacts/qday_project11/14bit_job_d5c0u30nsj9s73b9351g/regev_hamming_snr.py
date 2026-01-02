import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _hamming(a: str, b: str) -> int:
    if len(a) != len(b):
        raise ValueError("Hamming distance requires equal-length strings")
    return sum(1 for x, y in zip(a, b) if x != y)


def _load_counts(obj: Dict[str, Any]) -> Dict[str, int]:
    # Prefer full counts if present (future-proof), otherwise use counts_top25.
    counts = obj.get("counts")
    if isinstance(counts, dict):
        out: Dict[str, int] = {}
        for k, v in counts.items():
            try:
                out[str(k)] = int(v)
            except Exception:
                pass
        if out:
            return out

    top = obj.get("counts_top25")
    if isinstance(top, list):
        out = {}
        for row in top:
            if not isinstance(row, list) or len(row) != 2:
                continue
            bs, c = row
            try:
                out[str(bs)] = int(c)
            except Exception:
                pass
        return out

    return {}


def _hamming_histogram(*, counts: Dict[str, int], expected: str) -> Tuple[Dict[int, int], int]:
    hist: Dict[int, int] = {}
    total = 0
    for bs, c in counts.items():
        if not isinstance(bs, str) or len(bs) != len(expected):
            continue
        d = _hamming(bs, expected)
        hist[d] = hist.get(d, 0) + int(c)
        total += int(c)
    return dict(sorted(hist.items())), total


def _snr_from_hist(*, hist: Dict[int, int]) -> Dict[str, Any]:
    # Simple “signal vs noise” summary:
    # - signal = mass at distance 0
    # - near = mass at distance 1
    # - noise = all other mass
    s0 = int(hist.get(0, 0))
    s1 = int(hist.get(1, 0))
    total = int(sum(hist.values()))
    noise = max(0, total - s0)
    far = max(0, total - s0 - s1)

    def safe_div(a: float, b: float) -> float:
        return float(a / b) if b else float("inf")

    return {
        "total": total,
        "distance0": s0,
        "distance1": s1,
        "noise_nonzero": noise,
        "far_distance_ge2": far,
        "snr_d0_over_rest": safe_div(s0, max(1, total - s0)),
        "snr_d0_over_d1": safe_div(s0, max(1, s1)),
        "p_distance0": safe_div(s0, max(1, total)),
        "p_distance1": safe_div(s1, max(1, total)),
        "p_far_ge2": safe_div(far, max(1, total)),
    }


def _maybe_plot(*, out_dir: Path, stem: str, counts: Dict[str, int], expected: str, title_prefix: str) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    hist, total = _hamming_histogram(counts=counts, expected=expected)

    # Hamming density plot
    xs = list(hist.keys())
    ys = [hist[x] for x in xs]

    fig = plt.figure(figsize=(8, 4.5))
    plt.bar(xs, ys)
    plt.yscale("log")
    plt.xlabel("Hamming distance to expected bitstring")
    plt.ylabel("Counts (log scale)")
    plt.title(f"{title_prefix} Hamming density (N={total})")
    plt.tight_layout()
    fig.savefig(out_dir / f"{stem}.hamming_density.png", dpi=200)
    plt.close(fig)

    # Top outcomes plot
    topk = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:25]
    labels = [bs for bs, _ in topk]
    vals = [c for _, c in topk]

    fig2 = plt.figure(figsize=(10, 6))
    plt.barh(range(len(labels))[::-1], vals[::-1])
    plt.yticks(range(len(labels))[::-1], labels[::-1], fontsize=7)
    plt.xlabel("Counts")
    plt.title(f"{title_prefix} top outcomes")
    plt.tight_layout()
    fig2.savefig(out_dir / f"{stem}.top_outcomes.png", dpi=200)
    plt.close(fig2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("json_files", nargs="+", help="Benchmark JSON(s) from scripts/regev_strategy_bench.py")
    ap.add_argument("--out-dir", default=str(Path("artifacts") / "regev_analysis"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "inputs": [],
        "generated_at": None,
        "results": [],
    }

    try:
        from datetime import datetime, timezone

        summary["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        summary["generated_at"] = None

    for p_str in args.json_files:
        p = Path(p_str)
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)

        dominance = obj.get("dominance") if isinstance(obj, dict) else None
        expected = None
        if isinstance(dominance, dict):
            expected = dominance.get("expected_bitstring")

        counts = _load_counts(obj if isinstance(obj, dict) else {})

        record: Dict[str, Any] = {
            "file": str(p),
            "job_id": (obj.get("job_id") if isinstance(obj, dict) else None),
            "backend": (obj.get("backend") if isinstance(obj, dict) else None),
            "bit_length": (obj.get("bit_length") if isinstance(obj, dict) else None),
            "n_solution_qubits": (obj.get("n_solution_qubits") if isinstance(obj, dict) else None),
            "strategy": (obj.get("strategy") if isinstance(obj, dict) else None),
            "iqft_do_swaps": (obj.get("iqft_do_swaps") if isinstance(obj, dict) else None),
            "measure_reverse": (obj.get("measure_reverse") if isinstance(obj, dict) else None),
            "compiled_depth": ((obj.get("compiled_metrics") or {}).get("depth") if isinstance(obj, dict) else None),
            "twoq": ((obj.get("compiled_metrics") or {}).get("two_qubit_gate_count") if isinstance(obj, dict) else None),
            "counts_source": "counts" if isinstance(obj, dict) and isinstance(obj.get("counts"), dict) else "counts_top25",
        }

        if not expected or not isinstance(expected, str):
            record["error"] = "No dominance.expected_bitstring found; cannot compute Hamming histogram"
            summary["results"].append(record)
            continue

        hist, total = _hamming_histogram(counts=counts, expected=expected)
        snr = _snr_from_hist(hist=hist)

        record["hamming_histogram"] = hist
        record["snr"] = snr
        record["counts_total_included"] = total

        stem = p.stem
        title_prefix = f"{record.get('backend')} | {record.get('strategy')} | n={record.get('n_solution_qubits')}"
        _maybe_plot(out_dir=out_dir, stem=stem, counts=counts, expected=expected, title_prefix=title_prefix)

        summary["results"].append(record)
        summary["inputs"].append(str(p))

    out_path = out_dir / "regev_hamming_snr_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote: {out_path}")
    print(f"Plots (if matplotlib installed): {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
