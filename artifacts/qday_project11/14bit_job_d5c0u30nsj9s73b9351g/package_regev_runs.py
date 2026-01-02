import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def _copy_file(*, src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _extract_benchmark_metadata(obj: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "backend",
        "bit_length",
        "n_solution_qubits",
        "strategy",
        "opt_level",
        "seed_transpiler",
        "shots",
        "secret_s",
        "measure_reverse",
        "iqft_do_swaps",
        "isa_retranspile",
        "physical_qubits",
        "restrict_routing_to_physical_qubits",
        "job_id",
    ]
    out: Dict[str, Any] = {}
    for k in keys:
        if k in obj:
            out[k] = obj.get(k)

    dominance = obj.get("dominance")
    if isinstance(dominance, dict):
        for k in ["expected_rank", "p_expected", "top_bitstring", "p_top"]:
            if k in dominance:
                out[f"dominance.{k}"] = dominance.get(k)

    return out


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir",
        default=str(Path("artifacts") / "regev_package"),
        help="Output directory to create the package in (will create a timestamped subfolder).",
    )
    ap.add_argument(
        "--include-pattern",
        action="append",
        default=[],
        help="Additional glob(s) relative to repo root to include (repeatable).",
    )
    ap.add_argument(
        "--no-benchmarks",
        action="store_true",
        help="If set, do not include benchmarks/regev_fez_*.json files.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, overwrite an existing package directory (otherwise fails if target exists).",
    )

    args = ap.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_dir).resolve() / f"regev_{timestamp}"

    if out_root.exists():
        if not args.overwrite:
            print(f"ERROR: {out_root} already exists. Use --overwrite to replace.")
            return 2
        shutil.rmtree(out_root)

    include_files: List[Path] = []

    # Primary runner scripts.
    include_files.append(repo_root / "scripts" / "regev_strategy_bench.py")
    include_files.append(repo_root / "scripts" / "analyze_job_k_dominance.py")
    include_files.append(repo_root / "scripts" / "ecdlp_full_arithmetic_bench.py")

    # Original harness (for cross-checking).
    include_files.append(repo_root / "tests" / "Regev_Algorithm" / "regev_v5_harness.py")
    include_files.append(repo_root / "tests" / "Regev_Algorithm" / "regev_v5_harness_blind.py")
    include_files.append(repo_root / "tests" / "Regev_Algorithm" / "run_fez_regev.ps1")

    # Curves.
    include_files.append(repo_root / "tests" / "ECCCurves.json")
    include_files.append(repo_root / "tests" / "QDay" / "ECCCurves.json")

    # HOT framework sources needed to understand HOTCompiler + policies.
    for p in sorted((repo_root / "hot_framework").glob("*.py")):
        include_files.append(p)

    # Benchmarks.
    if not args.no_benchmarks:
        for p in sorted((repo_root / "benchmarks").glob("regev_fez_*.json")):
            include_files.append(p)

    # User-specified patterns.
    for pat in args.include_pattern:
        for p in sorted(repo_root.glob(pat)):
            if p.is_file():
                include_files.append(p)

    # Deduplicate + filter missing.
    uniq: Dict[str, Path] = {}
    missing: List[str] = []
    for p in include_files:
        if not p.exists():
            missing.append(_safe_rel(p, repo_root))
            continue
        key = str(p.resolve()).lower()
        uniq[key] = p

    files = list(uniq.values())

    manifest: Dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repo_root": str(repo_root),
        "package_root": str(out_root),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
        },
        "files": [],
        "missing_requested_files": missing,
    }

    for src in sorted(files, key=lambda p: _safe_rel(p, repo_root)):
        rel = _safe_rel(src, repo_root)
        dst = out_root / rel
        _copy_file(src=src, dst=dst)

        entry: Dict[str, Any] = {
            "src": rel,
            "dst": rel,
            "size_bytes": int(dst.stat().st_size),
            "sha256": _sha256(dst),
        }

        if src.suffix.lower() == ".json" and ("/benchmarks/" in f"/{rel}/" or rel.startswith("benchmarks/")):
            try:
                with open(dst, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                if isinstance(obj, dict):
                    entry["benchmark"] = _extract_benchmark_metadata(obj)
            except Exception as e:
                entry["benchmark_parse_error"] = str(e)

        manifest["files"].append(entry)

    (out_root / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
    with open(out_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote package to: {out_root}")
    print(f"Manifest: {out_root / 'manifest.json'}")
    if missing:
        print("Missing files (not included):")
        for m in missing:
            print(f"- {m}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
