import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _find_ecc_curves_json() -> Path:
    # Prefer a local copy (artifact bundle can be made fully standalone).
    local = Path(__file__).resolve().parent / "ECCCurves.json"
    if local.exists():
        return local

    # Default repo location.
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / "tests" / "QDay" / "ECCCurves.json"
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        "ECCCurves.json not found. Expected either a local copy next to make_ecdlp_plots.py "
        "or at <repo_root>/tests/QDay/ECCCurves.json"
    )


def _load_curve_params_from_ecc_curves(*, bit_length: int):
    # Import from local artifact copy.
    from quantum_ecc_solver_proper import ECCParams, ECPoint

    curves_path = _find_ecc_curves_json()
    curves = _load_json(curves_path)
    target = next((r for r in curves if int(r.get("bit_length", -1)) == int(bit_length)), None)
    if target is None:
        raise RuntimeError(f"No {bit_length}-bit curve found in {curves_path}")

    params = ECCParams(
        p=int(target["prime"]),
        a=0,
        b=2,
        P=ECPoint(int(target["generator_point"][0]), int(target["generator_point"][1])),
        Q=ECPoint(int(target["public_key"][0]), int(target["public_key"][1])),
        n=int(target["subgroup_order"]),
    )
    expected_k = int(target.get("private_key", -1))
    return params, expected_k


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_compiled_depth(run: Dict[str, Any]) -> Optional[int]:
    meta = run.get("meta") or {}
    for k in ("compiled_circuit_metrics", "compiled_circuit_metrics_padded", "submitted_circuit_metrics"):
        d = (meta.get(k) or {}).get("depth")
        if isinstance(d, int):
            return d
        if isinstance(d, (float, str)):
            try:
                return int(d)
            except Exception:
                pass
    return None


def _get_job_id(run: Dict[str, Any]) -> Optional[str]:
    meta = run.get("meta") or {}
    jid = meta.get("job_id")
    return str(jid) if jid else None


def _get_expected_k(run: Dict[str, Any]) -> Optional[int]:
    meta = run.get("meta") or {}
    k = meta.get("expected_k")
    return int(k) if isinstance(k, int) else None


def _extract_counts_from_sampler_result(obj: Any) -> Dict[str, int]:
    """Best-effort extraction of classical counts from Qiskit Sampler results.

    Supports multiple Qiskit Runtime/Sampler versions via duck typing.
    """

    # Common v2 primitive result layout: result[0].data.c
    try:
        pub0 = obj[0]
    except Exception:
        pub0 = None

    candidates: List[Any] = []
    if pub0 is not None:
        candidates.append(pub0)
        data = getattr(pub0, "data", None)
        if data is not None:
            candidates.append(data)
            for attr in ("c", "counts", "quasi_dists"):
                if hasattr(data, attr):
                    candidates.append(getattr(data, attr))

    candidates.append(obj)

    for c in candidates:
        if c is None:
            continue

        # If already a dict[str,int]
        if isinstance(c, dict) and c and all(isinstance(k, str) for k in c.keys()):
            try:
                return {str(k): int(v) for k, v in c.items()}
            except Exception:
                pass

        # BitArray supports get_counts() in some versions
        get_counts = getattr(c, "get_counts", None)
        if callable(get_counts):
            try:
                out = get_counts()
                if isinstance(out, dict):
                    return {str(k): int(v) for k, v in out.items()}
            except Exception:
                pass

        # Some containers have .counts
        counts_attr = getattr(c, "counts", None)
        if isinstance(counts_attr, dict):
            try:
                return {str(k): int(v) for k, v in counts_attr.items()}
            except Exception:
                pass

        # Quasi distribution: dict[int,float] -> convert to pseudo-counts via rounding by shots
        if isinstance(c, dict) and c and all(isinstance(k, int) for k in c.keys()):
            # Use shots if present; otherwise treat as probabilities.
            shots = None
            try:
                md = getattr(obj, "metadata", None) or {}
                shots = md.get("shots")
            except Exception:
                shots = None
            if shots is None:
                try:
                    shots = int(sum(float(v) for v in c.values()))
                except Exception:
                    shots = None
            if shots is None or shots <= 0:
                continue
            out: Dict[str, int] = {}
            for k_int, p in c.items():
                try:
                    count = int(round(float(p) * shots))
                except Exception:
                    continue
                out[format(int(k_int), "b")] = count
            if out:
                return out

    raise RuntimeError("Unable to extract counts from sampler result")


def _runtime_service():
    from qiskit_ibm_runtime import QiskitRuntimeService

    token = os.environ.get("IBM_QUANTUM_TOKEN")
    if not token:
        raise RuntimeError("IBM_QUANTUM_TOKEN is required to fetch job results")

    instance = os.environ.get("IBM_QUANTUM_CRN")
    if instance:
        return QiskitRuntimeService(channel="ibm_cloud", token=token, instance=instance)

    return QiskitRuntimeService(channel="ibm_quantum", token=token)


def _compute_diagnostics_from_job(*, job_id: str, bit_length: int) -> Dict[str, Any]:
    from qiskit_ibm_runtime import Sampler

    # Use local copy in the artifact folder.
    from quantum_ecc_solver_proper import _extract_k_from_counts_2d_with_diagnostics

    service = _runtime_service()
    job = service.job(job_id)
    result = job.result()

    counts = _extract_counts_from_sampler_result(result)
    params, expected_k = _load_curve_params_from_ecc_curves(bit_length=bit_length)

    _, diag = _extract_k_from_counts_2d_with_diagnostics(
        counts=counts,
        params=params,
        order_bits=bit_length,
        expected_k=expected_k,
    )
    return diag


def _load_or_compute_diagnostics(*, run_json: Dict[str, Any], run_path: Path) -> Dict[str, Any]:
    meta = run_json.get("meta") or {}
    diag = meta.get("dominance_diagnostics")
    if isinstance(diag, dict) and diag.get("total_shots"):
        return diag

    bit_length = int(run_json.get("bit_length"))
    job_id = _get_job_id(run_json)
    if not job_id:
        raise RuntimeError(f"Missing job_id in {run_path}")

    return _compute_diagnostics_from_job(job_id=job_id, bit_length=bit_length)


def plot_expected_prob_vs_depth(rows: List[Dict[str, Any]], *, out_path: Path) -> None:
    rows = [r for r in rows if r.get("compiled_depth") is not None and r.get("expected_k_prob") is not None]
    rows = sorted(rows, key=lambda r: int(r["compiled_depth"]))

    x = np.array([r["compiled_depth"] for r in rows], dtype=float)
    y = np.array([r["expected_k_prob"] for r in rows], dtype=float)
    bits = [int(r["bit_length"]) for r in rows]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, y, marker="o", linewidth=2)
    for xi, yi, b in zip(x, y, bits):
        ax.annotate(str(b), (xi, yi), textcoords="offset points", xytext=(6, 6), fontsize=9)

    ax.set_xlabel("Compiled physical depth")
    ax.set_ylabel("Expected key support probability (expected_k_prob)")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.set_title("ECDLP signal decay: expected key support vs depth")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_candidate_dissipation_heatmap(rows: List[Dict[str, Any]], *, out_path: Path, max_rank: int = 50) -> None:
    rows = sorted(rows, key=lambda r: int(r["bit_length"]))
    bits = [int(r["bit_length"]) for r in rows]

    M = np.zeros((len(rows), max_rank), dtype=float)
    exp_rank = []

    for i, r in enumerate(rows):
        top = r.get("top_candidates") or []
        for j, item in enumerate(top[:max_rank]):
            try:
                M[i, j] = float(item.get("prob", 0.0))
            except Exception:
                M[i, j] = 0.0
        er = r.get("expected_k_rank")
        exp_rank.append(int(er) if isinstance(er, int) else None)

    fig, ax = plt.subplots(figsize=(10, 4.8))

    # log-scale colormap while avoiding log(0)
    eps = 1e-12
    img = ax.imshow(np.log10(M + eps), aspect="auto", interpolation="nearest", cmap="viridis")

    ax.set_yticks(range(len(bits)))
    ax.set_yticklabels([str(b) for b in bits])
    ax.set_xlabel("Candidate rank (1 = highest support)")
    ax.set_ylabel("Bit-length")
    ax.set_title("Candidate dissipation heatmap (log10 prob)")

    ax.set_xticks([0, 9, 19, 29, 39, 49])
    ax.set_xticklabels(["1", "10", "20", "30", "40", "50"])

    # overlay expected-k rank when available
    for i, er in enumerate(exp_rank):
        if er is None:
            continue
        if 1 <= er <= max_rank:
            ax.plot([er - 1], [i], marker="x", color="red", markersize=6, mew=2)

    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label("log10(probability)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--benchmarks-dir",
        default=str(Path(__file__).resolve().parents[1]),
        help="Path to benchmarks/ directory",
    )
    ap.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent),
        help="Output directory (defaults to this artifact folder)",
    )
    ap.add_argument("--max-rank", type=int, default=50)
    args = ap.parse_args()

    bench_dir = Path(args.benchmarks_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Include 6/8/10/11 (non-diagnostic) and 12/13/14 (diagnostic).
    run_files = [
        bench_dir / "ecdlp_fez_6bit_sabre.pure.o2.json",
        bench_dir / "ecdlp_fez_8bit_sabre.pure.o2.json",
        bench_dir / "ecdlp_fez_10bit_sabre.pure.o2.json",
        bench_dir / "ecdlp_fez_11bit_sabre.pure.o2.json",
        bench_dir / "diagnostic.ecdlp_fez_12bit_sabre.pure.o2.lowanc.json",
        bench_dir / "diagnostic.ecdlp_fez_13bit_sabre.pure.o2.lowanc.json",
        bench_dir / "diagnostic.ecdlp_fez_14bit_sabre.pure.o2.lowanc.json",
    ]

    rows: List[Dict[str, Any]] = []
    for p in run_files:
        if not p.exists():
            continue
        run = _load_json(p)
        depth = _get_compiled_depth(run)
        bit_length = int(run.get("bit_length"))

        diag = _load_or_compute_diagnostics(run_json=run, run_path=p)

        rows.append(
            {
                "bit_length": bit_length,
                "compiled_depth": depth,
                "expected_k_prob": diag.get("expected_k_prob"),
                "expected_k_rank": diag.get("expected_k_rank"),
                "top_candidates": diag.get("top_candidates"),
            }
        )

    if not rows:
        raise RuntimeError("No benchmark rows found")

    plot_expected_prob_vs_depth(rows, out_path=out_dir / "expected_k_prob_vs_depth.png")
    plot_candidate_dissipation_heatmap(
        rows, out_path=out_dir / "candidate_dissipation_heatmap.png", max_rank=int(args.max_rank)
    )

    with open(out_dir / "plot_data.ecdlp_signal_decay.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
