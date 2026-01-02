import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


def _load_dotenv() -> None:
    if load_dotenv is None:
        return
    dotenv_path = Path.cwd() / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path, override=True)
    else:
        load_dotenv(override=True)


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ibm_fez")
    ap.add_argument("--bit-length", type=int, required=True)
    ap.add_argument("--strategy", choices=["sabre", "hot", "hot_sabre"], default="sabre")
    ap.add_argument("--circuit-profile", choices=["nisq", "pure"], default="nisq")
    ap.add_argument("--opt-level", type=int, default=2)
    ap.add_argument("--transpile-mode", choices=["preset", "transpile"], default="preset")
    ap.add_argument("--seed-transpiler", type=int, default=1234)
    ap.add_argument("--num-processes", type=int, default=1)
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--run-hardware", action="store_true")
    ap.add_argument("--shots", type=int, default=8192)
    ap.add_argument("--job-tag", default="")
    ap.add_argument("--out-json", required=True)

    args = ap.parse_args()

    if bool(args.run_hardware) == bool(args.compile_only):
        raise ValueError("Provide exactly one of --compile-only or --run-hardware")

    _load_dotenv()

    repo_root = Path(__file__).resolve().parents[1]
    qday_path = repo_root / "tests" / "QDay"
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if str(qday_path) not in sys.path:
        sys.path.insert(0, str(qday_path))

    try:
        from tests.QDay.quantum_ecc_solver_proper import compile_ecdlp  # type: ignore
    except Exception:
        from quantum_ecc_solver_proper import compile_ecdlp  # type: ignore

    if args.job_tag:
        os.environ.setdefault("ECC_JOB_TAG", str(args.job_tag))

    t0 = time.perf_counter()
    k, meta = compile_ecdlp(
        backend_name=str(args.backend),
        bit_length=int(args.bit_length),
        strategy=str(args.strategy),
        circuit_profile=str(args.circuit_profile),
        opt_level=int(args.opt_level),
        transpile_mode=str(args.transpile_mode),
        seed_transpiler=int(args.seed_transpiler),
        num_processes=int(args.num_processes),
        compile_only=bool(args.compile_only),
        shots=int(args.shots),
    )
    wall = float(time.perf_counter() - t0)

    out: Dict[str, Any] = {
        "backend": str(args.backend),
        "bit_length": int(args.bit_length),
        "strategy": str(args.strategy),
        "circuit_profile": str(args.circuit_profile),
        "opt_level": int(args.opt_level),
        "transpile_mode": str(args.transpile_mode),
        "seed_transpiler": int(args.seed_transpiler),
        "num_processes": int(args.num_processes),
        "shots": int(args.shots),
        "compile_only": bool(args.compile_only),
        "run_hardware": bool(args.run_hardware),
        "job_tag": (str(args.job_tag) if args.job_tag else None),
        "wall_seconds": wall,
        "success": True,
        "found_k": int(k) if k is not None else None,
        "meta": meta,
    }

    expected_k = None
    if isinstance(meta, dict):
        expected_k = meta.get("expected_k")
    if expected_k is not None and k is not None:
        out["k_matches_expected"] = bool(int(k) == int(expected_k))
    else:
        out["k_matches_expected"] = None

    out_path = Path(str(args.out_json))
    _write_json(out_path, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
