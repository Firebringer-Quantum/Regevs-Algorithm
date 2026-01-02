import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import QFT
from qiskit.transpiler import CouplingMap

try:
    from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
except Exception:  # pragma: no cover
    QiskitRuntimeService = None
    Sampler = None

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

try:
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    from hot_framework.compiler import HOTCompiler
    try:
        # Current hot_framework layout in this repo.
        from hot_framework.core import AlgorithmSpec, MeasurementPolicy, OrphanPolicy, SabrePolicy
    except Exception:
        # Fallback for alternate layouts.
        from hot_framework.spec import AlgorithmSpec
        from hot_framework.policy import MeasurementPolicy, OrphanPolicy, SabrePolicy
except Exception as e:  # pragma: no cover
    HOTCompiler = None
    AlgorithmSpec = None
    MeasurementPolicy = None
    OrphanPolicy = None
    SabrePolicy = None
    _HOT_IMPORT_ERROR = e
else:
    _HOT_IMPORT_ERROR = None


@dataclass
class DominanceSummary:
    expected_s: int
    expected_bitstring: str
    expected_rank: Optional[int]
    p_expected: float
    bit_reversed_expected: int
    bit_reversed_bitstring: str
    bit_reversed_rank: Optional[int]
    p_bit_reversed: float
    top_bitstring: str
    top_int: int
    top_rank: int
    p_top: float


def _parse_int_list(s: str) -> List[int]:
    s = (s or "").strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out: List[int] = []
    for p in parts:
        out.append(int(p))
    return out


def _connected_components(nodes: List[int], edges: List[Tuple[int, int]]) -> List[List[int]]:
    adj: Dict[int, set] = {n: set() for n in nodes}
    for a, b in edges:
        if a in adj and b in adj:
            adj[a].add(b)
            adj[b].add(a)

    comps: List[List[int]] = []
    seen = set()
    for start in nodes:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj.get(u, ()): 
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        comps.append(sorted(comp))
    return comps


def _load_ecc_private_key(*, ecc_curves_path: Path, bit_length: int) -> int:
    with open(ecc_curves_path, "r", encoding="utf-8") as f:
        curves = json.load(f)
    row = next((r for r in curves if int(r.get("bit_length", -1)) == int(bit_length)), None)
    if row is None:
        raise RuntimeError(f"No {bit_length}-bit curve found in {ecc_curves_path}")
    return int(row.get("private_key", 0))


def build_regev_toy_circuit(*, n_solution_qubits: int, secret_s: int, iqft_do_swaps: bool, measure_reverse: bool) -> QuantumCircuit:
    y = QuantumRegister(n_solution_qubits, "y")
    c = ClassicalRegister(n_solution_qubits, "c")
    qc = QuantumCircuit(y, c, name="regev_strategy_bench")

    for q in y:
        qc.h(q)

    secret_s = int(secret_s) % (2**n_solution_qubits)

    # Treat y[0] as LSB.
    for j, q in enumerate(y):
        denom = 2 ** (n_solution_qubits - j)
        theta = 2.0 * math.pi * (secret_s / denom)
        theta_wrapped = ((theta + math.pi) % (2.0 * math.pi)) - math.pi
        qc.rz(theta_wrapped, q)

    qc.append(QFT(num_qubits=n_solution_qubits, inverse=True, do_swaps=iqft_do_swaps).to_instruction(), y)

    if measure_reverse:
        qc.measure(list(reversed(y)), c)
    else:
        qc.measure(y, c)

    return qc


def _counts_to_dominance(*, counts: Dict[str, int], shots: int, expected_s: int, n_solution_qubits: int) -> DominanceSummary:
    expected_bitstring = format(expected_s, f"0{n_solution_qubits}b")

    sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    top_bitstring, top_count = sorted_counts[0]

    expected_rank = next((i + 1 for i, (bs, _) in enumerate(sorted_counts) if bs == expected_bitstring), None)
    p_expected = counts.get(expected_bitstring, 0) / max(1, shots)

    bit_reversed_expected = int(expected_bitstring[::-1], 2)
    bit_reversed_bitstring = format(bit_reversed_expected, f"0{n_solution_qubits}b")
    bit_reversed_rank = next((i + 1 for i, (bs, _) in enumerate(sorted_counts) if bs == bit_reversed_bitstring), None)
    p_bit_reversed = counts.get(bit_reversed_bitstring, 0) / max(1, shots)

    return DominanceSummary(
        expected_s=expected_s,
        expected_bitstring=expected_bitstring,
        expected_rank=expected_rank,
        p_expected=p_expected,
        bit_reversed_expected=bit_reversed_expected,
        bit_reversed_bitstring=bit_reversed_bitstring,
        bit_reversed_rank=bit_reversed_rank,
        p_bit_reversed=p_bit_reversed,
        top_bitstring=top_bitstring,
        top_int=int(top_bitstring, 2),
        top_rank=1,
        p_top=top_count / max(1, shots),
    )


def _compile_sabre(
    qc: QuantumCircuit,
    backend,
    opt_level: int,
    seed_transpiler: Optional[int],
    initial_layout: Optional[List[int]],
    coupling_map: Optional[CouplingMap],
):
    return transpile(
        qc,
        backend=backend,
        coupling_map=coupling_map,
        optimization_level=opt_level,
        seed_transpiler=seed_transpiler,
        initial_layout=initial_layout,
    )


def _compile_hot(
    qc: QuantumCircuit,
    backend,
    hot_compiler,
    opt_level: int,
    seed_transpiler: Optional[int],
    initial_layout: Optional[List[int]],
    enable_sabre: bool,
    isa_retranspile: bool,
    coupling_map: Optional[CouplingMap],
):
    if HOTCompiler is None or hot_compiler is None:
        raise RuntimeError(f"hot_framework is not available: {_HOT_IMPORT_ERROR}")

    sabre_policy = SabrePolicy(
        enabled=bool(enable_sabre),
        optimization_level=int(opt_level),
        seed_transpiler=seed_transpiler,
        layout_method="trivial" if initial_layout else "sabre",
        routing_method="sabre",
    )

    # For this toy Regev circuit, measure only the solution register.
    measurement_policy = MeasurementPolicy(stagger_readout=False)
    orphan_policy = OrphanPolicy(mode="strict")

    # Let HOT create AlgorithmSpec internally (it requires fields like interaction_graph).
    # We re-transpile to the backend afterwards for ISA compliance.
    hot_result = hot_compiler.compile(
        qc,
        algorithm_class="fixed_unitary",
        orphan_policy=orphan_policy,
        measurement_policy=measurement_policy,
        sabre_policy=sabre_policy,
        selection_policy="minimize_error",
    )

    compiled = getattr(hot_result, "compiled_circuit", None)
    if compiled is None:
        raise RuntimeError("HOT compilation returned no compiled_circuit")

    if not isa_retranspile:
        return compiled

    # Ensure ISA compliance.
    # Important: preserve the HOT-selected island placement. If we re-transpile without
    # pinning, Qiskit may remap to arbitrary/default physical qubits (often 0..n-1),
    # which defeats the purpose of HOT island selection.
    pinned_layout = list(initial_layout) if initial_layout else None
    if pinned_layout is None:
        island = getattr(hot_result, "island", None)
        if island is not None:
            pinned_layout = (
                getattr(island, "physical_qubits", None)
                or getattr(island, "qubits", None)
                or getattr(island, "selected_qubits", None)
            )
    if isinstance(pinned_layout, (list, tuple)) and len(pinned_layout) == int(qc.num_qubits):
        return transpile(
            compiled,
            backend=backend,
            coupling_map=coupling_map,
            optimization_level=0,
            seed_transpiler=seed_transpiler,
            initial_layout=list(pinned_layout),
        )

    return transpile(
        compiled,
        backend=backend,
        coupling_map=coupling_map,
        optimization_level=0,
        seed_transpiler=seed_transpiler,
    )


def _circuit_metrics(qc: QuantumCircuit) -> Dict[str, Any]:
    active_qubits = set()
    try:
        for inst, qargs, _cargs in qc.data:
            for q in qargs:
                try:
                    active_qubits.add(qc.find_bit(q).index)
                except Exception:
                    pass
    except Exception:
        active_qubits = set()

    ops = {str(k): int(v) for k, v in qc.count_ops().items()}
    twoq = 0
    for name, count in ops.items():
        if name in {"cx", "cz", "ecr", "swap", "rzz"}:
            twoq += int(count)
    return {
        "num_qubits": int(qc.num_qubits),
        "active_num_qubits": int(len(active_qubits)) if active_qubits else int(qc.num_qubits),
        "active_qubit_indices": sorted(active_qubits) if active_qubits else None,
        "num_clbits": int(qc.num_clbits),
        "depth": int(qc.depth()),
        "size": int(qc.size()),
        "count_ops": ops,
        "two_qubit_gate_count": int(twoq),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ibm_fez")
    ap.add_argument("--bit-length", type=int, required=True)
    ap.add_argument("--n", type=int, required=True, help="Number of solution qubits")
    ap.add_argument("--strategy", choices=["sabre", "hot", "hot_sabre"], help="Single strategy to run")
    ap.add_argument(
        "--strategies",
        default=None,
        help="Comma-separated strategies to run in one process (e.g. sabre,hot,hot_sabre). Reuses service/backend/HOTCompiler.",
    )
    ap.add_argument("--opt-level", type=int, default=2)
    ap.add_argument("--seed-transpiler", type=int, default=1234)
    ap.add_argument("--shots", type=int, default=2048)
    ap.add_argument("--run-hardware", action="store_true")
    ap.add_argument(
        "--job-tag",
        default="",
        help="Optional tag/label to record alongside hardware job_id (and pass as IBM Runtime job tag when supported).",
    )
    ap.add_argument("--measure-reverse", action="store_true")
    ap.add_argument("--iqft-do-swaps", action="store_true")
    ap.add_argument("--ecc-curves-path", default=str(Path("tests") / "ECCCurves.json"))
    ap.add_argument(
        "--physical-qubits",
        default="",
        help="Comma-separated physical qubits to pin as initial_layout (must match --n).",
    )
    ap.add_argument(
        "--restrict-routing-to-physical-qubits",
        action="store_true",
        help="If set with --physical-qubits, restrict the transpiler coupling map to the induced subgraph on that set so routing cannot spill onto other device qubits.",
    )
    ap.add_argument(
        "--no-isa-retranspile",
        action="store_true",
        help="Skip the backend ISA re-transpile step after HOT compilation (faster, but may not be backend-runnable).",
    )
    ap.add_argument("--out-json", required=True)

    args = ap.parse_args()

    if load_dotenv is not None:
        dotenv_path = Path.cwd() / ".env"
        if dotenv_path.exists():
            load_dotenv(dotenv_path=dotenv_path, override=True)
        else:
            load_dotenv(override=True)

    ecc_curves_path = Path(args.ecc_curves_path)
    ecc_k = _load_ecc_private_key(ecc_curves_path=ecc_curves_path, bit_length=args.bit_length)
    secret_s = ecc_k % (2**args.n)

    qc = build_regev_toy_circuit(
        n_solution_qubits=args.n,
        secret_s=secret_s,
        iqft_do_swaps=bool(args.iqft_do_swaps),
        measure_reverse=bool(args.measure_reverse),
    )

    physical_qubits = _parse_int_list(args.physical_qubits)
    if physical_qubits and len(physical_qubits) != int(args.n):
        raise ValueError("--physical-qubits length must match --n")

    if args.strategies:
        strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    elif args.strategy:
        strategies = [args.strategy]
    else:
        raise ValueError("Must provide --strategy or --strategies")

    if len(strategies) > 1 and "{strategy}" not in args.out_json:
        raise ValueError("When using --strategies, --out-json must contain '{strategy}' placeholder")

    if QiskitRuntimeService is None or Sampler is None:
        raise RuntimeError("qiskit-ibm-runtime is required")

    token = os.getenv("IBM_QUANTUM_TOKEN")
    crn = os.getenv("IBM_QUANTUM_CRN")
    if not token or not crn:
        raise RuntimeError("IBM_QUANTUM_TOKEN / IBM_QUANTUM_CRN must be set in environment")

    timing_global: Dict[str, float] = {}
    t_svc = time.perf_counter()
    service = QiskitRuntimeService(channel="ibm_cloud", token=token, instance=crn)
    backend = service.backend(args.backend)
    timing_global["service_backend_seconds"] = float(time.perf_counter() - t_svc)

    restricted_coupling = None
    if physical_qubits and args.restrict_routing_to_physical_qubits:
        try:
            edges: List[Tuple[int, int]] = []
            cfg = backend.configuration()
            for a, b in getattr(cfg, "coupling_map", []) or []:
                if a in physical_qubits and b in physical_qubits:
                    edges.append((a, b))
                    edges.append((b, a))

            # Validate connectivity of the induced subgraph.
            undirected_edges = []
            for a, b in edges:
                if a <= b:
                    undirected_edges.append((a, b))
            comps = _connected_components(list(physical_qubits), undirected_edges)
            if len(comps) != 1:
                raise ValueError(
                    "--restrict-routing-to-physical-qubits requires the provided physical-qubits set to be connected on the backend coupling map. "
                    f"Got {len(comps)} components: {comps}"
                )

            restricted_coupling = CouplingMap(edges)
        except Exception:
            restricted_coupling = None

    hot_compiler = None
    if any(s in {"hot", "hot_sabre"} for s in strategies):
        if HOTCompiler is None:
            raise RuntimeError(f"hot_framework is not available: {_HOT_IMPORT_ERROR}")
        t_hot_init = time.perf_counter()
        hot_compiler = HOTCompiler(backend)
        timing_global["hot_compiler_init_seconds"] = float(time.perf_counter() - t_hot_init)

    wrote_any = False
    for strategy in strategies:
        job_tag = (args.job_tag or "").strip() or None
        out: Dict[str, Any] = {
            "backend": args.backend,
            "bit_length": int(args.bit_length),
            "n_solution_qubits": int(args.n),
            "strategy": strategy,
            "opt_level": int(args.opt_level),
            "seed_transpiler": int(args.seed_transpiler),
            "shots": int(args.shots),
            "ecc_private_key": int(ecc_k),
            "secret_s": int(secret_s),
            "measure_reverse": bool(args.measure_reverse),
            "iqft_do_swaps": bool(args.iqft_do_swaps),
            "isa_retranspile": (not bool(args.no_isa_retranspile)),
            "physical_qubits": physical_qubits or None,
            "restrict_routing_to_physical_qubits": bool(args.restrict_routing_to_physical_qubits),
            "pre_metrics": _circuit_metrics(qc),
            "timing": {"global": dict(timing_global)},
            "success": False,
        }

        # Compilation
        if strategy == "sabre":
            t0 = time.perf_counter()
            transpiled = _compile_sabre(
                qc,
                backend=backend,
                opt_level=args.opt_level,
                seed_transpiler=args.seed_transpiler,
                initial_layout=(physical_qubits or None),
                coupling_map=restricted_coupling,
            )
            out["timing"]["compile_seconds"] = float(time.perf_counter() - t0)
        elif strategy == "hot":
            t0 = time.perf_counter()
            transpiled = _compile_hot(
                qc,
                backend=backend,
                hot_compiler=hot_compiler,
                opt_level=args.opt_level,
                seed_transpiler=args.seed_transpiler,
                initial_layout=(physical_qubits or None),
                enable_sabre=False,
                isa_retranspile=(not bool(args.no_isa_retranspile)),
                coupling_map=restricted_coupling,
            )
            out["timing"]["compile_seconds"] = float(time.perf_counter() - t0)
        else:
            t0 = time.perf_counter()
            transpiled = _compile_hot(
                qc,
                backend=backend,
                hot_compiler=hot_compiler,
                opt_level=args.opt_level,
                seed_transpiler=args.seed_transpiler,
                initial_layout=(physical_qubits or None),
                enable_sabre=True,
                isa_retranspile=(not bool(args.no_isa_retranspile)),
                coupling_map=restricted_coupling,
            )
            out["timing"]["compile_seconds"] = float(time.perf_counter() - t0)

        out["compiled_metrics"] = _circuit_metrics(transpiled)

        out_path = args.out_json.replace("{strategy}", strategy)

        if not args.run_hardware:
            out["success"] = True
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
            wrote_any = True
            continue

        sampler = Sampler(backend)
        # Persist runtime metadata for traceability.
        runtime_job: Dict[str, Any] = {
            "tag": job_tag,
        }

        # Some qiskit-ibm-runtime versions support job_tags=... on primitive runs.
        try:
            if job_tag:
                job = sampler.run([transpiled], shots=args.shots, job_tags=[job_tag])
            else:
                job = sampler.run([transpiled], shots=args.shots)
        except TypeError:
            job = sampler.run([transpiled], shots=args.shots)

        out["job_id"] = job.job_id()
        runtime_job["job_id"] = out["job_id"]
        out["runtime_job"] = runtime_job

        result = job.result()
        pub = result[0]
        counts = pub.data.c.get_counts()

        out["counts_top25"] = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:25]
        out["dominance"] = asdict(_counts_to_dominance(counts=counts, shots=args.shots, expected_s=secret_s, n_solution_qubits=args.n))
        out["success"] = True

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        wrote_any = True

    return 0 if wrote_any else 1


if __name__ == "__main__":
    raise SystemExit(main())
