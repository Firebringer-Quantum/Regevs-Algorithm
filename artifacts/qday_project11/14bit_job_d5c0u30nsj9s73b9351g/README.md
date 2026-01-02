# QDay Challenge — Project 11

This folder is a **push-ready artifact bundle** for the QDay Challenge submission (Project 11). It contains the raw IBM Quantum Runtime job payloads for an on-hardware run, plus the minimum surrounding context needed to reproduce and validate the result.

## Paper (publish-ready)

The publish-ready paper is stored at:

- `benchmarks/ECDLP_Quantum_Cryptanalysis_Paper.docx`

This README is derived from that document and tailored to the QDay artifact-review workflow.

## Run identity

- **Backend**: `ibm_fez`
- **Primitive**: `Sampler` (Runtime)
- **Job ID**: `d5c0u30nsj9s73b9351g`
- **Shots**: `8092`

## What this run demonstrates (summary)

This job is an on-hardware execution of our quantum period-finding approach applied to ECC (as described in the paper).

The paper emphasizes that **cryptanalytic success** is not limited to “the correct key is the dominant measurement outcome.” Instead, a run can be considered cryptanalytically successful if the post-processing yields a **candidate pool** that contains the correct key (classically verifiable).

## Artifact inventory (this folder)

- `ecdlp_full_arithmetic_bench.py`
  - Standalone runner used to generate structured benchmark JSON outputs.
- `job-d5c0u30nsj9s73b9351g-info.json`
  - Runtime metadata for the job, including a serialized representation of the submitted circuit payload.
- `job-d5c0u30nsj9s73b9351g-result.json`
  - Raw sampler result payload.
- `quantum_ecc_solver_proper.py`
  - Full arithmetic ECDLP implementation (circuit construction, compilation, and post-processing).
- `README.md`
  - This document.

## Related structured summary JSON (recommended)

In the main `benchmarks/` directory, the repo includes a structured run summary with compiler metrics and dominance diagnostics:

- `benchmarks/diagnostic.ecdlp_fez_14bit_sabre.pure.o2.lowanc.json`

Fields of interest:

- `meta.job_id` (should match `d5c0u30nsj9s73b9351g`)
- `meta.compiled_circuit_metrics*` (depth, 2Q gate count, active qubits)
- `meta.dominance_diagnostics` (candidate pool size, top candidates, expected-key support/rank)

## How to reproduce (hardware)

Prereqs:

- An IBM Quantum token available as `IBM_QUANTUM_TOKEN`.

Python deps:

```powershell
python -m pip install -r requirements.txt
```

Typical command pattern used in this repo:

```powershell
$env:ECC_ALLOW_LOW_ANCILLA="1"
python scripts\ecdlp_full_arithmetic_bench.py \
  --backend ibm_fez \
  --bit-length 14 \
  --strategy sabre \
  --circuit-profile pure \
  --opt-level 2 \
  --transpile-mode preset \
  --run-hardware \
  --shots 8092 \
  --out-json benchmarks\diagnostic.ecdlp_fez_14bit_sabre.pure.o2.lowanc.json
```

## How to reproduce (offline, from job id)

If you do not want to re-run hardware, you can pull the result again by job id and re-run post-processing locally.

```powershell
python -c "import os; from qiskit_ibm_runtime import QiskitRuntimeService; s=QiskitRuntimeService(channel='ibm_quantum', token=os.environ['IBM_QUANTUM_TOKEN']); job=s.job('d5c0u30nsj9s73b9351g'); r=job.result(); print('Got result:', type(r))"
```

The structured, reviewer-friendly metrics (candidate pool size, dominance diagnostics, etc.) are stored in:

- `benchmarks/diagnostic.ecdlp_fez_14bit_sabre.pure.o2.lowanc.json`

## Making this folder fully self-contained (recommended)

For a standalone GitHub artifact bundle, copy the following into this folder as well:

- `benchmarks/ECDLP_Quantum_Cryptanalysis_Paper.docx`
- `benchmarks/diagnostic.ecdlp_fez_14bit_sabre.pure.o2.lowanc.json`
- `scripts/regev_strategy_bench.py`
- `scripts/regev_hamming_snr.py`
- `docs/regev_collateral.md`
- `scripts/package_regev_runs.py`

Suggested copy block:

```powershell
$dst = 'benchmarks\14 bit Regev - job-d5c0u30nsj9s73b9351g'
Copy-Item -Force -LiteralPath 'benchmarks\ECDLP_Quantum_Cryptanalysis_Paper.docx' -Destination $dst
Copy-Item -Force -LiteralPath 'benchmarks\diagnostic.ecdlp_fez_14bit_sabre.pure.o2.lowanc.json' -Destination $dst
Copy-Item -Force -LiteralPath 'scripts\regev_strategy_bench.py' -Destination $dst
Copy-Item -Force -LiteralPath 'scripts\regev_hamming_snr.py' -Destination $dst
Copy-Item -Force -LiteralPath 'docs\regev_collateral.md' -Destination $dst
Copy-Item -Force -LiteralPath 'scripts\package_regev_runs.py' -Destination $dst
```

## Notes for reviewers

- The raw Runtime payloads here are sufficient to confirm job identity (`job_id`, backend, timestamps) and to re-run post-processing.
- The paper contains the full narrative, experimental design, and interpretation guidance.

## Visualizations (paper-ready)

This artifact bundle includes a reproducible plotting script:

- `make_ecdlp_plots.py`

It generates:

- `expected_k_prob_vs_depth.png`
  - The “money shot”: expected-key support (probability) vs compiled physical depth.
- `candidate_dissipation_heatmap.png`
  - Heatmap of candidate probability mass by rank (log-scaled) across bit-lengths.
- `plot_data.ecdlp_signal_decay.json`
  - The extracted table used to make the plots.

### Generate plots

```powershell
python make_ecdlp_plots.py --benchmarks-dir .. --out-dir .
```

Notes:

- For 12–14 bit runs, the script reads `expected_k_prob` and `top_candidates` directly from the `diagnostic.*.json` files.
- For 6/8/10/11 bit runs, the script re-fetches the Sampler results by `job_id` and recomputes dominance diagnostics locally.
  This requires `IBM_QUANTUM_TOKEN` (and optionally `IBM_QUANTUM_CRN`) to be set.
