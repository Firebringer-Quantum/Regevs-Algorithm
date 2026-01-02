# Regev Hardware Benchmarks (Collateral)

## Signal-to-Noise Ratio (SNR) via Hamming-distance

We quantify “signal vs noise” by measuring the Hamming distance of observed bitstrings from the expected secret `s` (expressed as an `n`-bit string). For each run we build a histogram over distances:

- Distance 0: exact match to the expected secret bitstring.
- Distance 1: single-bit errors near the secret.
- Distance ≥2: broader noise.

We summarize with two simple ratios:

- `SNR(d0/rest) = count(d=0) / count(d>0)`
- `SNR(d0/d1)   = count(d=0) / count(d=1)`

This is computed from the benchmark JSON outputs (ideally using full `counts`; otherwise approximated from `counts_top25`).

### Hamming Density Plot

A “Hamming density plot” is the histogram of counts vs Hamming distance to the expected secret. On successful runs (e.g. 7-bit with `--iqft-do-swaps`, and the 21-bit solve), the plot shows:

- a dominant spike at Distance 0
- a smaller cluster at Distance 1 (nearest neighbors)
- a long tail beyond

The script `scripts/regev_hamming_snr.py` generates these plots.

## Hardware Optimization (Depth 763 on ibm_fez / Heron R2)

### Problem framing

The key constraint for Regev-style phase-estimation circuits on real hardware is not only depth, but *preserving the intended basis convention* end-to-end. In early runs, the circuit executed but the measured bitstrings did not correspond to the intended secret because the inverse-QFT output ordering was not corrected.

### Key decisions that preserved the state

- **IQFT output convention fixed:** We enabled `iqft_do_swaps=true`. This applies the standard QFT output reordering inside the IQFT itself. Without this, the circuit’s secret is effectively permuted at readout, producing “failures” even when coherence is adequate.

- **Compilation strategy:** We used Qiskit’s SABRE-based preset routing at `optimization_level=2` on `ibm_fez`.

- **Island selection by transpiler:** For the 21-qubit run, the transpiler selected a connected 21-qubit subset (an “island”) within the 156-qubit device that satisfied connectivity needs for the IQFT decomposition and produced a depth of **763** with **888** two-qubit gates.

- **Measurement hygiene:** Measurements are placed only at the end of the circuit, minimizing mid-circuit readout disturbance.

### Result

For the 21-bit run:

- Compiled depth: **763**
- Two-qubit gates: **888**
- Expected secret is **rank 1** in the output distribution
- IBM Runtime `job_id`: `d5bme50nsj9s73b8pi30`

Even though the absolute probability mass on the secret is small at 21 qubits (as expected on NISQ hardware), rank-1 dominance indicates the compiled circuit retained sufficient structure to recover the planted secret.

## Benchmarking argument (positioning the 21-bit solve)

### What we can claim directly from artifacts

From the stored artifacts (benchmark JSON + job id), we can verify:

- The run executed on `ibm_fez` (Heron R2 family)
- The compiled circuit depth and 2Q gate count
- The expected secret appears as the most frequent measurement outcome (rank 1)

### Public-record positioning (draft)

To position this result as a top public hardware record, we need to be careful about wording unless we cite an up-to-date leaderboard.

A safe draft framing is:

- This is **one of the largest publicly verifiable Regev-style toy ECDLP phase-estimation demonstrations** we are aware of on IBM public hardware, because:
  - it uses a real backend (`ibm_fez`)
  - it solves an ECC-derived secret at **21 bits**
  - the secret is **rank 1** in measured outputs
  - the exact job id is provided for independent verification

If you want a stronger claim (“top of current public records”), we should add a short citation section summarizing comparable public demos and their bit-lengths.

## Clean visualizations

Use:

```powershell
python scripts\regev_hamming_snr.py benchmarks\regev_fez_7bit_n7_iqftswaps_sabre.o2.json benchmarks\regev_fez_21bit_n21_iqftswaps_sabre.o2.json --out-dir artifacts\regev_analysis
```

This produces:

- `*.hamming_density.png` (Hamming distance density)
- `*.top_outcomes.png` (top outcomes bar chart)
- `regev_hamming_snr_summary.json` (numbers suitable for a paper appendix)
