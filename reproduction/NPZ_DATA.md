# Every retained NPZ archive

This inventory covers **all 64 NPZ files under `Code/`**, including ignored and hidden paths, as of 2026-09-21. All are retained run products or required failure/noncoverage evidence. One unselected Table 1 seed archive was removed from Code; the [cleanup ledger](../results/table1-seed-pruning-2026-09-21.json) records its recovery location. Protected `Paper/` and frozen archives are outside this cleanup scope.

The inventory contains **64 files, 8,287,147 bytes, in 9 run directories**. The [machine-readable map](../results/npz-map.json) gives full SHA-256 hashes, exact schemas, per-run generator hashes, profiles, seed/RNG/configuration evidence and any plot policy. Every file has a separate row below; shared schemas avoid repeating identical definitions 53 times.

| Classification | Files | Why retained |
| --- | ---: | --- |
| New primary/table/replot data | 10 | Current and prior table inputs, paired indicators, original NMA data and exact plot caches. |
| Completed NMA statistical noncoverage | 51 | Nonempty completed regions that do not cover truth; not numerical failures. |
| Statistical noncoverage within failed original run | 2 | Completed noncoverage outcomes before that separate run stopped. |
| Original failed-run partial archive | 1 | Preserved numerical-failure evidence, including unattempted NaNs. |

All current NMA simulation outcomes retain the full denominator: certified emptiness would count as noncoverage/zero width without study removal or resampling. The successful run had **zero empty sets**, 51 noncoverage outcomes and no numerical failures. The original failed attempt stopped at rho=0, replicate 114 after 113 completed outcomes; the failing outcome was **not classified as empty**. Its two `failed-coverage` snapshots are statistical outcomes, not the numerical failure itself.

## Locate the data you need

- Figure 10 raw results: `nma-100-reproduction-20260921` summary and replicate archives. Seed 2025, 100 replications per rho, n=100, df=99; manuscript/original description remains 500, and no new 500-replication run was made.
- Figure 10 comparison-only replot: `nma-100-manual-width-replot-20260921/nma-simulation-plotted-summary.npz` under `data/`. The manuscript Figure 10 is frozen to its existing paper PDFs unless the author explicitly requests replacement. Only the replot's two WLS-MA width series are historical manual values. Its sibling cache retains raw empirical values exactly. **Its empirical-oracle coverage does not validate the old manual widths**; their exact historical calibration/sample provenance remains unknown.
- Figure 14 paired uncertainty: original `paired-comparison-20260921` NPZs contain all 3.6 million Boolean decisions across FPR/power. The `paired-visibility-replot-20260921` caches contain rates only; they do not replace those indicators. These reruns are comparison-only: manuscript Figure 14 is frozen to its existing PDFs unless the author explicitly requests replacement.
- Figure 3c–d widths: `interval-width-layout-replot-20260921/data/replot-cache.npz` retains all 80,000 widths exactly; the original full run stores them as **NPY**, not NPZ.
- Main Table 1: `table-1-seed-2025-20260921/data/nma-simulation-treatment-inputs.npz` uses private PCG64 seed **2025**, one sample at each of four rhos. Its estimates are adopted in the manuscript. This table is separate from Figure 10.

Normal/copula/multivariate coverage simulations save NPY summaries rather than NPZs. See [SAVED_DATA.md](SAVED_DATA.md) for all NPY/NPZ axis definitions and replot commands, and the [single results index](../results/README.md#paper-figure-index) for PDFs in manuscript order (versions chronological within each panel). The separately authorized cleanup of 19 obsolete historical `Code/data` files (18 NPY and one Feather) is recorded in [its ledger](../data/pruning-2026-09-21.json); the five other-paper NPY inputs remain. That cleanup is not an NPZ deletion.

## Reading and uncertainty

```python
import numpy as np
with np.load(path, allow_pickle=False) as saved:
    print(saved.files)
    # Choose the member using the schema below; do not infer axes from shape alone.
```

For a paired method comparison, subtract Boolean decisions after converting to signed integers. The estimated gap is `difference.mean(axis=-1)` and empirical standard error is `difference.std(axis=-1, ddof=1) / sqrt(N)`. Zero observed discordances do not prove zero population uncertainty. Never treat NaNs in a failed partial archive as coverage indicators or average it as a completed experiment.

## Run provenance

### R1: network-meta-analysis/simulation / full-reproduction-20260921

Run record: [Code/results/runs/network-meta-analysis/simulation/full-reproduction-20260921/run.json](../results/runs/network-meta-analysis/simulation/full-reproduction-20260921/run.json). Status **failed**, profile `paper`, seed **2025**. Recorded experiment seed.

Generator: `Code/reproduction/network_meta_analysis/simulation.py`. RNG: numpy.random global RandomState/MT19937, seed once; rho-major, replicate-minor. Evidence: `recorded seed; original source snapshot and seeded replay for completed run`.

Configuration: `{"allocated_replicates_per_rho": 500, "completed_replicates": 113, "degrees_freedom": 99, "failed_replicate_1_based": 114, "failed_rho": 0.0, "level": 0.05, "parameter_dimension": 9, "sample_size": 100, "study_count": 28}`.

Configuration qualification: incomplete array shape, failure JSON and dated execution record; not an original run.json configuration.

### R2: network-meta-analysis/simulation / nma-100-manual-width-replot-20260921

Run record: [Code/results/runs/network-meta-analysis/simulation/nma-100-manual-width-replot-20260921/run.json](../results/runs/network-meta-analysis/simulation/nma-100-manual-width-replot-20260921/run.json). Status **completed**, profile `paper`, seed **2025**. Source simulation seed only; replot draws no samples.

Generator: `Code/reproduction/replot.py`. RNG: numpy.random global RandomState/MT19937, seed once; rho-major, replicate-minor. Evidence: `recorded seed; original source snapshot and seeded replay for completed run`.

Configuration: `{"degrees_freedom": 99, "level": 0.05, "parameter_dimension": 9, "profile": "paper", "replicates_per_rho": 100, "rhos": [0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9], "sample_size": 100, "schema_version": 1, "seed": 2025, "study_count": 28, "workload_revision": "nma-100-replicates-2026-09-21"}`.

Configuration qualification: run.json configuration/source_recorded_configuration.

Plot-only source: `Code/results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921`; no sampling or refitting. Full source-file and generator hashes are recorded in the JSON map and immutable run record.

The two manual plotted widths have separate archival provenance, not seed-2025 simulation provenance. All other plotted values and the empirical raw cache use the source run. The `empirical_wls_ma_multiplier` is not the calibration multiplier of those manual widths.

### R3: network-meta-analysis/simulation / nma-100-reproduction-20260921

Run record: [Code/results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921/run.json](../results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921/run.json). Status **completed**, profile `paper`, seed **2025**. Recorded experiment seed.

Generator: `Code/reproduction/network_meta_analysis/simulation.py`. RNG: numpy.random global RandomState/MT19937, seed once; rho-major, replicate-minor. Evidence: `recorded seed; original source snapshot and seeded replay for completed run`.

Configuration: `{"degrees_freedom": 99, "level": 0.05, "parameter_dimension": 9, "profile": "paper", "replicates_per_rho": 100, "rhos": [0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9], "sample_size": 100, "schema_version": 1, "seed": 2025, "study_count": 28, "workload_revision": "nma-100-replicates-2026-09-21"}`.

Configuration qualification: run.json configuration/source_recorded_configuration.

### R4: network-meta-analysis/treatment-estimates / table-1-seed-2025-20260921

Run record: [Code/results/runs/network-meta-analysis/treatment-estimates/table-1-seed-2025-20260921/run.json](../results/runs/network-meta-analysis/treatment-estimates/table-1-seed-2025-20260921/run.json). Status **completed**, profile `paper`, seed **2025**. Recorded experiment seed.

Generator: `Code/reproduction/network_meta_analysis/treatment_estimates.py`. RNG: numpy.random.default_rng / PCG64 Evidence: `diagnostics/nma-treatment-estimates.json simulation section`.

Configuration: `{"degrees_freedom": 99, "replicates_per_rho": 1, "rhos": [0.0, 0.3, 0.6, 0.9], "sample_size": 100, "study_count": 28}`.

Configuration qualification: diagnostics/nma-treatment-estimates.json simulation section.

### R5: one-dimensional/false-positive-rate / paired-comparison-20260921

Run record: [Code/results/runs/one-dimensional/false-positive-rate/paired-comparison-20260921/run.json](../results/runs/one-dimensional/false-positive-rate/paired-comparison-20260921/run.json). Status **completed**, profile `paper`, seed **2025**. Recorded experiment seed.

Generator: `Code/reproduction/one_dimensional/false_positive_rate.py`. RNG: RandomState/MT19937; rho-major, case-minor Evidence: `Code/results/runs/one-dimensional/false-positive-rate/paired-comparison-20260921/data/false-positive-rate-paired.npz`.

Configuration: `{"axes": ["method", "rho", "case", "draw"], "cases": ["ar1", "equicorrelated"], "level": 0.05, "methods": ["HCauchy", "EHMP", "HMP", "Cauchy", "Levy", "Fisher", "Stouffer", "Bonferroni", "Simes"], "rhos": [0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9], "sampling_scheme": "shared-across-methods; distinct batches across rho/case", "simulations_per_case": 10000, "study_count": 500}`.

Configuration qualification: Code/results/runs/one-dimensional/false-positive-rate/paired-comparison-20260921/data/false-positive-rate-paired.npz.

### R6: one-dimensional/false-positive-rate / paired-visibility-replot-20260921

Run record: [Code/results/runs/one-dimensional/false-positive-rate/paired-visibility-replot-20260921/run.json](../results/runs/one-dimensional/false-positive-rate/paired-visibility-replot-20260921/run.json). Status **completed**, profile `paper`, seed **2025**. Source simulation seed only; replot draws no samples.

Generator: `Code/reproduction/replot.py`. RNG: RandomState/MT19937; rho-major, case-minor Evidence: `Code/results/runs/one-dimensional/false-positive-rate/paired-comparison-20260921/data/false-positive-rate-paired.npz`.

Configuration: `{"axes": ["method", "rho", "case", "draw"], "cases": ["ar1", "equicorrelated"], "level": 0.05, "methods": ["HCauchy", "EHMP", "HMP", "Cauchy", "Levy", "Fisher", "Stouffer", "Bonferroni", "Simes"], "rhos": [0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9], "sampling_scheme": "shared-across-methods; distinct batches across rho/case", "simulations_per_case": 10000, "study_count": 500}`.

Configuration qualification: Code/results/runs/one-dimensional/false-positive-rate/paired-comparison-20260921/data/false-positive-rate-paired.npz.

Plot-only source: `Code/results/runs/one-dimensional/false-positive-rate/paired-comparison-20260921`; no sampling or refitting. Full source-file and generator hashes are recorded in the JSON map and immutable run record.

### R7: one-dimensional/interval-width / interval-width-layout-replot-20260921

Run record: [Code/results/runs/one-dimensional/interval-width/interval-width-layout-replot-20260921/run.json](../results/runs/one-dimensional/interval-width/interval-width-layout-replot-20260921/run.json). Status **completed**, profile `paper`, seed **2025**. Source simulation seed only; replot draws no samples.

Generator: `Code/reproduction/replot.py`. RNG: numpy.random global RandomState/MT19937; rho-major, AR1 then equicorrelation. Evidence: `documented source draw order; source seed in run.json; no RNG used by replot`.

Configuration: `{"draws_per_case": 10000, "level": 0.05, "num_study": 500}`.

Configuration qualification: Current documented profile interpretation, not proof of original m/n/df settings. Legacy NPY does not store configuration; retain original run/code hashes and any source-recorded settings. Validation is structural, not numerical recalibration..

Plot-only source: `Code/results/runs/one-dimensional/interval-width/full-reproduction-20260921`; no sampling or refitting. Full source-file and generator hashes are recorded in the JSON map and immutable run record.

### R8: one-dimensional/power / paired-comparison-20260921

Run record: [Code/results/runs/one-dimensional/power/paired-comparison-20260921/run.json](../results/runs/one-dimensional/power/paired-comparison-20260921/run.json). Status **completed**, profile `paper`, seed **2025**. Recorded experiment seed.

Generator: `Code/reproduction/one_dimensional/power_comparison.py`. RNG: RandomState/MT19937; rho-major, case-minor Evidence: `Code/results/runs/one-dimensional/power/paired-comparison-20260921/data/power-comparison-paired.npz`.

Configuration: `{"axes": ["method", "rho", "case", "draw"], "cases": ["dense", "sparse"], "level": 0.05, "methods": ["HCauchy", "EHMP", "HMP", "Cauchy", "Levy", "Fisher", "Stouffer", "Bonferroni", "Simes"], "rhos": [0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9], "sampling_scheme": "shared-across-methods; distinct batches across rho/case", "simulations_per_case": 10000, "study_count": 500}`.

Configuration qualification: Code/results/runs/one-dimensional/power/paired-comparison-20260921/data/power-comparison-paired.npz.

### R9: one-dimensional/power / paired-visibility-replot-20260921

Run record: [Code/results/runs/one-dimensional/power/paired-visibility-replot-20260921/run.json](../results/runs/one-dimensional/power/paired-visibility-replot-20260921/run.json). Status **completed**, profile `paper`, seed **2025**. Source simulation seed only; replot draws no samples.

Generator: `Code/reproduction/replot.py`. RNG: RandomState/MT19937; rho-major, case-minor Evidence: `Code/results/runs/one-dimensional/power/paired-comparison-20260921/data/power-comparison-paired.npz`.

Configuration: `{"axes": ["method", "rho", "case", "draw"], "cases": ["dense", "sparse"], "level": 0.05, "methods": ["HCauchy", "EHMP", "HMP", "Cauchy", "Levy", "Fisher", "Stouffer", "Bonferroni", "Simes"], "rhos": [0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9], "sampling_scheme": "shared-across-methods; distinct batches across rho/case", "simulations_per_case": 10000, "study_count": 500}`.

Configuration qualification: Code/results/runs/one-dimensional/power/paired-comparison-20260921/data/power-comparison-paired.npz.

Plot-only source: `Code/results/runs/one-dimensional/power/paired-comparison-20260921`; no sampling or refitting. Full source-file and generator hashes are recorded in the JSON map and immutable run record.

## Per-file locations and integrity

Paths in each group are relative to the linked run directory. SHA-256 values are complete, not abbreviated. Every group inherits its run's generator/profile/seed/RNG/configuration above; every row links its exact key/shape/dtype schema below.

### Files in R1

Run directory: `Code/results/runs/network-meta-analysis/simulation/full-reproduction-20260921`.

| File | Bytes | SHA-256 | Schema | Classification |
| --- | ---: | --- | --- | --- |
| [diagnostics/failed-coverage-rho-0.0-replicate-47.npz](../results/runs/network-meta-analysis/simulation/full-reproduction-20260921/diagnostics/failed-coverage-rho-0.0-replicate-47.npz) | 3,528 | `141fb7d4d25da1ff6916f7d6a4c1b84d9d3a48059cc26aebe9e9036f47804866` | [statistical-noncoverage](#statistical-noncoverage) | failed-run-statistical-noncoverage |
| [diagnostics/failed-coverage-rho-0.0-replicate-79.npz](../results/runs/network-meta-analysis/simulation/full-reproduction-20260921/diagnostics/failed-coverage-rho-0.0-replicate-79.npz) | 3,528 | `52be4fdb091edf86b21b6626b257c055f648acefde5c0dc625a5b17f7585f3f9` | [statistical-noncoverage](#statistical-noncoverage) | failed-run-statistical-noncoverage |
| [diagnostics/nma-simulation-partial-replicates.npz](../results/runs/network-meta-analysis/simulation/full-reproduction-20260921/diagnostics/nma-simulation-partial-replicates.npz) | 5,806,898 | `757002757a8cd313b7274614933d732e84d93dd135bc3034e69dc06cb03fce85` | [failed-partial](#failed-partial) | failed-run-partial-evidence |

### Files in R2

Run directory: `Code/results/runs/network-meta-analysis/simulation/nma-100-manual-width-replot-20260921`.

| File | Bytes | SHA-256 | Schema | Classification |
| --- | ---: | --- | --- | --- |
| [data/nma-simulation-plotted-summary.npz](../results/runs/network-meta-analysis/simulation/nma-100-manual-width-replot-20260921/data/nma-simulation-plotted-summary.npz) | 3,384 | `9f87c46a8ee8b76368645ea56170229204ee14b7e5e27eaf57189848c2ec7536` | [nma-plotted-summary](#nma-plotted-summary) | new-run-data |
| [data/replot-cache.npz](../results/runs/network-meta-analysis/simulation/nma-100-manual-width-replot-20260921/data/replot-cache.npz) | 4,215 | `3a04fcb23ba0f353f9ee78a0046dcbcdc6215cbae7361977bc86eb6e82744e4a` | [cache-nma](#cache-nma) | new-run-data |

### Files in R3

Run directory: `Code/results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921`.

| File | Bytes | SHA-256 | Schema | Classification |
| --- | ---: | --- | --- | --- |
| [data/nma-simulation-replicates.npz](../results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921/data/nma-simulation-replicates.npz) | 1,166,898 | `d94bbcf365b040abc09d1585a99239a27d4c33dc9a8f5794f8a7662b708f149d` | [nma-replicates](#nma-replicates) | new-run-data |
| [data/nma-simulation-summary.npz](../results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921/data/nma-simulation-summary.npz) | 4,416 | `25c60809e1b6680801ef092a6e18a01a4d6f683519118581fdef34f954da07cc` | [nma-summary](#nma-summary) | new-run-data |
| [diagnostics/failed-coverage-rho-0.0-replicate-47.npz](../results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921/diagnostics/failed-coverage-rho-0.0-replicate-47.npz) | 3,528 | `141fb7d4d25da1ff6916f7d6a4c1b84d9d3a48059cc26aebe9e9036f47804866` | [statistical-noncoverage](#statistical-noncoverage) | completed-run-statistical-noncoverage |
| [diagnostics/failed-coverage-rho-0.0-replicate-79.npz](../results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921/diagnostics/failed-coverage-rho-0.0-replicate-79.npz) | 3,528 | `52be4fdb091edf86b21b6626b257c055f648acefde5c0dc625a5b17f7585f3f9` | [statistical-noncoverage](#statistical-noncoverage) | completed-run-statistical-noncoverage |
| [diagnostics/failed-coverage-rho-0.1-replicate-11.npz](../results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921/diagnostics/failed-coverage-rho-0.1-replicate-11.npz) | 3,528 | `13b578be9730713cc74d121f9132148b1bcaf8ea301d8dcb984d83fe52f2004d` | [statistical-noncoverage](#statistical-noncoverage) | completed-run-statistical-noncoverage |
| [diagnostics/failed-coverage-rho-0.1-replicate-38.npz](../results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921/diagnostics/failed-coverage-rho-0.1-replicate-38.npz) | 3,528 | `5e37c1e795e6d7759c26f1dda20982ae8524f8da9db7e13d976b18aae2094a29` | [statistical-noncoverage](#statistical-noncoverage) | completed-run-statistical-noncoverage |
| [diagnostics/failed-coverage-rho-0.1-replicate-92.npz](../results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921/diagnostics/failed-coverage-rho-0.1-replicate-92.npz) | 3,528 | `142ad3b82a8e7f4027c93e13f7267a49007384868794d407d6ec370274c21840` | [statistical-noncoverage](#statistical-noncoverage) | completed-run-statistical-noncoverage |
| [diagnostics/failed-coverage-rho-0.1-replicate-94.npz](../results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921/diagnostics/failed-coverage-rho-0.1-replicate-94.npz) | 3,528 | `2eb40952256c1ff168d16addb1296897a85c559007ad12517bc0b1fa79b49256` | [statistical-noncoverage](#statistical-noncoverage) | completed-run-statistical-noncoverage |
| [diagnostics/failed-coverage-rho-0.2-replicate-1.npz](../results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921/diagnostics/failed-coverage-rho-0.2-replicate-1.npz) | 3,528 | `37f4aaece73a53b0e2c23a53f5248611fd407c97fe1079b6db06566293fc8b60` | [statistical-noncoverage](#statistical-noncoverage) | completed-run-statistical-noncoverage |
| [diagnostics/failed-coverage-rho-0.2-replicate-47.npz](../results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921/diagnostics/failed-coverage-rho-0.2-replicate-47.npz) | 3,528 | `ae1347c38aa462f73cb407392610ba47ef0705f96deb0d8e3a43f54c90e5867f` | [statistical-noncoverage](#statistical-noncoverage) | completed-run-statistical-noncoverage |
| [diagnostics/failed-coverage-rho-0.2-replicate-5.npz](../results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921/diagnostics/failed-coverage-rho-0.2-replicate-5.npz) | 3,528 | `326257eacf8fa63a96c776f2d2151ba1c1290071f379ff9cef7df124350d2785` | [statistical-noncoverage](#statistical-noncoverage) | completed-run-statistical-noncoverage |
| [diagnostics/failed-coverage-rho-0.2-replicate-52.npz](../results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921/diagnostics/failed-coverage-rho-0.2-replicate-52.npz) | 3,528 | `09bacf3502d474cbde8853776a906254f4155cef5746f4f7f6f03de03222b44b` | [statistical-noncoverage](#statistical-noncoverage) | completed-run-statistical-noncoverage |
| [diagnostics/failed-coverage-rho-0.2-replicate-56.npz](../results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921/diagnostics/failed-coverage-rho-0.2-replicate-56.npz) | 3,528 | `93aa0c19d7335c61f9ce074ed1265ccd77af0eaad5366a08cdff036fa9e09463` | [statistical-noncoverage](#statistical-noncoverage) | completed-run-statistical-noncoverage |
| [diagnostics/failed-coverage-rho-0.2-replicate-61.npz](../results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921/diagnostics/failed-coverage-rho-0.2-replicate-61.npz) | 3,528 | `138f5eb20e3d178c501fff23e5812893cf59aeb2c145c09aaed15090e0ddefcf` | [statistical-noncoverage](#statistical-noncoverage) | completed-run-statistical-noncoverage |
| [diagnostics/failed-coverage-rho-0.2-replicate-66.npz](../results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921/diagnostics/failed-coverage-rho-0.2-replicate-66.npz) | 3,528 | `72a01dcd157a162bd344cd05ed6bf92bb9dd8e2a35ebf8a79abc56407484f4f9` | [statistical-noncoverage](#statistical-noncoverage) | completed-run-statistical-noncoverage |
| [diagnostics/failed-coverage-rho-0.2-replicate-7.npz](../results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921/diagnostics/failed-coverage-rho-0.2-replicate-7.npz) | 3,528 | `3731d3287fcf757b38e585f70cf30b94626b8b1af806f3619c5c1f9e196263ad` | [statistical-noncoverage](#statistical-noncoverage) | completed-run-statistical-noncoverage |
| [diagnostics/failed-coverage-rho-0.2-replicate-76.npz](../results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921/diagnostics/failed-coverage-rho-0.2-replicate-76.npz) | 3,528 | `868a2d7000c892e8d10f99859fcbb5e24d32c93c68fc92f84ba8f90fc9692c3c` | [statistical-noncoverage](#statistical-noncoverage) | completed-run-statistical-noncoverage |
| [diagnostics/failed-coverage-rho-0.2-replicate-99.npz](../results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921/diagnostics/failed-coverage-rho-0.2-replicate-99.npz) | 3,528 | `b4aaa4d586344e222cc9d151e0eed081160801caf4444a4aca501679d50795ae` | [statistical-noncoverage](#statistical-noncoverage) | completed-run-statistical-noncoverage |
| [diagnostics/failed-coverage-rho-0.3-replicate-6.npz](../results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921/diagnostics/failed-coverage-rho-0.3-replicate-6.npz) | 3,528 | `8d4982527ff8e1bce006b0c64f885f6d0f9f974cb42b7755abf4e14c8491aff6` | [statistical-noncoverage](#statistical-noncoverage) | completed-run-statistical-noncoverage |
| [diagnostics/failed-coverage-rho-0.3-replicate-62.npz](../results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921/diagnostics/failed-coverage-rho-0.3-replicate-62.npz) | 3,528 | `419f47a8b2ea95d8f4dc5f6db3cdfa06ffe83825f67fbc174b850d8bdc77dcd5` | [statistical-noncoverage](#statistical-noncoverage) | completed-run-statistical-noncoverage |
| [diagnostics/failed-coverage-rho-0.3-replicate-63.npz](../results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921/diagnostics/failed-coverage-rho-0.3-replicate-63.npz) | 3,528 | `4d259bd6dd3b2381b6bf8deb23bb0c9a9acbe52cd9a812da85104c48950838d5` | [statistical-noncoverage](#statistical-noncoverage) | completed-run-statistical-noncoverage |
| [diagnostics/failed-coverage-rho-0.3-replicate-97.npz](../results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921/diagnostics/failed-coverage-rho-0.3-replicate-97.npz) | 3,528 | `aee2da2b0773af1e3e2ebd02c15485e658a75272fd0c790d980be543a8665705` | [statistical-noncoverage](#statistical-noncoverage) | completed-run-statistical-noncoverage |
| [diagnostics/failed-coverage-rho-0.4-replicate-29.npz](../results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921/diagnostics/failed-coverage-rho-0.4-replicate-29.npz) | 3,528 | `cab7280db7bb9df59fd034da15174934feba9eb49bbc78fb37b953ee54614cc5` | [statistical-noncoverage](#statistical-noncoverage) | completed-run-statistical-noncoverage |
| [diagnostics/failed-coverage-rho-0.4-replicate-31.npz](../results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921/diagnostics/failed-coverage-rho-0.4-replicate-31.npz) | 3,528 | `6ab1b9eacd9b12da5a2c3837758c8a47f21291f549e677ffdd19adcb5c3704f7` | [statistical-noncoverage](#statistical-noncoverage) | completed-run-statistical-noncoverage |
| [diagnostics/failed-coverage-rho-0.4-replicate-79.npz](../results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921/diagnostics/failed-coverage-rho-0.4-replicate-79.npz) | 3,528 | `60f575665b1497fe440918e55150857b2002496265bc5b5e72cd58905fc2cab1` | [statistical-noncoverage](#statistical-noncoverage) | completed-run-statistical-noncoverage |
| [diagnostics/failed-coverage-rho-0.5-replicate-1.npz](../results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921/diagnostics/failed-coverage-rho-0.5-replicate-1.npz) | 3,528 | `4cb566faa37760659030ca45ca408aa81f1fc11902f3afaaffe6648fa5ed5d88` | [statistical-noncoverage](#statistical-noncoverage) | completed-run-statistical-noncoverage |
| [diagnostics/failed-coverage-rho-0.5-replicate-17.npz](../results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921/diagnostics/failed-coverage-rho-0.5-replicate-17.npz) | 3,528 | `4b7af2d748aca56a62da3b4b4a18c397749bc029d692ccd9c1a8277cd7a96f91` | [statistical-noncoverage](#statistical-noncoverage) | completed-run-statistical-noncoverage |
| [diagnostics/failed-coverage-rho-0.5-replicate-25.npz](../results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921/diagnostics/failed-coverage-rho-0.5-replicate-25.npz) | 3,528 | `9d39ef03c8eaddefc51a57965dfb809a6898668eeb9f5f6fce91d239a919d10d` | [statistical-noncoverage](#statistical-noncoverage) | completed-run-statistical-noncoverage |
| [diagnostics/failed-coverage-rho-0.5-replicate-48.npz](../results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921/diagnostics/failed-coverage-rho-0.5-replicate-48.npz) | 3,528 | `3fc5d2e65211fe89be3c554aba4c27cf1e6b38d5e0327589660f61b9f97d1ec3` | [statistical-noncoverage](#statistical-noncoverage) | completed-run-statistical-noncoverage |
| [diagnostics/failed-coverage-rho-0.5-replicate-53.npz](../results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921/diagnostics/failed-coverage-rho-0.5-replicate-53.npz) | 3,528 | `575b4fb02c8a59e0aa18e5da1c9d0c5178e8a96567876899bc0c0bf5aea01f23` | [statistical-noncoverage](#statistical-noncoverage) | completed-run-statistical-noncoverage |
| [diagnostics/failed-coverage-rho-0.5-replicate-91.npz](../results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921/diagnostics/failed-coverage-rho-0.5-replicate-91.npz) | 3,528 | `acc28a21d2fbf6347772eafe9682da77b3457aea7edf38d871818a850e3d43fd` | [statistical-noncoverage](#statistical-noncoverage) | completed-run-statistical-noncoverage |
| [diagnostics/failed-coverage-rho-0.6-replicate-19.npz](../results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921/diagnostics/failed-coverage-rho-0.6-replicate-19.npz) | 3,528 | `cfe6cb02a7ab3f19f985d2231c4a0123c94f12e18fc5d321558df6336ff8c420` | [statistical-noncoverage](#statistical-noncoverage) | completed-run-statistical-noncoverage |
| [diagnostics/failed-coverage-rho-0.6-replicate-24.npz](../results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921/diagnostics/failed-coverage-rho-0.6-replicate-24.npz) | 3,528 | `b7bf9c1d9f8441b14ab53803ce594c099e23c12c6604c21752706a24ab96fadd` | [statistical-noncoverage](#statistical-noncoverage) | completed-run-statistical-noncoverage |
| [diagnostics/failed-coverage-rho-0.6-replicate-28.npz](../results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921/diagnostics/failed-coverage-rho-0.6-replicate-28.npz) | 3,528 | `89ba97f6c25922a5418d3863dc00bcd99a1460d452c5009a9536ede7e9a20b5c` | [statistical-noncoverage](#statistical-noncoverage) | completed-run-statistical-noncoverage |
| [diagnostics/failed-coverage-rho-0.6-replicate-32.npz](../results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921/diagnostics/failed-coverage-rho-0.6-replicate-32.npz) | 3,528 | `3149f3e4149f95a89a16c1bb7acf752919b89b478dab03625fe9e3c51d19d5b8` | [statistical-noncoverage](#statistical-noncoverage) | completed-run-statistical-noncoverage |
| [diagnostics/failed-coverage-rho-0.6-replicate-50.npz](../results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921/diagnostics/failed-coverage-rho-0.6-replicate-50.npz) | 3,528 | `a7e59740faaac63c66f0a7d641aa3ba951e7b12de1d377970f600f0e00fe1d6b` | [statistical-noncoverage](#statistical-noncoverage) | completed-run-statistical-noncoverage |
| [diagnostics/failed-coverage-rho-0.6-replicate-67.npz](../results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921/diagnostics/failed-coverage-rho-0.6-replicate-67.npz) | 3,528 | `682c2e55f12f132dbb36df6f79f8ce9310636bd306385d7c912bf412e131ef73` | [statistical-noncoverage](#statistical-noncoverage) | completed-run-statistical-noncoverage |
| [diagnostics/failed-coverage-rho-0.6-replicate-72.npz](../results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921/diagnostics/failed-coverage-rho-0.6-replicate-72.npz) | 3,528 | `f0fb7ccd296ed1596e1fe6770b3e652eb16bf6a68177d8277e2c2a1f27787cdb` | [statistical-noncoverage](#statistical-noncoverage) | completed-run-statistical-noncoverage |
| [diagnostics/failed-coverage-rho-0.6-replicate-86.npz](../results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921/diagnostics/failed-coverage-rho-0.6-replicate-86.npz) | 3,528 | `2610943e440078fd886aefdb54c7ccc83c66f9730ab5722da122a4751c06c4fd` | [statistical-noncoverage](#statistical-noncoverage) | completed-run-statistical-noncoverage |
| [diagnostics/failed-coverage-rho-0.7-replicate-26.npz](../results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921/diagnostics/failed-coverage-rho-0.7-replicate-26.npz) | 3,528 | `1e6b5575d2e77784138e112a5bb3b82153a7799552bdb64424feda7e31d30f63` | [statistical-noncoverage](#statistical-noncoverage) | completed-run-statistical-noncoverage |
| [diagnostics/failed-coverage-rho-0.7-replicate-35.npz](../results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921/diagnostics/failed-coverage-rho-0.7-replicate-35.npz) | 3,528 | `93c7326d376f56eb0208ab61b1c36a19d55c987ba918e1b7eeb69b2a11a23994` | [statistical-noncoverage](#statistical-noncoverage) | completed-run-statistical-noncoverage |
| [diagnostics/failed-coverage-rho-0.7-replicate-61.npz](../results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921/diagnostics/failed-coverage-rho-0.7-replicate-61.npz) | 3,528 | `bb7ee6ca426a2d97b0df254f1702e896c58e72a3b43d761d39176a5cffa00b91` | [statistical-noncoverage](#statistical-noncoverage) | completed-run-statistical-noncoverage |
| [diagnostics/failed-coverage-rho-0.7-replicate-65.npz](../results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921/diagnostics/failed-coverage-rho-0.7-replicate-65.npz) | 3,528 | `a0b14f468eb1cffe4b59c97c762c4a9941788b7334f822beabf228a968c5fb93` | [statistical-noncoverage](#statistical-noncoverage) | completed-run-statistical-noncoverage |
| [diagnostics/failed-coverage-rho-0.7-replicate-95.npz](../results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921/diagnostics/failed-coverage-rho-0.7-replicate-95.npz) | 3,528 | `4a9c7674e0d29cf8e8e2f35d7f823934cbc6254f89ae63f4195640b47d64e742` | [statistical-noncoverage](#statistical-noncoverage) | completed-run-statistical-noncoverage |
| [diagnostics/failed-coverage-rho-0.8-replicate-30.npz](../results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921/diagnostics/failed-coverage-rho-0.8-replicate-30.npz) | 3,528 | `bd398c22463fcdca6e1e107b64496a1b5a6b71a376fb285cb8d32eb77ba9ae9b` | [statistical-noncoverage](#statistical-noncoverage) | completed-run-statistical-noncoverage |
| [diagnostics/failed-coverage-rho-0.8-replicate-41.npz](../results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921/diagnostics/failed-coverage-rho-0.8-replicate-41.npz) | 3,528 | `e7ebd2ae883327c5835d31748a62bf4ed78f066ae97e1a2088ee7ac38dd37c8c` | [statistical-noncoverage](#statistical-noncoverage) | completed-run-statistical-noncoverage |
| [diagnostics/failed-coverage-rho-0.8-replicate-58.npz](../results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921/diagnostics/failed-coverage-rho-0.8-replicate-58.npz) | 3,528 | `e8e17f609e014f0c036934db1047225f2d0f5714acbcfea4bfe9aeae69327611` | [statistical-noncoverage](#statistical-noncoverage) | completed-run-statistical-noncoverage |
| [diagnostics/failed-coverage-rho-0.9-replicate-11.npz](../results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921/diagnostics/failed-coverage-rho-0.9-replicate-11.npz) | 3,528 | `420b96df82887f9be2b13e71a6c8cc3cc4b442432d4e9bd5ce3c2d3b20836e35` | [statistical-noncoverage](#statistical-noncoverage) | completed-run-statistical-noncoverage |
| [diagnostics/failed-coverage-rho-0.9-replicate-34.npz](../results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921/diagnostics/failed-coverage-rho-0.9-replicate-34.npz) | 3,528 | `cd979251f4c9ec79c2caa19b41962a3bb589e547076500c92c8155fcda59e78a` | [statistical-noncoverage](#statistical-noncoverage) | completed-run-statistical-noncoverage |
| [diagnostics/failed-coverage-rho-0.9-replicate-78.npz](../results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921/diagnostics/failed-coverage-rho-0.9-replicate-78.npz) | 3,528 | `8d3aede8ea5c8860c75affecf1fe60c4f2e2b88d3b109ea09c3c3fb1dccc16ce` | [statistical-noncoverage](#statistical-noncoverage) | completed-run-statistical-noncoverage |
| [diagnostics/failed-coverage-rho-0.9-replicate-94.npz](../results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921/diagnostics/failed-coverage-rho-0.9-replicate-94.npz) | 3,528 | `46befb6397d1173c1889f1d144b973fac834c50c9d308d8cd46e66c15cf1b151` | [statistical-noncoverage](#statistical-noncoverage) | completed-run-statistical-noncoverage |
| [diagnostics/failed-coverage-rho-0.9-replicate-95.npz](../results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921/diagnostics/failed-coverage-rho-0.9-replicate-95.npz) | 3,528 | `931bf31241bbb1f991e309995acc76de43c3df78b3fe6be9b3c7726f21d2f4be` | [statistical-noncoverage](#statistical-noncoverage) | completed-run-statistical-noncoverage |
| [diagnostics/failed-coverage-rho-0.9-replicate-97.npz](../results/runs/network-meta-analysis/simulation/nma-100-reproduction-20260921/diagnostics/failed-coverage-rho-0.9-replicate-97.npz) | 3,528 | `d108466e8e8723692571841c1103468b01aeabd16db5c2304ceabc85fee4da16` | [statistical-noncoverage](#statistical-noncoverage) | completed-run-statistical-noncoverage |

### Files in R4

Run directory: `Code/results/runs/network-meta-analysis/treatment-estimates/table-1-seed-2025-20260921`.

| File | Bytes | SHA-256 | Schema | Classification |
| --- | ---: | --- | --- | --- |
| [data/nma-simulation-treatment-inputs.npz](../results/runs/network-meta-analysis/treatment-estimates/table-1-seed-2025-20260921/data/nma-simulation-treatment-inputs.npz) | 120,400 | `e7f73c779319a0a4f3437cb361d7102f96c563a2de0dc58fff4cc2fa4d44bc83` | [table-inputs](#table-inputs) | new-run-data |

### Files in R5

Run directory: `Code/results/runs/one-dimensional/false-positive-rate/paired-comparison-20260921`.

| File | Bytes | SHA-256 | Schema | Classification |
| --- | ---: | --- | --- | --- |
| [data/false-positive-rate-paired.npz](../results/runs/one-dimensional/false-positive-rate/paired-comparison-20260921/data/false-positive-rate-paired.npz) | 136,090 | `08ccc2155b6c70f69d76a2ef1dcb18e3281dfaaee874de6ef8af2f06e8778ed7` | [paired-fpr](#paired-fpr) | new-run-data |

### Files in R6

Run directory: `Code/results/runs/one-dimensional/false-positive-rate/paired-visibility-replot-20260921`.

| File | Bytes | SHA-256 | Schema | Classification |
| --- | ---: | --- | --- | --- |
| [data/replot-cache.npz](../results/runs/one-dimensional/false-positive-rate/paired-visibility-replot-20260921/data/replot-cache.npz) | 2,758 | `17c7a5d30d6c775e8b1b958e4d9197c9cf44749722fb48bc4163b228da68ecdf` | [cache-fpr](#cache-fpr) | new-run-data |

### Files in R7

Run directory: `Code/results/runs/one-dimensional/interval-width/interval-width-layout-replot-20260921`.

| File | Bytes | SHA-256 | Schema | Classification |
| --- | ---: | --- | --- | --- |
| [data/replot-cache.npz](../results/runs/one-dimensional/interval-width/interval-width-layout-replot-20260921/data/replot-cache.npz) | 606,841 | `53ca0454e363af9be199fbe3775c06093a61539f84ab9317d59ede3a286da4cc` | [cache-width](#cache-width) | new-run-data |

### Files in R8

Run directory: `Code/results/runs/one-dimensional/power/paired-comparison-20260921`.

| File | Bytes | SHA-256 | Schema | Classification |
| --- | ---: | --- | --- | --- |
| [data/power-comparison-paired.npz](../results/runs/one-dimensional/power/paired-comparison-20260921/data/power-comparison-paired.npz) | 245,331 | `604d965f2a7f9c6dbadbd96acc41dc34a4b92e443d93930c685420310c9ab519` | [paired-power](#paired-power) | new-run-data |

### Files in R9

Run directory: `Code/results/runs/one-dimensional/power/paired-visibility-replot-20260921`.

| File | Bytes | SHA-256 | Schema | Classification |
| --- | ---: | --- | --- | --- |
| [data/replot-cache.npz](../results/runs/one-dimensional/power/paired-visibility-replot-20260921/data/replot-cache.npz) | 2,932 | `7cb5e739576cd8d614c1caa84da4eeb56e1caeaabad2c08e0e417f4c00c64151` | [cache-power](#cache-power) | new-run-data |

## Archive schemas

Shapes and dtypes below come from each retained NPY member header. Empty shape `()` means a scalar. All archives have non-object dtypes and are safe to load with pickle disabled. Means/widths are in effect units, variances in squared effect units, rates are proportions, and thresholds/multipliers have the meanings stated below.

### statistical-noncoverage

Input snapshot of a completed HCCT noncoverage outcome, not a numerical solver failure.

| Member | Shape | dtype | Meaning |
| --- | --- | --- | --- |
| `Sigma` | `(28,)` | `float64` | Estimated variances of the study means, not a covariance matrix. |
| `point` | `(9,)` | `float64` | True nine-coordinate treatment vector; not an optimizer estimate. |
| `projs` | `(28, 9)` | `float64` | 28-by-9 study comparison design matrix. |
| `xi_hat` | `(28,)` | `float64` | Estimated study means, in treatment-effect units. |

### failed-partial

Original failed 500-per-rho attempt: partial outcomes and numerical-failure evidence; not complete estimates.

| Member | Shape | dtype | Meaning |
| --- | --- | --- | --- |
| `covered_hcct` | `(10, 500)` | `float64` | Numeric 0/1 HCCT joint-region coverage at truth; empty sets are zero. |
| `covered_wls` | `(10, 500)` | `float64` | Numeric 0/1 simultaneous WLS coordinate-box coverage at truth. |
| `covered_wls_ma` | `(10, 500)` | `float64` | Numeric 0/1 in-sample empirical-oracle WLS box coverage. |
| `design` | `(28, 9)` | `float64` | 28-by-9 study comparison design matrix. |
| `empty_hcct` | `(10, 500)` | `float64` | Numeric 0/1 certified-empty flag; never a proxy for numerical failure. |
| `estimated_mean_variances` | `(10, 500, 28)` | `float64` | Estimated mean variances W/[n(n-1)], in squared effect units. |
| `hcct_boundary_errors` | `(10, 500, 2, 2)` | `float64` | Score-unit endpoint boundary errors; axes rho, replicate, direction, lower/upper. |
| `hcct_kkt_residuals` | `(10, 500, 2, 2)` | `float64` | Normalized endpoint stationarity residuals; same axes as boundary errors. |
| `hcct_minimum_points` | `(10, 500, 2, 9)` | `float64` | Minimum parameter vectors in original treatment-effect units. |
| `hcct_minimum_scores` | `(10, 500, 2)` | `float64` | Combined-score minimum for each direction solve. |
| `hcct_support_points` | `(10, 500, 2, 2, 9)` | `float64` | Endpoint parameter vectors; axes rho, replicate, direction, lower/upper, coordinate. |
| `rhos` | `(10,)` | `float64` | Correlation grid in the stored order. |
| `true_theta` | `(9,)` | `float64` | True nine-coordinate treatment vector against placebo. |
| `widths_hcct` | `(10, 500, 2)` | `float64` | HCCT full widths, direction theta1 then theta2; certified-empty width is zero. |
| `wls_estimates` | `(10, 500, 9)` | `float64` | Inverse-estimated-variance WLS treatment estimates. |
| `wls_maximum_standardized_errors` | `(10, 500)` | `float64` | Maximum absolute WLS standardized error across nine coordinates. |
| `wls_standard_errors` | `(10, 500, 9)` | `float64` | WLS standard errors, in treatment-effect units. |
| `xi_hat` | `(10, 500, 28)` | `float64` | Estimated study means, in treatment-effect units. |

### nma-plotted-summary

Actual Figure 10 plotted arrays: two historical manual WLS-MA width series, otherwise unchanged raw summaries.

| Member | Shape | dtype | Meaning |
| --- | --- | --- | --- |
| `coverage_hcct` | `(10,)` | `float64` | Unconditional HCCT joint-region coverage proportion, by rho. |
| `coverage_wls` | `(10,)` | `float64` | Simultaneous nine-coordinate WLS box coverage proportion, by rho. |
| `coverage_wls_ma` | `(10,)` | `float64` | Achieved in-sample empirical-oracle coverage; not validation of historical manual widths. |
| `empirical_wls_ma_multiplier` | `(10,)` | `float64` | Raw empirical multiplier, explicitly not the multiplier of plotted historical widths. |
| `empty_hcct_count` | `(10,)` | `int64` | Certified-empty outcome count, by rho. |
| `replicates_per_rho` | `(10,)` | `int64` | Number of replications in each completed rho cell. |
| `rhos` | `(10,)` | `float64` | Correlation grid in the stored order. |
| `width_hcct_1` | `(10,)` | `float64` | Unconditional mean HCCT full width in theta1. |
| `width_hcct_2` | `(10,)` | `float64` | Unconditional mean HCCT full width in theta2. |
| `width_wls_1` | `(10,)` | `float64` | Mean nominal WLS full width in theta1. |
| `width_wls_2` | `(10,)` | `float64` | Mean nominal WLS full width in theta2. |
| `width_wls_ma_1` | `(10,)` | `float64` | Empirical-oracle full width in raw/cache archives; historical manual theta1 width ONLY in plotted-summary. |
| `width_wls_ma_2` | `(10,)` | `float64` | Empirical-oracle full width in raw/cache archives; historical manual theta2 width ONLY in plotted-summary. |

### cache-nma

Raw empirical NMA plot-input cache, not the manual-width plotted summary and not another simulation.

| Member | Shape | dtype | Meaning |
| --- | --- | --- | --- |
| `axis__rhos` | `(10,)` | `float64` | Explicit coordinate labels/indices for the rhos axis. |
| `coverage_hcct` | `(10,)` | `float64` | Unconditional HCCT joint-region coverage proportion, by rho. |
| `coverage_wls` | `(10,)` | `float64` | Simultaneous nine-coordinate WLS box coverage proportion, by rho. |
| `coverage_wls_ma` | `(10,)` | `float64` | Achieved in-sample empirical-oracle coverage; not validation of historical manual widths. |
| `empty_hcct_count` | `(10,)` | `int64` | Certified-empty outcome count, by rho. |
| `experiment_id` | `()` | `<U32` | Original simulation experiment identifier. |
| `profile` | `()` | `<U5` | Source profile; paper NMA is the authorized 100-rerun, not original 500 effort. |
| `replicates_per_rho` | `(10,)` | `int64` | Number of replications in each completed rho cell. |
| `schema_version` | `()` | `int64` | Normalized plot-cache schema version. |
| `seed` | `()` | `int64` | Recorded original simulation seed; replotting does not sample. |
| `width_hcct_1` | `(10,)` | `float64` | Unconditional mean HCCT full width in theta1. |
| `width_hcct_2` | `(10,)` | `float64` | Unconditional mean HCCT full width in theta2. |
| `width_wls_1` | `(10,)` | `float64` | Mean nominal WLS full width in theta1. |
| `width_wls_2` | `(10,)` | `float64` | Mean nominal WLS full width in theta2. |
| `width_wls_ma_1` | `(10,)` | `float64` | Empirical-oracle full width in raw/cache archives; historical manual theta1 width ONLY in plotted-summary. |
| `width_wls_ma_2` | `(10,)` | `float64` | Empirical-oracle full width in raw/cache archives; historical manual theta2 width ONLY in plotted-summary. |
| `wls_ma_multiplier` | `(10,)` | `float64` | In-sample empirical 95th percentile of maximum standardized coordinate error. |

### nma-replicates

All 1,000 completed NMA outcomes: inputs, estimates, endpoints and saved solver diagnostics.

| Member | Shape | dtype | Meaning |
| --- | --- | --- | --- |
| `covered_hcct` | `(10, 100)` | `float64` | Numeric 0/1 HCCT joint-region coverage at truth; empty sets are zero. |
| `covered_wls` | `(10, 100)` | `float64` | Numeric 0/1 simultaneous WLS coordinate-box coverage at truth. |
| `covered_wls_ma` | `(10, 100)` | `float64` | Numeric 0/1 in-sample empirical-oracle WLS box coverage. |
| `design` | `(28, 9)` | `float64` | 28-by-9 study comparison design matrix. |
| `empty_hcct` | `(10, 100)` | `float64` | Numeric 0/1 certified-empty flag; never a proxy for numerical failure. |
| `estimated_mean_variances` | `(10, 100, 28)` | `float64` | Estimated mean variances W/[n(n-1)], in squared effect units. |
| `hcct_boundary_errors` | `(10, 100, 2, 2)` | `float64` | Score-unit endpoint boundary errors; axes rho, replicate, direction, lower/upper. |
| `hcct_kkt_residuals` | `(10, 100, 2, 2)` | `float64` | Normalized endpoint stationarity residuals; same axes as boundary errors. |
| `hcct_minimum_points` | `(10, 100, 2, 9)` | `float64` | Minimum parameter vectors in original treatment-effect units. |
| `hcct_minimum_scores` | `(10, 100, 2)` | `float64` | Combined-score minimum for each direction solve. |
| `hcct_support_points` | `(10, 100, 2, 2, 9)` | `float64` | Endpoint parameter vectors; axes rho, replicate, direction, lower/upper, coordinate. |
| `rhos` | `(10,)` | `float64` | Correlation grid in the stored order. |
| `true_theta` | `(9,)` | `float64` | True nine-coordinate treatment vector against placebo. |
| `widths_hcct` | `(10, 100, 2)` | `float64` | HCCT full widths, direction theta1 then theta2; certified-empty width is zero. |
| `wls_estimates` | `(10, 100, 9)` | `float64` | Inverse-estimated-variance WLS treatment estimates. |
| `wls_maximum_standardized_errors` | `(10, 100)` | `float64` | Maximum absolute WLS standardized error across nine coordinates. |
| `wls_standard_errors` | `(10, 100, 9)` | `float64` | WLS standard errors, in treatment-effect units. |
| `xi_hat` | `(10, 100, 28)` | `float64` | Estimated study means, in treatment-effect units. |

### nma-summary

Raw 100-per-rho coverage/width summaries, with in-sample empirical-oracle WLS-MA widths.

| Member | Shape | dtype | Meaning |
| --- | --- | --- | --- |
| `coverage_hcct` | `(10,)` | `float64` | Unconditional HCCT joint-region coverage proportion, by rho. |
| `coverage_wls` | `(10,)` | `float64` | Simultaneous nine-coordinate WLS box coverage proportion, by rho. |
| `coverage_wls_ma` | `(10,)` | `float64` | Achieved in-sample empirical-oracle coverage; not validation of historical manual widths. |
| `empty_hcct_count` | `(10,)` | `int64` | Certified-empty outcome count, by rho. |
| `replicates_per_rho` | `(10,)` | `int64` | Number of replications in each completed rho cell. |
| `rhos` | `(10,)` | `float64` | Correlation grid in the stored order. |
| `width_hcct_1` | `(10,)` | `float64` | Unconditional mean HCCT full width in theta1. |
| `width_hcct_2` | `(10,)` | `float64` | Unconditional mean HCCT full width in theta2. |
| `width_wls_1` | `(10,)` | `float64` | Mean nominal WLS full width in theta1. |
| `width_wls_2` | `(10,)` | `float64` | Mean nominal WLS full width in theta2. |
| `width_wls_ma_1` | `(10,)` | `float64` | Empirical-oracle full width in raw/cache archives; historical manual theta1 width ONLY in plotted-summary. |
| `width_wls_ma_2` | `(10,)` | `float64` | Empirical-oracle full width in raw/cache archives; historical manual theta2 width ONLY in plotted-summary. |
| `wls_ma_multiplier` | `(10,)` | `float64` | In-sample empirical 95th percentile of maximum standardized coordinate error. |

### table-inputs

Fresh seeded samples for Main Table 1 support, one sample per rho; not recovery of its unavailable original realization.

| Member | Shape | dtype | Meaning |
| --- | --- | --- | --- |
| `design` | `(28, 9)` | `float64` | 28-by-9 study comparison design matrix. |
| `estimated_mean_variances` | `(4, 28)` | `float64` | Estimated mean variances W/[n(n-1)], in squared effect units. |
| `means` | `(4, 28)` | `float64` | Study sample means for the table: rho, study. |
| `rhos` | `(4,)` | `float64` | Correlation grid in the stored order. |
| `samples` | `(4, 100, 28)` | `float64` | Raw normal observations for the table: rho, observation, study. |
| `true_mean_covariances` | `(4, 28, 28)` | `float64` | Known covariance of study means by rho, separate from estimated diagonal variances. |
| `true_theta` | `(9,)` | `float64` | True nine-coordinate treatment vector against placebo. |

### paired-fpr

Per-draw false-positive decisions for all nine methods on shared samples; supports paired uncertainty.

| Member | Shape | dtype | Meaning |
| --- | --- | --- | --- |
| `axes` | `(4,)` | `<U6` | Names of axes of the rejections tensor. |
| `cases` | `(2,)` | `<U14` | AR1/equicorrelation for FPR or dense/sparse alternatives for power. |
| `level` | `()` | `float64` | Nominal test level. |
| `methods` | `(9,)` | `<U10` | Saved method order; all nine methods are retained, including unplotted HMP. |
| `rejections` | `(9, 10, 2, 10000)` | `bool` | Boolean test decisions; axes method, rho, case, draw; True is rejection. |
| `rhos` | `(10,)` | `float64` | Correlation grid in the stored order. |
| `rng` | `()` | `<U42` | Stored RNG algorithm and draw-order description. |
| `sampling_scheme` | `()` | `<U55` | Sharing within a rho/case and distinct batches between cases. |
| `seed` | `()` | `int64` | Recorded original simulation seed; replotting does not sample. |
| `simulations_per_case` | `()` | `int64` | Number of shared outcomes in each rho/case. |
| `study_count` | `()` | `int64` | Number of studies in each outcome. |
| `thresholds` | `(9,)` | `float64` | Rejection cutoffs aligned with the methods array. |

### cache-fpr

Unchanged paired FPR rates used in the visibility replot; does not replace per-draw indicators.

| Member | Shape | dtype | Meaning |
| --- | --- | --- | --- |
| `axis__methods` | `(9,)` | `<U10` | Explicit coordinate labels/indices for the methods axis. |
| `axis__rhos` | `(10,)` | `float64` | Explicit coordinate labels/indices for the rhos axis. |
| `axis__series` | `(2,)` | `<U14` | Explicit coordinate labels/indices for the series axis. |
| `experiment_id` | `()` | `<U35` | Original simulation experiment identifier. |
| `profile` | `()` | `<U5` | Source profile; paper NMA is the authorized 100-rerun, not original 500 effort. |
| `rates` | `(9, 10, 2)` | `float64` | Saved false-positive proportions; axes method, rho, dependence. |
| `schema_version` | `()` | `int64` | Normalized plot-cache schema version. |
| `seed` | `()` | `int64` | Recorded original simulation seed; replotting does not sample. |

### cache-width

All 80,000 original scalar widths for the grid-style replot; no refitting or resampling.

| Member | Shape | dtype | Meaning |
| --- | --- | --- | --- |
| `axis__dependence` | `(2,)` | `<U14` | Explicit coordinate labels/indices for the dependence axis. |
| `axis__replicates` | `(10000,)` | `int64` | Explicit coordinate labels/indices for the replicates axis. |
| `axis__rhos` | `(4,)` | `float64` | Explicit coordinate labels/indices for the rhos axis. |
| `experiment_id` | `()` | `<U30` | Original simulation experiment identifier. |
| `profile` | `()` | `<U5` | Source profile; paper NMA is the authorized 100-rerun, not original 500 effort. |
| `schema_version` | `()` | `int64` | Normalized plot-cache schema version. |
| `seed` | `()` | `int64` | Recorded original simulation seed; replotting does not sample. |
| `widths` | `(2, 4, 10000)` | `float64` | Saved scalar full interval widths; axes dependence, rho, replicate. |

### paired-power

Per-draw power decisions for all nine methods on shared samples plus the alternative mean vectors.

| Member | Shape | dtype | Meaning |
| --- | --- | --- | --- |
| `alternative_means` | `(2, 500)` | `float64` | Dense/sparse power alternative mean vectors; case, study. |
| `axes` | `(4,)` | `<U6` | Names of axes of the rejections tensor. |
| `cases` | `(2,)` | `<U6` | AR1/equicorrelation for FPR or dense/sparse alternatives for power. |
| `level` | `()` | `float64` | Nominal test level. |
| `methods` | `(9,)` | `<U10` | Saved method order; all nine methods are retained, including unplotted HMP. |
| `rejections` | `(9, 10, 2, 10000)` | `bool` | Boolean test decisions; axes method, rho, case, draw; True is rejection. |
| `rhos` | `(10,)` | `float64` | Correlation grid in the stored order. |
| `rng` | `()` | `<U42` | Stored RNG algorithm and draw-order description. |
| `sampling_scheme` | `()` | `<U55` | Sharing within a rho/case and distinct batches between cases. |
| `seed` | `()` | `int64` | Recorded original simulation seed; replotting does not sample. |
| `simulations_per_case` | `()` | `int64` | Number of shared outcomes in each rho/case. |
| `study_count` | `()` | `int64` | Number of studies in each outcome. |
| `thresholds` | `(9,)` | `float64` | Rejection cutoffs aligned with the methods array. |

### cache-power

Unchanged paired power rates used in the visibility replot; does not replace per-draw indicators.

| Member | Shape | dtype | Meaning |
| --- | --- | --- | --- |
| `axis__methods` | `(9,)` | `<U10` | Explicit coordinate labels/indices for the methods axis. |
| `axis__rhos` | `(10,)` | `float64` | Explicit coordinate labels/indices for the rhos axis. |
| `axis__series` | `(2,)` | `<U6` | Explicit coordinate labels/indices for the series axis. |
| `experiment_id` | `()` | `<U21` | Original simulation experiment identifier. |
| `power` | `(9, 10, 2)` | `float64` | Saved power proportions; axes method, rho, alternative. |
| `profile` | `()` | `<U5` | Source profile; paper NMA is the authorized 100-rerun, not original 500 effort. |
| `schema_version` | `()` | `int64` | Normalized plot-cache schema version. |
| `seed` | `()` | `int64` | Recorded original simulation seed; replotting does not sample. |

## Refresh or verify this inventory

The [read-only builder](../../Note/build_npz_catalog_2026-09-21.py) scans `Code/` without respecting ignore rules, rejects unclassified/object-array archives and only prints its results. It never writes, deletes, resimulates, or opens protected archives. From the project root:

```bash
python Note/build_npz_catalog_2026-09-21.py --check
python Note/build_npz_catalog_2026-09-21.py --format json
python Note/build_npz_catalog_2026-09-21.py --format markdown
```

`--emit-patch` prints a proposed documentation/map patch to stdout for review; it does not apply it. Catalog checks are inventory/provenance checks, not a new mathematical or simulation audit.
