# River pharmaceutical sentinels

Analysis and final-figure code for *Sentinel panels balance concentration and risk screening in river pharmaceutical mixtures*.

The main comparison concerns complementary sentinel-panel selection for summed concentration and benchmark-exceedance screening. Direct site-level boosting and panel-specific censored distributional inference are separate prediction routes. The latter uses a pooled AFT location model, per-API scales and a censored one-factor Gaussian copula; it combines inferred unmeasured eligible constituents with measured panel values. Concentrations are summed in ng/L; this is not flow-weighted mass loading. The primary benchmarks combine the source workbook's lowest PNEC/CEC/AMR values, so the endpoint is a screening quantity.

## What is included

- `scripts/01_parse_data.py`, `02_build_features.py`: original workbook parsing, alignment and feature construction.
- `scripts/sentry_lib.py`: current panel-specific censored predictor, observed endpoints, panel encoding and direct-model features. Unused older amortised-mask and threshold-classifier experiments are excluded.
- `04_benchmark20.py`: covariate benchmarks, random-row reference and optional HydroWASTE enrichment.
- `10_sentinel_v3.py`: nested campaign-grouped selection, direct boosting and structured/component comparisons.
- `12_extra_baselines.py`: alternating ranking rule and additional panel baselines.
- `13_calibrated_direct.py`, `14_threshold_transfer.py`, `17_conformal_threeway.py`: probability calibration, changed benchmarks and separated selection/calibration/test evaluation.
- `05_causal_wastewater.py`, `07_partial_panels.py`, `08_interpret.py`, `15_hydrowaste_features.py`, `16_hw_enrichment.py`: adjusted field-note associations, partial panels, descriptive model attributions and infrastructure enrichment. The historical `causal` filename is retained; the estimated field-note associations do not establish causality.
- `18_indeterminate_nd.py`: optional supplementary below-detection-limit sensitivity, conditional on a fitted lognormal tail.
- `collect_results.py`: collection of per-fold result CSVs; `build_s17.py`: reconstruction of the verified measured-only comparison, without model fitting; `coverage_summary.py`: observed endpoint coverage of model-eligible constituents.
- `figures_publication.py`, its panel builders and small self-contained `figlib/`: the final ten-figure layout, including 26 independent quantitative panels. `assets/study_design.png` is the retained single schematic source.

No manuscript drafts, Word/Zotero code, bibliographic library, source dataset, site-level processed dataset, fitted model or credential is included. Original exploratory amortised runs, unknown-identity reference-panel ablations and old geographic-transfer comparisons are excluded.

## Environment

Use Python 3.10 or later (the release smoke checks used Python 3.12). Install the scientific dependencies with:

```sh
python -m pip install -r requirements.txt
```

The original analysis environment was not locked. `requirements.txt` is intentionally unpinned and is **not** an original-environment lockfile. `validation_environment.json` records only the environment of the limited release checks; it must not be interpreted as proof that those exact package versions produced the original saved research results. Library changes, floating-point behaviour and fonts can change fitted results or rendering. Arial is used when available, with DejaVu Sans as the portable fallback.

`SENTRY_NTHREAD` controls the core thread count (default 12). `SENTRY_EXTRA=hw` enables HydroWASTE features. Unset the latter for base-feature runs. Analysis scripts resolve input/output paths against this repository, so commands may be launched from another working directory using the script's full or relative path. Run the examples below from the repository root for convenience. Numeric seed/fold arguments are positional; `--help` prints usage before data loading or fitting.

## Obtain the source data

1. Obtain `Supplemental Tables_PNAS_2022.xlsx` from the [Global Pharmaceuticals findings page](https://www.globalpharms.org/the-findings), accompanying [Wilkinson et al. (2022)](https://doi.org/10.1073/pnas.2113947119). Save it in the repository root as **`Supplemental_Tables_PNAS_2022.xlsx`**. The workbook used by this project had SHA-256 `30d637370542003b09f92053646de541dd033d810a9e1a3751d731051cf73e5b` and 1,291,036 bytes. Its own copyright/redistribution conditions apply; the workbook and individual-level derivatives are not redistributed here.
2. For infrastructure analyses, obtain HydroWASTE v1.0 from [Figshare](https://doi.org/10.6084/m9.figshare.14847786), extract `HydroWASTE_v10.csv` into `external_data/hydrowaste/`. The downloaded archive used here had MD5 `90cfce4e9ccb83775235374aa2df42a3` and SHA-256 `6e19d8f31d95d533b2b29056b823eeb915d933fa22ea090f234dc38f6427d044`. Cite [Ehalt Macedo et al. (2022)](https://doi.org/10.5194/essd-14-559-2022); source licence CC BY 4.0.
3. For the Fig. 1 basemap, download [Natural Earth 1:110m countries](https://www.naturalearthdata.com/downloads/110m-cultural-vectors/110m-admin-0-countries/) and extract the shapefile and companion files into `external_data/naturalearth/`. The original archive SHA-256 was `0f243aeac8ac6cf26f0417285b0bd33ac47f1b5bdb719fd3e0df37d03ea37110`. Natural Earth data are public domain. The numerical plot can render without the optional basemap.

Then preprocess:

```sh
python scripts/01_parse_data.py
python scripts/02_build_features.py
python scripts/15_hydrowaste_features.py
```

The first two commands create the site, compound, concentration/status and covariate CSVs in `data_processed/`. The resulting analytic frame has 1,049 aligned sites and 137 campaigns. The source survey assayed 61 constituents; the implemented marginal-model eligibility rule yields 46 APIs with at least five detections. Training-fold candidate-panel eligibility additionally requires at least 5% detection. Observed endpoints include the assay-wide observed concentrations, whereas unmeasured marginal predictions cover eligible APIs only. These coverage conventions are part of the analysis, not an assertion that all 61 APIs are predicted separately.

## Reproduce analyses

These commands describe the research workflow. The release preparation did **not** rerun all training, cross-validation, selection or interpretation analyses. Full model runs, especially nested greedy selection and broad benchmarks, can take substantial time.

For each seed `0, 1` and fold `0, 1, 2, 3, 4`, run:

```sh
python scripts/10_sentinel_v3.py SEED FOLD
python scripts/12_extra_baselines.py SEED FOLD
python scripts/14_threshold_transfer.py SEED FOLD
python scripts/17_conformal_threeway.py SEED FOLD
```

Replace `SEED` and `FOLD` by their integers. Each script writes a separate seed/fold CSV; omit `FOLD` to run all five folds in one process. Do not mix partial and complete duplicate runs in a results directory: the collector prefers per-fold files when present. The `SENTRY_QUICK` setting in script 10 is a reduced exploratory run, not a reproduction of the final analyses.

After scripts 10 and 12 finish, collect their outputs before calibrating direct predictions:

```sh
python scripts/collect_results.py --only sentinel
python scripts/13_calibrated_direct.py
```

The published covariate comparisons used two campaign-grouped repetitions, but the random-row reference used one five-fold repetition:

```sh
python scripts/04_benchmark20.py --tag cov --repeats 2
python scripts/04_benchmark20.py --tag cov_randomsplit --repeats 1 --split random
```

For the enriched benchmark, set `SENTRY_EXTRA=hw` and run script 04 with `--tag cov_hw --repeats 2`. Restore/unset `SENTRY_EXTRA` afterwards. This is shell-specific: PowerShell uses `$env:SENTRY_EXTRA='hw'` and `Remove-Item Env:SENTRY_EXTRA`; a POSIX shell can prefix a command with `SENTRY_EXTRA=hw`.

Additional results:

```sh
python scripts/07_partial_panels.py 6
python scripts/05_causal_wastewater.py
python scripts/08_interpret.py
python scripts/18_indeterminate_nd.py
python scripts/collect_results.py --only transfer partial conformal
python scripts/build_s17.py
python scripts/coverage_summary.py
```

`16_hw_enrichment.py SEED FOLD` is run for both the base and `SENTRY_EXTRA=hw` feature sets for the supplementary infrastructure comparison. The field-note script uses campaign-grouped cross-fitting, an approximate cluster-based interval from median estimates/SEs across three seeds, and 20 cumulative-concentration permutations. The three-way screening script calibrates site scores within held-out campaign sets; its outputs are empirical operating results, not a guaranteed bound for every individual campaign.

`build_s17.py` validates the saved panel identities against the reconstructed training-side detection/alternating rules and produces `tables/Table_S17.csv` and `Table_S17_summary.csv`. It does not refit Direct-LGBM. The per-fold direct scores must come from scripts 10 and 12.

## Reproduce figures

After preprocessing and all required analysis/collection commands:

```sh
python scripts/figures_publication.py
# Or a selected figure:
python scripts/figures_publication.py --only 3
```

The entry point currently loads all shared data/result tables even for a selected figure. Required result tables include `benchmark20/bench20_{cov,cov_randomsplit,cov_hw}_summary.csv`, `sentinel_v3/v3_plus_extra_all.csv`, `sentinel_v3/calibration_check.csv`, `threshold_transfer/transfer_all.csv`, `partial_panels/partial_panels_all.csv`, `conformal/conformal_all.csv`, the descriptive `interpret/*.csv` and `causal/dml_wastewater_exposure.csv`. No legacy `ablations/` file is required by the released final builders.

Outputs are PNG/PDF in `figures/Fig01_*` through `Fig10_*`, with independently rendered PNG/PDF panels in each `subfigures/` folder. Quantitative panels are generated from numeric data, not the schematic asset. Figure exports are generated locally and are not bundled in this code release; rendering requires the analysis tables described above.

See `RELEASE_VALIDATION.md` for the precise limited checks performed and their limits. No claim of a completed fresh end-to-end training run is made.
