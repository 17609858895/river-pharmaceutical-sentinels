# Release validation

Checks performed on 7 November 2026. These establish limited code/package integrity; they do not establish that every analysis was freshly rerun in a new environment.

- All released Python sources pass syntax/compilation checks.
- Seventeen analysis/collection/figure entry points returned successfully from `--help` when launched from a working directory outside the repository. Two additional entry points, `coverage_summary.py` and `fig02_schematic.py`, also passed that check. Help exits before data access or fitting.
- The released parser and feature builder were executed on the original workbook in an isolated local validation directory. All eleven resulting CSVs were byte-identical to the corresponding current research inputs. No source dataset is included in this repository.
- A synthetic six-record, two-round AFT smoke check returned finite positive predictions. Observed endpoint calculations and the all-measured mixture plug-in were checked against direct calculations. This small artificial fit is not a retraining of the research models.
- The released S17 builder reconstructed all twenty saved detection-rank/alternating panel-fold comparisons, validated each panel identity against its training-side rule, and matched the current research comparison table to a maximum absolute numerical difference of 1.11e-16. No Direct-LGBM model was fitted for this check.
- The released self-contained plotting helpers rendered the current three-panel Fig. 3 from the existing research data/results read-only into an isolated validation directory. The combined PNG was pixel-identical to the current final research figure (7464 × 5895 pixels); its three standalone builders also exported successfully. This checks that removing private plotting-helper dependencies did not alter that figure. The remaining nine figures were not regenerated for code-release validation.
- The final code tree was scanned for absolute Windows drive paths, personal account identifiers, author contact information, credential/token patterns, excluded document/data/model artifact types and unbundled private helper dependencies.

`validation_environment.json` identifies the Python/scientific-package environment used for these release checks only. The original research environment was not locked. Full nested selection, benchmark fitting, calibration, threshold-transfer, screening and attribution analyses were not rerun during release preparation. Exact original numeric results across different library versions are not guaranteed by these limited checks.
