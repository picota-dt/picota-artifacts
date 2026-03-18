# Energigran

This case reproduces the experiments in `energigran.py`. The script trains KAN and TabNet variants on an hourly table to predict `generation` with a 24-hour horizon, then reports metrics, rule violations, and optional SHAP importances.

## Requirements

- Python 3.12 or compatible.
- PyTorch, NumPy, SciPy, and SHAP.
- Run the commands from `study_cases/energigran/code/`.
- Provide your own TSV or CSV input with the columns expected by the script.

## Bootstrap

From `study_cases/energigran/code/`:

```bash
bash bootstrap.sh
```

Useful options:

- `--skip-ml-deps` skips NumPy/SciPy/SHAP/Torch installation.
- `--cpu-torch` installs the CPU build of PyTorch.

`bootstrap.sh` creates a `.venv` next to the code and installs the local Python dependencies. With full installation enabled, it also verifies that the local case modules import correctly.

## Dataset

The script accepts `--csv` and also checks `PICOTA_ENERGIGRAN_DATA` if it is set. If neither is provided, it looks for a local dataset under `study_cases/energigran/data/`, but this repository does not ship one.

Use your own input explicitly:

```bash
export PICOTA_ENERGIGRAN_DATA=/path/to/energigran.tsv
```

## Direct Execution

```bash
python3 energigran.py --help
python3 energigran.py --csv /path/to/energigran.tsv --trainer-mode all
```

Relevant flags:

- `--trainer-mode` selects `KAN`, `KAN-Mm`, `tabnet`, `tabnet-mr`, or `all`.
- `--model-out` controls the model output path.
- `--shap-output` controls the CSV with global SHAP importances.
- `--disable-shap` disables SHAP.
- `--limit-hours` caps the number of hourly rows for quick runs.

Default outputs:

- model: `temp/test-models-alternative/SolarPlant/generation_h24_from_tsv.bin`
- SHAP: `temp/analysis/solarPlant/generation_h24_shap_importance.csv`

## Limitations

- Without an input file, the script fails with `Input table not found`.
- This repository does not include a bundled Energigran dataset or trained artifacts.
- Real training still requires a TSV/CSV large enough to pass the script’s minimum-row checks.
