# Europlátano

This case reproduces the experiments in `europlatano.py`. The script trains KAN and TabNet variants on an agricultural production table to predict `Production` with a 28-day horizon, then reports metrics, rule violations, and optional SHAP importances.

## Requirements

- Python 3.12 or compatible.
- PyTorch, NumPy, SciPy, and SHAP.
- Run the commands from `study_cases/europlatano/code/`.
- Provide a TSV input with the schema expected by the script.

## Bootstrap

From `study_cases/europlatano/code/`:

```bash
bash bootstrap.sh
```

Useful options:

- `--skip-ml-deps` skips NumPy/SciPy/SHAP/Torch installation.
- `--cpu-torch` installs the CPU build of PyTorch.

`bootstrap.sh` creates a `.venv` next to the code and installs the local Python dependencies. It also verifies that the local case modules import correctly.

## Dataset

The script accepts `--tsv` and also checks `PICOTA_EUROPLATANO_DATA` if it is set. If neither is provided, it looks for a local dataset under `study_cases/europlatano/data/`.

The repository includes `study_cases/europlatano/data_example/europlatano.tsv`, but it is too small for training and currently fails the script’s minimum parsed-row check.

Use your own input explicitly:

```bash
export PICOTA_EUROPLATANO_DATA=/path/to/europlatano.tsv
```

## Direct Execution

```bash
python3 europlatano.py --help
python3 europlatano.py --tsv /path/to/europlatano.tsv --trainer-mode all
```

Relevant flags:

- `--trainer-mode` selects `KAN`, `KAN-Mm`, `tabnet`, `tabnet-mm`, or `all`.
- `--model-out` controls the model output path.
- `--shap-output` controls the CSV with global SHAP importances.
- `--disable-shap` disables SHAP.
- `--limit-rows` caps the number of raw rows for quick runs.

Default outputs:

- model: `temp/test-models-alternative/Europlatano/Production_h28_from_tsv.bin`
- SHAP: `temp/analysis/europlatano/Production_h28_shap_importance.csv`

## Limitations

- The bundled `data_example/europlatano.tsv` is not large enough to train and fails with `Need >=10 parsed rows`.
- This repository does not include the confidential Europlátano training dataset or trained artifacts.
- Real training still requires a TSV large enough to pass the parsed-row and horizon-pairing checks.
