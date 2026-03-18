# Picota Artifacts

This repository contains the current artifact bundle for two runnable case studies and their supporting material:

- `study_cases/energigran`
- `study_cases/europlatano`

Each case is self-contained under its own `code/`, `analysis/`, and `util/` directories. The runnable entry points are:

- `study_cases/energigran/code/energigran.py`
- `study_cases/europlatano/code/europlatano.py`

The repository also includes:

- `related_work/`, a paper and reference archive
- `picota.html`, a standalone documentation page

## Layout

- `study_cases/energigran/code/` contains the Energigran training script, local trainers, the local `kan/` package, and `bootstrap.sh`
- `study_cases/energigran/analysis/` contains analysis outputs from the case
- `study_cases/energigran/util/` contains supporting notebooks and utilities
- `study_cases/europlatano/code/` contains the Europlátano training script, local trainers, the local `kan/` package, and `bootstrap.sh`
- `study_cases/europlatano/analysis/` contains analysis outputs from the case
- `study_cases/europlatano/util/` contains supporting utilities and data-preparation scripts
- `study_cases/europlatano/data_example/europlatano.tsv` is a bundled example TSV for inspection and smoke testing

## Case Pointers

- Energigran predicts `generation` from hourly tabular inputs. See [study_cases/energigran/README.md](/Users/oroncal/workspace/research/picota-artifacts/study_cases/energigran/README.md) for run instructions.
- Europlátano predicts `Production` with a 28-day horizon from a production table. See [study_cases/europlatano/README.md](/Users/oroncal/workspace/research/picota-artifacts/study_cases/europlatano/README.md) for run instructions.

## Quick Start

From each case’s `code/` directory:

```bash
cd study_cases/energigran/code
bash bootstrap.sh
python3 energigran.py --help
```

```bash
cd study_cases/europlatano/code
bash bootstrap.sh
python3 europlatano.py --help
```

For actual training runs, pass a dataset explicitly with `--csv` or `--tsv`, or set the matching environment variable described in the case README.

## Dataset Availability

- Energigran does not ship a sample dataset in this repository.
- Europlátano ships `study_cases/europlatano/data_example/europlatano.tsv`, but it is too small for real training and fails the minimum-row checks.

## Limitations

- These scripts are intended to be run locally from their `code/` directories.
- The bundled Europlátano example is useful for inspection, not for training.
- A successful run still requires a dataset large enough to satisfy each script’s input validation and training split requirements.
- The historical Quassar / `model.tara` / `report.pdf` artifact layout is not part of this checkout.

## License

This repository is released under the MIT License.

