# QC-SCM

Real-time industrial quality inspection using a two-stage detector:

1. **Box detection** localizes the product.
2. **Defect detection** analyzes the localized product and publishes stable results.

The repository is split by responsibility so production runtime code, historical
training versions, deployable models, and the operator dashboard are not mixed
in one `Boxes` directory.

## Repository layout

```text
src/qc_scm/runtime/       FastAPI service, pipeline, detectors, and config
training/versions/        Reproducible versioned training pipelines
  v1_0/                   Original baseline pipelines
  v2_0/                   V2.0 pipelines
  v2_1/                   Current V2.1 pipelines
dashboard/                Local operator UI and service manager
docs/                     Architecture and operational documentation
```

Each training version contains its own `configs/`, `training/`, `export/`,
`inference/`, `scripts/`, `models/`, and `runs/` directories. This keeps
historical experiments reproducible without ambiguous names such as `box` or
`BOX`.

## Runtime setup

Python 3.10+ and the `qc` Conda environment are recommended.

```bash
conda activate qc
pip install -e .
```

The runtime no longer requires Firebase. Detection results are written to a
temporary JSON file for each active session under
`src/qc_scm/runtime/.session_data/`; the file is deleted when the session
closes. Firebase was used as an earlier cloud integration, but is currently
disabled. The default detector configuration points to the V2.0
ONNX exports:

```text
training/versions/v2_0/box_detector/models/exported/best.onnx
training/versions/v2_0/defect_detector/models/exported/best.onnx
```

Start the runtime from the repository root:

```bash
./src/qc_scm/runtime/scripts/run_dev.sh
```

Or run it directly:

```bash
cd src/qc_scm/runtime
python main.py
```

## Dashboard

The dashboard is a static web UI. Its service manager starts and monitors the
runtime using the paths in `dashboard/service_manager.py`.

```bash
python dashboard/service_manager.py
```

For automatic user-level startup on Linux:

```bash
./dashboard/install_service_manager.sh
```

## Training

Run a specific historical pipeline from its version directory. For example:

```bash
cd training/versions/v2_1/box_detector
python scripts/run_all.py
```

Use the corresponding `training/versions/v2_1/defect_detector` directory for
defect training.
Training outputs stay with the version that produced them; do not copy the same
checkpoint into multiple version directories.

## Safety and repository hygiene

- Runtime configuration examples are committed; credentials and local state are
  not.
- Generated logs, service state, caches, and Python bytecode are ignored.
- The versioned model directories are the source of truth for deployable
  exports. Update the runtime YAML when intentionally promoting a model.

See [dashboard/README.md](dashboard/README.md) and
[src/qc_scm/runtime/config/README.md](src/qc_scm/runtime/config/README.md) for
operational details.
