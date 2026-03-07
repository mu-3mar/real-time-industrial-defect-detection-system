# Repository Refactoring Summary

## 1. New Folder Structure Tree

```
project-root/
├── Boxes/
│   └── flow/
│       ├── api/
│       │   ├── __init__.py
│       │   └── api_server.py
│       ├── api/
│       │   ├── __init__.py
│       │   └── api_server.py
│       ├── config/
│       │   ├── .env.example
│       │   ├── api.yaml
│       │   ├── app.yaml
│       │   ├── box_detector.yaml
│       │   ├── defect_detector.yaml
│       │   ├── firebase.yaml
│       │   ├── firebase_config.json.example
│       │   ├── README.md
│       │   ├── stream.yaml
│       │   └── webrtc.example.yaml
│       ├── core/
│       │   ├── device_manager.py
│       │   ├── firebase_client.py
│       │   ├── model_loader.py
│       │   ├── pipeline.py
│       │   ├── session_manager.py
│       │   ├── session_worker.py
│       │   ├── state.py
│       │   ├── stream.py
│       │   └── webrtc_track.py
│       ├── docs/
│       │   ├── README.md
│       │   ├── endpoints.md
│       │   └── REFACTORING_SUMMARY.md
│       ├── scripts/
│       │   └── run_dev.sh
│       ├── requirements/
│       │   └── requirements.txt
│       ├── detectors/
│       │   └── detector.py
│       ├── models/
│       │   └── (onnx model files)
│       ├── utils/
│       │   ├── geometry.py
│       │   └── visualizer.py
│       └── main.py
│
├── pyproject.toml
├── .gitignore
├── README.md
├── index.html
└── unused_files.md
```

(Boxes/trainig/ unchanged; not listed above.)

---

## 2. Files Moved

| From | To |
|------|-----|
| `Boxes/flow/api_server.py` | `Boxes/flow/api/api_server.py` |
| `device_manager.py` (repo root) | `Boxes/flow/core/device_manager.py` |
| `API-ENDPOINTS.md` (repo root) | `Boxes/flow/docs/endpoints.md` |
| `.env.example` (repo root) | `Boxes/flow/config/.env.example` |

**New files created (not moves):**

- `Boxes/flow/api/__init__.py`
- `Boxes/flow/docs/README.md`, `endpoints.md`, `REFACTORING_SUMMARY.md`
- `Boxes/flow/config/.env.example`
- `Boxes/flow/scripts/run_dev.sh`
- `Boxes/flow/requirements/requirements.txt`
- `unused_files.md` (repo root)

**Deleted after move:**

- `Boxes/flow/api_server.py` (replaced by api/api_server.py)
- `device_manager.py` (replaced by Boxes/flow/core/device_manager.py)
- `API-ENDPOINTS.md` (replaced by docs/endpoints.md)
- `.env.example` (replaced by configs/.env.example)

---

## 3. Unused Files Detected

See **unused_files.md** at repo root.

**Summary:** One candidate — `Boxes/trainig/box-YOLO/scripts/convet seg - od.py` (typo in name, not referenced in docs or imports). Do not delete without review.

---

## 4. Imports and Path Updates

### Boxes/flow

| File | Change |
|------|--------|
| `main.py` | `api_server:app` → `api.api_server:app` |
| `api/api_server.py` | Removed repo-root `sys.path` and `from device_manager`; added `from core.device_manager import select_device`. Base path: `Path(__file__).resolve().parent.parent` (flow dir). Load `.env` from `Boxes/flow/config/.env`, flow dir, and repo root. Model paths resolved relative to base when not absolute. |
| `config/box_detector.yaml` | `model_path`: `Boxes/flow/models/...` → `models/detect_box_int8.onnx` |
| `config/defect_detector.yaml` | `model_path`: `Boxes/flow/models/...` → `models/defect_box_int8.onnx` |

### Boxes/trainig

| File | Change |
|------|--------|
| `box-YOLO/training/train.py` | Replaced repo-root path + `from device_manager` with Boxes/flow on `sys.path` and `from core.device_manager import select_device`. |
| `defect-YOLO/training/train.py` | Same as above. |

### .gitignore

- Added `Boxes/flow/config/.env` so secrets remain untracked.

---

## Verification Checklist

After refactoring:

1. **Run server:** `./Boxes/flow/scripts/run_dev.sh` or `cd Boxes/flow && python main.py`. Ensure `FIREBASE_DATABASE_URL` is set in `Boxes/flow/config/.env` (copy from `.env.example`).
2. **API server:** Uvicorn loads `api.api_server:app`; health at `GET /api/health`.
3. **Firebase:** Unchanged; still uses `core.firebase_client` and Realtime Database.
4. **Models:** Config uses `models/detect_box_int8.onnx` and `models/defect_box_int8.onnx`; resolved relative to `Boxes/flow` at runtime.
5. **Training:** From `Boxes/trainig/box-YOLO` or `defect-YOLO`, `python training/train.py` (or `scripts/run_all.py`) should still work with `core.device_manager` from Boxes/flow.
