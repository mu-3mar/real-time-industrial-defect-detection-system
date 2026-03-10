# `Boxes/flow/models`

Model artifacts used by the flow runtime.

Typical contents:

- `detect_box.*`: box detector (ONNX / TensorRT engine / training checkpoint)
- `defect_box.*`: defect detector (ONNX / TensorRT engine / training checkpoint)

Config references:

- `config/box_detector.yaml` → `models/detect_box.engine`
- `config/defect_detector.yaml` → `models/defect_box.engine`
