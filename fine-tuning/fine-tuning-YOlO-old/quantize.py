from onnxruntime.quantization import quantize_dynamic, QuantType

input_model = "fine-tuning/fine-tuning-YOlO-old/runs/train/train/weights/best.onnx"

output_model = "fine-tuning/fine-tuning-YOlO-old/runs/train/train/weights/best_int8.onnx"

print("[INFO] Input model :", input_model)
print("[INFO] Output model:", output_model)

quantize_dynamic(
    model_input=input_model,
    model_output=output_model,
    weight_type=QuantType.QUInt8,
)

print("[OK] Quantized model saved to:", output_model)
