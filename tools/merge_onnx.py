"""
Merge external ONNX data into a single self-contained .onnx file.
This is needed because onnxruntime-web (WASM) cannot load external data files.
"""
import onnx
from onnx.external_data_helper import convert_model_to_external_data
import os

model_path = os.path.join("public", "models", "pidinet_tiny.onnx")
output_path = os.path.join("public", "models", "pidinet_tiny_merged.onnx")

print(f"Loading model from {model_path}...")
model = onnx.load(model_path, load_external_data=True)

print("Converting to self-contained model (no external data)...")
# Remove all external data references and embed weights inline
for tensor in model.graph.initializer:
    tensor.data_location = onnx.TensorProto.DEFAULT

print(f"Saving merged model to {output_path}...")
onnx.save(model, output_path)

size = os.path.getsize(output_path)
print(f"Done! Merged model size: {size / 1024:.1f} KB")

# Print model inputs/outputs
print(f"Inputs: {[i.name for i in model.graph.input]}")
print(f"Outputs: {[o.name for o in model.graph.output]}")
