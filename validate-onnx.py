import onnx
import onnxruntime as ort
import numpy as np

filename = "sr.onnx"
#input_name = "input"
input_name = "hello"

model = onnx.load(filename)
onnx.checker.check_model(model)
onnx.helper.printable_graph(model.graph)

ort_session = ort.InferenceSession(filename)
outputs = ort_session.run(None, {input_name: np.random.randn(1, 1, 112, 112).astype(np.float32)})
print(outputs[0])
