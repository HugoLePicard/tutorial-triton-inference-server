import os
import torch
import numpy
import onnx
import onnxruntime as ort

from diffusers import StableDiffusionPipeline

from utils import *

pipe = StableDiffusionPipeline.from_pretrained(DIFFUSION_MODEL_ID, torch_dtype =torch.float16)

TEXT_ENCODER_DIR = os.path.join(MODELS_DIR_PATH, "text_encoder")
os.makedirs(TEXT_ENCODER_DIR, exist_ok =True)
TEXT_ENCODER_ONNX_PATH = os.path.join(TEXT_ENCODER_DIR, "model.onnx")

input_ids = torch.ones((1, 77), dtype =torch.int64).to("cuda")

text_encoder = pipe.text_encoder.eval().to("cuda")

with torch.no_grad():
    torch.onnx.export(
        text_encoder,
        input_ids,
        TEXT_ENCODER_ONNX_PATH,
        input_names   = ["tokens"],
        output_names  = ["last_hidden_state", "pooler_out"],
        opset_version = 18,
        dynamic_axes  = {
            "tokens": {0: "batch"},
        }
    )

with torch.no_grad():
    output_pytorch = text_encoder(input_ids)

ort_session = ort.InferenceSession(TEXT_ENCODER_ONNX_PATH, providers=["CUDAExecutionProvider"])
onnx_output = ort_session.run(None, {"tokens": input_ids.cpu().numpy()})

print("Are text encoder outputs the same?", numpy.allclose(output_pytorch[0].cpu().numpy(), onnx_output[0], rtol=1e-02, atol=1e-02))