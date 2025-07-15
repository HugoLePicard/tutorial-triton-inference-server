import os
import torch
import numpy
import onnx
import onnxruntime as ort

from diffusers import StableDiffusionPipeline

from utils import *

pipe = StableDiffusionPipeline.from_pretrained(DIFFUSION_MODEL_ID, torch_dtype=torch.float16)

VAE_DECODER_DIR = os.path.join(MODELS_DIR_PATH, "vae_decoder")
os.makedirs(VAE_DECODER_DIR, exist_ok=True)
VAE_DECODER_ONNX_PATH = os.path.join(VAE_DECODER_DIR, "model.onnx")

vae_decoder = pipe.vae
vae_decoder.forward = vae_decoder.decode 
vae_decoder.eval().to("cuda")

sample = torch.randn(1, 4, 64, 64, dtype=torch.float16).to("cuda")

with torch.no_grad():
    torch.onnx.export(
        vae_decoder,
        sample,
        VAE_DECODER_ONNX_PATH,
        input_names  = ["sample"],
        output_names = ["image"],
        dynamic_axes = {
            "sample": {0: "batch"},
        },
        opset_version = 18,
    )

with torch.no_grad():
    output_pytorch = vae_decoder(sample)[0]

ort_session  = ort.InferenceSession(VAE_DECODER_ONNX_PATH, providers=["CUDAExecutionProvider"])
sample_numpy = sample.cpu().numpy()
output_onnx  = ort_session.run(None, {"sample": sample_numpy})

print("Are VAE decoder outputs the same?", numpy.allclose(output_pytorch.cpu().numpy(), output_onnx[0], rtol=1e-02, atol=1e-02))