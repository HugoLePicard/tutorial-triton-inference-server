import os
import torch
import numpy
import onnx
import onnxruntime as ort

from diffusers import StableDiffusionPipeline

from utils import * 

pipe = StableDiffusionPipeline.from_pretrained(DIFFUSION_MODEL_ID, torch_dtype=torch.float16)

VAE_ENCODER_DIR = os.path.join(MODELS_DIR_PATH, "vae_encoder")
os.makedirs(VAE_ENCODER_DIR, exist_ok=True)
VAE_ENCODER_ONNX_PATH = os.path.join(VAE_ENCODER_DIR, "model.onnx")

vae_encoder = pipe.vae.eval().to("cuda")
image       = torch.randn(1, 3, 512, 512, dtype=torch.float16).to("cuda")

# This is a simplified example for tutorial purposes.
# Here the VAE encoder's forward pass returns a sampled latent,
# but because the sampling uses random noise inside the method, the ONNX tracing will "freeze" that noise.
# To do this properly, one should make sure that the noise is passed explicitly as an input to the model.
vae_encoder.forward = lambda x: vae_encoder.encode(x, False)[0].sample()

with torch.no_grad():
    torch.onnx.export(
        vae_encoder,
        image,
        VAE_ENCODER_ONNX_PATH,
        input_names  = ["image"],
        output_names = ["sample"],
        dynamic_axes = {
            "image": {0: "batch"},
        },
        opset_version = 18,
    )

with torch.no_grad():
    output_pytorch = vae_encoder(image)

ort_session = ort.InferenceSession(VAE_ENCODER_ONNX_PATH, providers=["CUDAExecutionProvider"])
image_numpy = image.cpu().numpy()
output_onnx = ort_session.run(None, {"image": image_numpy})

print("Are VAE encoder outputs the same?", numpy.allclose(output_pytorch.cpu().numpy(), output_onnx[0], rtol=1e-02, atol=1e-02))