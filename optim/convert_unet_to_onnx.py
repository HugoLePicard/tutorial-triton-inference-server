import os
import torch
import numpy
import onnxruntime as ort

from diffusers import StableDiffusionPipeline

from utils import *

pipe = StableDiffusionPipeline.from_pretrained(DIFFUSION_MODEL_ID, torch_dtype =torch.float16)

directory_unet = os.path.join(MODELS_DIR_PATH, "unet")
os.makedirs(directory_unet, exist_ok =True)
path_unet_onnx = os.path.join(directory_unet, 'model.onnx')

unet = pipe.unet.to("cuda")
unet.eval()

inputs = {
    "sample"               : torch.randn((2, 4, 64, 64)).to(torch.float16).to("cuda"),
    "timestep"             : torch.randn((1)).to(torch.float16).to("cuda"),
    "encoder_hidden_states": torch.randn((2,77,768)).to(torch.float16).to("cuda"),
}

input_names = ["sample", "timestep"]

with torch.no_grad():
    torch.onnx.export(
        unet, 
        inputs, 
        path_unet_onnx, 
        input_names   = list(inputs), 
        output_names  = ["outputs"], 
        opset_version = 18,
        dynamic_axes  = {
            "sample"                  : {0: "batch"},
            "encoder_hidden_states"   : {0: "batch"},
        }
    )

sample                = torch.randn((2, 4, 64, 64)).to(torch.float16).to("cuda")
timestep              = torch.randn((1)).to(torch.float16).to("cuda")
encoder_hidden_states = torch.randn((2,77,768)).to(torch.float16).to("cuda")

with torch.no_grad():
    output_pytorch = unet(sample, timestep, encoder_hidden_states)

ort_session                 = ort.InferenceSession(path_unet_onnx, providers=['CUDAExecutionProvider'])
sample_numpy                = numpy.array(sample.cpu()  , dtype=numpy.float16)
timestep_numpy              = numpy.array(timestep.cpu(), dtype=numpy.float16)
encoder_hidden_states_numpy = numpy.array(encoder_hidden_states.cpu(), dtype=numpy.float16)
output_onnx                 = ort_session.run(None, {"sample": sample_numpy, "timestep": timestep_numpy, "encoder_hidden_states": encoder_hidden_states_numpy})

print("Are unet outputs the same? ", numpy.allclose(output_pytorch[0].cpu().numpy(), output_onnx[0], rtol=1e-02, atol=1e-02))