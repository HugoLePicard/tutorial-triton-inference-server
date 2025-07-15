# Tutorial: Deploying Deep Learning Models with TensorRT & Triton Inference Server

## Model Optimization & Inference at Scale

This is a hands-on tutorial designed to help you get practical experience with TensorRT and Triton Inference Server — two powerful tools for deploying deep learning models in production.

- **TensorRT** is NVIDIA’s high-performance deep learning inference optimizer and runtime. It helps speed up models, notably on GPUs.
- **Triton Inference Server** is NVIDIA’s production-ready tool for serving deep learning models. It simplifies deployment by handling queuing, batching, scheduling, and supports models from many frameworks (PyTorch, TensorFlow, ONNX, etc.).

In this tutorial, we'll take a complete Stable Diffusion v1.5 pipeline, convert its components to ONNX, optimize them using TensorRT, and deploy the full pipeline on Triton Inference Server.

We'll cover:

- Converting and optimizing models with TensorRT
- What is a model ensemble and how to build one
- How to use Triton's Python backend and BLS (Business Logic Scripting)
- How to build an ensemble of ensembles for multi-stage inference

This is not an advanced or exhaustive guide. It’s meant to provide a working example and walk through the full deployment flow in a clear, hands-on way. The goal is to help you understand how the pieces fit together — not to go deep into every component.

## Installation

Clone the repository and install the required Python packages:

```bash
git clone https://github.com/your-username/tutorial-triton-inference-server.git
cd tutorial-triton-inference-server
pip install -r requirements.txt
```

We also include a setup.py to make local imports easier when working across multiple scripts.

To install the project in editable mode (so you can import modules like from utils import X):

```bash
pip install -e .
```

This will allow you to use relative imports cleanly throughout the project, without needing to modify PYTHONPATH.

## Step 1 – Exporting Models to ONNX

### What is ONNX?

[ONNX](https://onnx.ai/) (Open Neural Network Exchange) is an open standard format for machine learning models. It allows models trained in one framework (like PyTorch) to be exported and run in another (like TensorRT, ONNX Runtime, etc.), enabling flexible and high-performance inference across tools.

### Why ONNX for Stable Diffusion?

In a diffusion model pipeline, the heaviest components in terms of computation are:

- The **Text Encoder** (usually a CLIP transformer)
- The **UNet** (the denoising core of the model)
- The **VAE Decoder** (to reconstruct images from latents)
- The **VAE Encoder** (if using img2img or other conditioning)

These are the components we'll export to ONNX in this tutorial.

All export scripts are located in the `/optim` folder:

optim/
├── convert_text_encoder_to_onnx.py
├── convert_unet_to_onnx.py
├── convert_vae_decoder_to_onnx.py
├── convert_vae_encoder_to_onnx.py

Each script loads the original model from Hugging Face using the `diffusers` library and exports it to ONNX format using `torch.onnx.export`.

### Dynamic Axes: Flexibility vs Performance

When exporting to ONNX, you can choose which dimensions of your inputs are **dynamic** — i.e., allowed to change at inference time.

For example:

```python
dynamic_axes = {
    "sample": {0: "batch"},
    "encoder_hidden_states": {0: "batch"}
}
```

Here, the batch dimension is dynamic, which means the model will accept any batch size. This adds flexibility for deployment, especially in production queues. However, dynamic axes can limit the optimization TensorRT can apply (compared to fully static shapes), possibly reducing performance gains.

You can make:

All axes static (faster but rigid)

Some axes dynamic (flexible with moderate optimization)

All axes dynamic (most flexible, least optimized)

In this tutorial, we’ll keep things simple and only make the batch axis dynamic.

Lets run those scripts one by one with :

```bash
python ./optim/convert_text_encoder_to_onnx.py
python ./optim/convert_unet_to_onnx.py
python ./optim/convert_vae_encoder_to_onnx.py
python ./optim/convert_vae_decoder_to_onnx.py
```

Next step: we’ll optimize these ONNX models with TensorRT.

## Step 2 – Optimizing ONNX Models with TensorRT

### Why use Docker?

TensorRT is tightly coupled with specific versions of CUDA, cuDNN, and PyTorch. Instead of installing and maintaining this stack locally, we’ll use **Docker** to keep everything isolated and reproducible.

We provide two Dockerfiles:

- `./docker/pytorch/Dockerfile` — for ONNX export and TensorRT optimization
- `./docker/triton/Dockerfile` — for running the Triton Inference Server later

> It’s important that both Docker images use the **same base image** (here `nvcr.io/nvidia/...:24.10-py3`) to ensure compatibility between the optimized models and the runtime. Different versions may lead to runtime errors or unsupported ops.

### Building the PyTorch + TensorRT Optimization Image

We’ll first build the PyTorch container, which contains all the tooling needed to export and optimize ONNX models with TensorRT.

```bash
cd ./docker/pytorch
docker build -t pytorch:latest .
```

Once the image is built, we’ll use it to run optimization scripts (next section).

The Triton image (./docker/triton/Dockerfile) will be used later when deploying models in production. Let's build it now so its done!

```bash
cd ../triton
docker build -t triton:latest .
```

Go back to root

```bash
cd ../..
```

## Step 3 – Organizing for Triton & Starting Optimization

Now that all models have been exported to ONNX and our Docker images are built, we can move on to optimizing the models with TensorRT and setting up the Triton Inference Server.

### Triton Model Repository Structure

Triton follows a strict directory structure for serving models. All models must live inside a single directory (here: `triton/models`), and each model must follow this format:

triton/models/
├── unet/
│ ├── 1/
│ │ └── model.plan # or model.onnx, depending on what you serve
│ └── config.pbtxt
├── text_encoder/
│ ├── 1/
│ │ └── model.plan
│ └── config.pbtxt


Each subdirectory under `models/` represents a single model. Inside it:
- The `1/` folder contains the versioned model file (`model.onnx` for ONNX, `model.plan` for TensorRT).
- The `config.pbtxt` file defines the model’s inputs, outputs, and deployment parameters (like batch size, data types, GPU config, etc.).

> You can technically use ONNX models directly in Triton, but using TensorRT-optimized `.plan` files typically gives a significant boost in inference speed.

In the next step, we’ll show how to generate `.plan` files from the ONNX models using TensorRT.

## Step 4 – Preparing the Triton Model Directory

Let’s now move our exported ONNX models into the Triton-compatible directory layout.

Run the following script:

```bash
python ./scripts/move_onnx_models_to_triton.py
```

This will simply copy each ONNX model from data/models to triton/models/[model_name]/1/model.onnx and create a 2/ folder for the optimized TensorRT engine (model.plan) later.

### Triton Model Versioning
Triton supports model versioning natively. Inside each model folder (unet/, text_encoder/, etc.), you can include multiple version folders like 1/, 2/, etc.

We’ll later configure Triton to use the TensorRT version by pointing to model.plan inside 2/.

Next, we’ll run the actual TensorRT optimization step.

# Model Compilation with TensorRT

We’ll use a PyTorch-based container (which includes Python and commonly used ML libraries) to compile our models into TensorRT engines. This step is done inside the container to avoid installing TensorRT locally — thanks to Docker’s isolation and prebuilt environments.

## Running the PyTorch Container

```bash
docker run -it --ipc host -v ./triton:/workspace/triton --gpus=all pytorch:latest bash
```

- -it: interactive mode with a shell
- --ipc host: allows shared memory (needed for TensorRT)
- -v ./triton:/workspace/triton: mounts your local ./triton directory into the container
- --gpus=all: enables GPU access
- pytorch:latest: or any other base image you use that has PyTorch + TensorRT installed

# Inside the Container

Once inside the container:

1. Navigate to the models folder:

```bash
cd /triton/models
```

2. TensorRT Model Conversion

Once you're inside the Docker container and have cd into /triton/models/[model], it's time to convert each ONNX model into a TensorRT engine (.plan file) optimized for inference.

### Why Dynamic Shapes Matter

1. Static shapes: If your model was exported with fixed input shapes, you can run a simple trtexec command without specifying shapes.

2. Dynamic shapes: If your model was exported with dynamic axes (e.g., variable batch sizes or input dimensions), you must specify minimum, optimum, and maximum input shapes during conversion. This allows TensorRT to build an optimized engine that supports a range of input sizes.

## Basic Conversion (No Dynamic Shapes)

If your model has fixed input shapes:

```bash
trtexec --onnx=./1/model.onnx \
        --saveEngine=./2/model.plan \
        --fp16
```

This uses FP16 precision for better performance and smaller memory footprint. TensorRT will infer the input shapes from the model itself.

## Custom Shape Configuration (For Dynamic Axes)

If your model includes dynamic shapes, you need to specify shape ranges with --minShapes, --optShapes, and --maxShapes for each input.

Here are the recommended shape settings for each model:

### Unet

```bash
trtexec \
  --onnx=./1/model.onnx \
  --saveEngine=./2/model.plan \
  --minShapes=sample:1x4x64x64,timestep:1,encoder_hidden_states:1x77x768 \
  --optShapes=sample:2x4x64x64,timestep:1,encoder_hidden_states:2x77x768 \
  --maxShapes=sample:4x4x64x64,timestep:1,encoder_hidden_states:4x77x768 \
  --fp16
```

- sample: latent input (batch x channels x H x W)
- encoder_hidden_states: output of text encoder, variable batch size

### Text Encoder

```bash
trtexec \
  --onnx=./1/model.onnx \
  --saveEngine=./2/model.plan \
  --minShapes=tokens:1x77 \
  --optShapes=tokens:2x77 \
  --maxShapes=tokens:4x77 \
  --fp16
```

- tokens: input token IDs (batch x sequence_length)

### VAE Encoder

```bash
trtexec \
  --onnx=./1/model.onnx \
  --saveEngine=./2/model.plan \
  --minShapes=image:1x3x512x512 \
  --optShapes=image:1x3x512x512 \
  --maxShapes=image:2x3x512x512 \
  --fp16
```

- image: RGB image (normalized) input (batch x channels x H x W)

### VAE Decoder

```bash
trtexec \
  --onnx=./1/model.onnx \
  --saveEngine=./2/model.plan \
  --minShapes=sample:1x4x64x64 \
  --optShapes=sample:1x4x64x64 \
  --maxShapes=sample:2x4x64x64 \
  --fp16
```

- sample: latent representation to decode

### Note:

- Make sure your ONNX models were exported with **dynamic axes** (`dynamic_axes={...}` in `torch.onnx.export`) **if you're specifying custom shapes**. Otherwise, `trtexec` will raise an error about mismatched input shapes during optimization.

- **TensorRT-optimized models are hardware-dependent**. The `.plan` files generated by TensorRT are highly optimized for the **specific GPU architecture** (SM version) used during compilation.

  For example:
  - If you optimize the model on an **NVIDIA A100**, you **must** deploy it on another A100 or a GPU with the **same compute capability**.
  - To deploy on a different GPU (e.g., T4, V100, or RTX 4090), you’ll need to **recompile the engine** using `trtexec` on that target GPU.

  This ensures the generated kernels match the GPU's architecture and capabilities.

# Results

When you run trtexec on a model, TensorRT will not only optimize and export the .plan file, it will also benchmark the model's GPU inference performance. After a short warmup and a few runs, you'll see output like:

## Example Benchmark Results on A100 40G

For the UNet:

```bash
[I] GPU Compute Time: min = 14.7405 ms, max = 14.7978 ms, mean = 14.7559 ms, ...
```

This means that a single forward pass through the UNet takes ~14.76 ms on average.

For the Text Encoder:

```bash
[I] GPU Compute Time: mean = 0.855457 ms
```

This is extremely fast, less than 1 millisecond. It’s only run once at the beginning to encode the text prompt.

For the VAE Encoder:

```bash
[I] GPU Compute Time: mean = 8.52 ms
```

Used if you're feeding an image in latent form (e.g. for inpainting or image-to-image generation).

For the VAE Decoder:

```bash
[I] GPU Compute Time: mean = 17.05 ms
```

This decodes the final latent back to an image, typically run once at the end of the diffusion loop.

## How to Interpret the Metrics

Each result gives:

- min: Fastest single run
- max: Slowest single run
- mean: Average time across all runs
- median: Middle value
- percentile(90/95/99%): Time under which that % of runs complete (e.g., 99% of runs were faster than 14.79 ms)

These numbers help you understand latency stability and identify performance bottlenecks.

## Total Inference Time for 50 Diffusion Steps (Theoretical)

In a standard Stable Diffusion pipeline:

- You run the UNet once per denoising step: 50 steps ⇒ 50 × UNet
- The Text Encoder is run once at the beginning
- The VAE Decoder is run once at the end to produce the final image

So, total compute time in the ideal case (no overhead):

```bash
Total ≈ (50 × UNet time) + Text Encoder time + VAE Decoder time
      ≈ (50 × 14.76 ms) + 0.86 ms + 17.05 ms
      ≈ 738 ms + 0.86 ms + 17.05 ms
      ≈ ~756.9 ms
```

That’s approximately 1.32 frames per second, which remains very fast for full-resolution image generation — thanks to TensorRT’s low-latency execution on high-performance GPUs like the A100.

# Triton Inference Server - Model Config Explanation (`config.pbtxt`)

This document explains how to define a standard model configuration for use with Triton Inference Server, specifically for models optimized with TensorRT. This is **not** an ensemble model — it is a basic configuration for a single optimized model.

## Example: `unet/config.pbtxt`

```protobuf
name: "unet" 
backend: "tensorrt"
platform: "tensorrt_plan"
max_batch_size: 0

version_policy: { specific { versions: [2] } }

input [
  {
    name: "sample"
    data_type: TYPE_FP16
    dims: [ -1, 4, 64, 64 ]
  },
  {
    name: "timestep"
    data_type: TYPE_FP16
    dims: [ 1 ]
  },
  {
    name: "encoder_hidden_states"
    data_type: TYPE_FP16
    dims: [ -1, 77, 768 ]
  }
]

output [
  {
    name: "outputs"
    data_type: TYPE_FP16
    dims: [ -1, 4, 64, 64 ]
  }
]

instance_group [
  {
    kind: KIND_GPU
  }
]
```

## Field-by-Field Breakdown
### name
The name of the model. This must match the folder name in triton/models/. For example, triton/models/unet/.

### backend
Specifies the backend that will execute the model. Here, tensorrt is used for models optimized with TensorRT.

### platform
Set to tensorrt_plan when using serialized TensorRT engines (model.plan).

### max_batch_size
Set to 0 to disable batching. If your model supports batching, you can specify a positive integer.

### version_policy
Specifies which version of the model to load. In this case, Triton will only use version 2, i.e. the file at unet/2/model.plan.

## Input and Output Declarations
Every input and output must match exactly:

The names used during ONNX export

The shapes used during TensorRT optimization

The data types (e.g., FP16 if you used --fp16)

## Inputs

```protobuf
input [
  {
    name: "sample"
    data_type: TYPE_FP16
    dims: [ -1, 4, 64, 64 ]
  },
  {
    name: "timestep"
    data_type: TYPE_FP16
    dims: [ 1 ]
  },
  {
    name: "encoder_hidden_states"
    data_type: TYPE_FP16
    dims: [ -1, 77, 768 ]
  }
]
```

- sample: The latent tensor with dynamic batch size
- timestep: A scalar input representing the diffusion step
- encoder_hidden_states: The text conditioning tensor

## Outputs

```protobuf
output [
  {
    name: "outputs"
    data_type: TYPE_FP16
    dims: [ -1, 4, 64, 64 ]
  }
]
```

- outputs: The predicted noise (or velocity, depending on training) with the same shape as sample.

## Instance Group

```protobuf
instance_group [
  {
    kind: KIND_GPU
  }
]
```

Specifies that the model will run on the GPU. You can also add:

```protobuf
gpus: [ 0 ]
count: 2
```

To control GPU selection and parallelism.

## Notes
- Triton will strictly validate that incoming inference requests match the declared input/output names, types, and shapes.
- If anything is misaligned, Triton will raise an error at runtime and refuse to execute the model.
- If your ONNX export used --dynamic_axes, you must reflect that with dynamic dimensions (e.g. -1).
- If your TensorRT optimization used --fp16, you must set data_type: TYPE_FP16.
- Use trtexec logs to inspect actual tensor names and shapes to verify alignment.

# Launching the Triton Inference Server

Once all your models are in place under the triton/models/ directory (with the correct config.pbtxt and serialized model.plan files), you can launch the Triton Inference Server using Docker:

```bash
docker run --gpus all --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  --shm-size=10g --ipc=host --pid=host \
  -v ./triton/models/:/models \
  triton:latest \
  tritonserver --model-repository=/models
```

- --gpus all: enables GPU access
- --shm-size=10g: increases shared memory size to avoid out-of-memory errors for large tensors
- -v ./triton/models/:/models: mounts your local models directory
- tritonserver --model-repository=/models: tells Triton where to find your models

## Querying Individual Models in Python

Once the server is running, you can send inference requests using the Triton Python client. Example for one component (e.g., vae_decoder, unet, or text_encoder):

```python
import numpy
import tritonclient.grpc as grpcclient

input_array = pytorch_tensor.cpu().numpy().astype(numpy.float16)

input_tensor = grpcclient.InferInput("input_name", input_array.shape, "INPUT_TYPE")
input_tensor.set_data_from_numpy(input_array)

result = client.infer(
    model_name = "your_model_name_here",
    inputs     = [input_tensor],
    outputs    = [grpcclient.InferRequestedOutput("your_output_name_here")]
)

output_array = result.as_numpy("your_output_name_here")
```

Key Components
- InferInput("sample", shape, dtype): Matches the model's config.pbtxt input name and shape.
- .set_data_from_numpy(...): Loads the NumPy data into the request.
- client.infer(...): Sends the request to a specific model.
- as_numpy(...): Extracts the output as a NumPy array.

Repeat the same logic for other models:

- For the text encoder, input would be input_ids, output last_hidden_state.
- For the unet, inputs would include sample, timestep, and encoder_hidden_states.
- For the vae decoder, input is sample, output is image.

## Next Step: Stitching the Pipeline Together with BLS

Now that each component (text encoder, UNet, VAE encoder/decoder) is individually deployable and callable, we are ready to link them together into a single pipeline using BLS (Backend Lifecycle Scripting).

BLS lets you define a custom Python backend model that orchestrates multiple sub-model calls inside Triton itself — no need to call each model from your client manually. 

We'll define a Python model under triton/models/pipeline_text_to_image/ and script the full inference logic using Triton's Python backend API.

Let’s build that next.

# A Minimal Stable Diffusion Pipeline (Client-Side)
Before moving on to implementing the full pipeline using BLS (Backend Lifecycle Scripting) in Triton, which can be cumbersome and less developer-friendly, we recommend starting with a client-side implementation.

## Why?
- Debugging and iteration are much faster on the client.
- You get complete control over the pipeline logic.
- You can test each model independently and ensure inputs/outputs align.
- Once everything works, you can port it step-by-step into Triton BLS with confidence.

## What This Looks Like
Instead of building an ensemble model or BLS backend right away, we call each model individually from the client in Python:

1. Text encoder:
    - Call the text_encoder model to get the conditioning (encoder_hidden_states).

2. UNet loop:
    - For each timestep in the denoising process, call the unet model with the current latent and conditioning.

3. VAE decoder:
    - Call the vae_decoder model once at the end to decode the latent into an image.

This mimics the structure of Hugging Face's StableDiffusionPipeline, but each model call is routed through Triton.

## Minimal Implementation
We rewrote a minimalist version of the StableDiffusionPipeline from the 🤗 diffusers library, keeping only the essentials to interface with Triton.

You can find the pipeline implementation here:

```bash
pipelines/ensemble.py
```

To test if everything works end-to-end with your deployed models, run the following script:

```bash
./scripts/run_ensemble_pipeline.py
```

This will:

- Load a prompt
- Encode it with the text_encoder
- Run the diffusion process with the unet
- Decode the final latent with the vae_decoder
- Output the generated image

Once you have verified this works correctly on the client side, you're ready to port the logic to Triton using BLS.

# Triton BLS (Backend Lifecycle Scripting) with Python Backend
Once your pipeline works on the client side, you can move the logic server-side by using Triton’s Python backend. This allows you to define a full end-to-end pipeline (text encoder → unet loop → vae decoder) as a single Triton model called something like pipeline_text_to_image.

## 1. config.pbtxt for the BLS Model
To enable Python logic, you just need a minimal config.pbtxt:

```protobuf
name: "pipeline_text_to_image"
backend: "python"
max_batch_size: 0
```

This model doesn’t contain any .onnx or .plan file — instead, Triton will expect to find a model.py file under the 1/ folder:

triton/models/pipeline_text_to_image/
├── 1/
│   └── model.py
└── config.pbtxt

## 2. Basics of model.py
Triton requires you to define a class called TritonPythonModel with two key methods:

```python
class TritonPythonModel:
    def initialize(self, args):
        # Called once when the model is loaded
        pass

    def execute(self, requests):
        responses = []

        for request in requests:
            # extract input tensors, call other models, process outputs...
            responses.append(pb_utils.InferenceResponse(output_tensors))

        return responses
```

Each incoming request is an instance of InferenceRequest, and your response must be a list of InferenceResponse objects.

## 3. Using pb_utils
Triton provides a helper module triton_python_backend_utils (imported as pb_utils) to:

- Parse inputs from a request
- Call other Triton models (submodels like text_encoder, unet, vae_decoder)
- Return outputs as InferenceResponse

Here’s how to extract and parse input tensors:

```python
prompt_tensor = pb_utils.get_input_tensor_by_name(request, "prompt")
if not prompt_tensor:
    raise ValueError("prompt is required")

prompt = [t.decode("utf-8") for t in prompt_tensor.as_numpy().tolist()]

guidance_scale = 7.5
if (guidance_scale_tensor := pb_utils.get_input_tensor_by_name(request, "guidance_scale")):
    guidance_scale = guidance_scale_tensor.as_numpy().tolist()[0]

num_inference_steps = 50
if (steps_tensor := pb_utils.get_input_tensor_by_name(request, "num_inference_steps")):
    num_inference_steps = steps_tensor.as_numpy().tolist()[0]

seed = 0
if (seed_tensor := pb_utils.get_input_tensor_by_name(request, "seed")):
    seed = seed_tensor.as_numpy().tolist()[0]

negative_prompt = None
if (negative_prompt_tensor := pb_utils.get_input_tensor_by_name(request, "negative_prompt")):
    negative_prompt = [
        t.decode("utf-8") for t in negative_prompt_tensor.as_numpy().tolist()
    ]
```

This gives you full control over the inference parameters.

## 4. DLPack and Why We Use It

Triton supports dlpack for zero-copy memory sharing between PyTorch and Triton’s internal inference format.

This is how you convert PyTorch tensors into Triton-compatible input tensors:

```python
inference_request = pb_utils.InferenceRequest(
    model_name = "unet",
    inputs = [
        pb_utils.Tensor.from_dlpack("sample", torch.to_dlpack(latent_model_input)),
        pb_utils.Tensor.from_dlpack("timestep", torch.to_dlpack(timestep)),
        pb_utils.Tensor.from_dlpack("encoder_hidden_states", torch.to_dlpack(prompt_embeds)),
    ],
    requested_output_names = ["outputs"]
)
```

Then execute and parse the response:

```python
inference_response = inference_request.exec()
if inference_response.has_error():
    raise pb_utils.TritonModelException(inference_response.error().message())

output_tensor = pb_utils.get_output_tensor_by_name(inference_response, "outputs")
output_tensor = torch.from_dlpack(output_tensor.to_dlpack())
```

This method avoids CPU round-trips and preserves performance when chaining submodels.

You now have everything in place to move your full Stable Diffusion pipeline into Triton using BLS.

The logic is fully server-side, with direct model-to-model calls and no client overhead. You can manage prompts, timesteps, random seeds, and loop structure all inside model.py.

# Running the Final BLS Pipeline

Once your pipeline_text_to_image model is defined using Triton’s Python backend (model.py), the client no longer needs to handle the entire diffusion logic. The pipeline becomes extremely lightweight, as Triton now handles all the orchestration server-side.

You can find the minimal client for calling this final BLS pipeline here:

```bash
pipelines/bls.py
```

To test it end-to-end, simply run:

```bash
scripts/run_bls_pipeline.py
```

This script sends a request to the pipeline_text_to_image model deployed on Triton, passing high-level inputs like prompt, num_inference_steps, and guidance_scale. Triton takes care of everything else — encoding the prompt, looping through the UNet, and decoding the final image.

This marks the transition from a client-driven pipeline to a fully self-contained server-side inference system.

# Benchmarking the Three Pipelines

You can run the following script to benchmark and compare the three pipelines:

```bash
scripts/benchmark_pipelines.py
```

## This script benchmarks:
- Hugging Face Diffusers (pure PyTorch)
- Triton Ensemble Pipeline (client-side orchestration)
- Triton BLS Pipeline (server-side orchestration via Python backend)

## Results (20 runs)
```less
HuggingFace Diffusers Average Time: 2.35 s ± 0.20 s  
Triton Ensemble        Average Time: 0.97 s ± 0.01 s  
Triton BLS             Average Time: 0.91 s ± 0.01 s  
```

These results show the performance improvements achieved by offloading computation to Triton and minimizing client-side logic. The BLS version is the fastest, with everything handled directly on the server.

These results highlight the significant performance gains from offloading computation to Triton Inference Server. By eliminating Python overhead and leveraging optimized TensorRT engines, the inference time drops by more than half compared to the Hugging Face implementation.

The Triton BLS version is the fastest overall, as the entire pipeline is handled server-side — including model orchestration, latent processing, and memory management.

The remaining latency is mostly due to:
- Python execution within the BLS model.py (control flow, tensor conversions, etc.)
- gRPC communication overhead (sending the prompt and retrieving the final image over the network)

In pure compute terms (as estimated earlier), the pipeline could theoretically run in ~756 ms — so we're very close to the hardware limits, with most of the gap explained by I/O and Python glue logic.