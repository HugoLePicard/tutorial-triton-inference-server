# Deploying Deep Learning Models with TensorRT & Triton Inference Server
*A step-by-step tutorial using Stable Diffusion*

### What is TensorRT and Triton Inference Server

This hands-on tutorial will guide you through deploying deep learning models in production using two powerful tools from NVIDIA:

- **TensorRT** is a high-performance inference optimizer and runtime that accelerates deep learning models, especially on NVIDIA GPUs.
- **Triton Inference Server** is a scalable, production-ready platform for serving models. It handles queuing, batching, scheduling, and supports models from multiple frameworks (PyTorch, TensorFlow, ONNX, and more).

### What You’ll Learn

In this tutorial, you’ll walk through the full deployment of a Stable Diffusion v1.5 pipeline:

- Convert model components to ONNX
- Optimize them using TensorRT
- Deploy and serve them with Triton Inference Server

Along the way, you’ll learn how to:

- Convert and optimize models for GPU inference
- Use Triton’s ensemble models to link components
- Write Python-based orchestration with Triton’s Business Logic Scripting (BLS)
- Chain multiple ensembles into a multi-stage pipeline

### Disclaimer

This is not an advanced or exhaustive guide. The goal is to provide a working example that walks through the entire deployment process — from model conversion to production inference — in a clear, practical way.

Where relevant, we’ll point you to the official documentation and references for deeper dives.

## Installation

First, clone the repository and install the required Python packages:

```bash
git clone https://github.com/HugoLePicard/tutorial-triton-inference-server.git
cd tutorial-triton-inference-server
pip install -r requirements.txt
```

To make local imports easier across multiple scripts, we provide a setup.py. You can install the project in editable mode using:

```bash
pip install -e .
```

This allows you to import internal modules (e.g., from utils import X) without modifying your PYTHONPATH.

## Step 1 – Exporting Models to ONNX

### What is ONNX?

[ONNX](https://onnx.ai/) (Open Neural Network Exchange) is an open standard format for machine learning models. It allows models trained in one framework (like PyTorch) to be exported and run in others (such as TensorRT or ONNX Runtime), enabling flexible and high-performance inference across a wide range of tools.

### Components to Export in Stable Diffusion

In a typical diffusion model pipeline, the most computationally intensive components are:

- The **Text Encoder** (usually a CLIP transformer)
- The **UNet** (the denoising core of the model)
- The **VAE Decoder** (to reconstruct images from latents)
- The **VAE Encoder** (if using img2img or other conditioning)

These are the components we’ll convert to ONNX in this tutorial.

All export scripts are located in the `/optim` folder:

```bash
optim/
├── convert_text_encoder_to_onnx.py
├── convert_unet_to_onnx.py
├── convert_vae_decoder_to_onnx.py
├── convert_vae_encoder_to_onnx.py
```

Each script loads the original model from Hugging Face using the diffusers library, and exports it to ONNX format using torch.onnx.export. 

This makes the models portable and ready for hardware-optimized inference with tools like TensorRT.

### Dynamic Axes: Flexibility vs Performance

To keep this tutorial focused and practical, we’ll only cover **dynamic batch axes** when exporting models to ONNX. This is the most relevant case for our use case and has direct implications for the TensorRT conversion step later on.

If you're interested in exploring more advanced options, you can refer to the [PyTorch ONNX export documentation](https://pytorch.org/docs/stable/onnx.html#torch.onnx.export).

When exporting to ONNX, you can choose which input dimensions are **dynamic** — meaning they’re allowed to vary at inference time.

For example:

```python
dynamic_axes = {
    "sample": {0: "batch"},
    "encoder_hidden_states": {0: "batch"}
}
```

In this case, the batch dimension (0) is dynamic. This allows the model to accept variable batch sizes during inference, which is particularly useful in production environments where inputs may be batched dynamically.

However, this flexibility comes at a cost: TensorRT optimizations are most effective when input shapes are fixed. The more dynamic the shapes, the fewer assumptions TensorRT can make, and the smaller the performance gain.

You can choose to:

Use static axes (e.g., fixed batch size and resolution)
- Fastest inference, but rigid and less reusable.

Make some axes dynamic (e.g., batch size only)
- Good balance between performance and flexibility.

Allow all axes to be dynamic
- Maximum flexibility, but minimal TensorRT optimization.

In our export scripts, we also make sure to export the models in FP16 format, which is supported by both ONNX and TensorRT. Using FP16 reduces memory usage and increases inference speed on modern GPUs, especially during TensorRT optimization in the next step.

Lets run those scripts one by one with :

```bash
python ./optim/convert_text_encoder_to_onnx.py
python ./optim/convert_unet_to_onnx.py
python ./optim/convert_vae_encoder_to_onnx.py
python ./optim/convert_vae_decoder_to_onnx.py
```

Next step: we’ll optimize these ONNX models with TensorRT.

## Step 2 – Optimizing ONNX Models with TensorRT

Now that we’ve exported the core components of Stable Diffusion to ONNX, the next step is to optimize them for fast inference on GPU hardware.

For this, we’ll use **NVIDIA TensorRT**, a high-performance inference engine that compiles your ONNX graphs into GPU-specific executables called `.plan` files. These files are highly optimized and can drastically reduce latency during inference.

TensorRT supports multiple precision modes (e.g., FP16, INT8) and applies layer fusion, kernel auto-tuning, and memory optimizations to get the most out of your GPU.

### Why Use Docker?

TensorRT is tightly coupled to specific versions of CUDA, cuDNN, and PyTorch. Managing these dependencies manually can be time-consuming and error-prone.

To keep things clean, consistent, and reproducible, we’ll use **Docker**.

We provide two Dockerfiles:

- `./docker/pytorch/Dockerfile` — for TensorRT optimization
- `./docker/triton/Dockerfile` — for running the Triton Inference Server later

Both Dockerfiles use the same base image (`nvcr.io/nvidia/...:24.10-py3`) to ensure compatibility between the optimized `.plan` files and the runtime. Mismatched versions can lead to runtime errors or unsupported operations.

Also, installing TensorRT manually is notoriously painful, another good reason to containerize the process :)

### Building the PyTorch + TensorRT Optimization Image

We’ll start by building the PyTorch container, which includes all the tools needed to export and optimize ONNX models using TensorRT.

```bash
cd ./docker/pytorch
docker build -t pytorch:latest .
```

Once the image is built, we’ll use it to run the optimization scripts in the next step.

We’ll also build the Triton container now, which will be used later to serve the optimized models in production:

```bash
cd ../triton
docker build -t triton:latest .
```

## Step 3 – Organizing for Triton & Starting Optimization

Now that all models have been exported to ONNX and the Docker images are built, we can move on to optimizing the models with TensorRT and preparing them for deployment with Triton Inference Server.

### Triton Model Repository Structure

Triton follows a strict directory structure for serving models. All models must reside in a single root directory (in our case: `triton/models/`), and each model must follow this format:

```bash
triton/models/
├── unet/
│   ├── 1/
│   │   └── model.plan  # or model.onnx, depending on what you serve
│   └── config.pbtxt
├── text_encoder/
│   ├── 1/
│   │   └── model.plan
│   └── config.pbtxt
```

Each subdirectory under models/ corresponds to a separate model. Inside each:
- The 1/ folder contains the versioned model file — either model.onnx (for ONNXRuntime) or model.plan (for TensorRT).
- The config.pbtxt file defines the model’s input/output names, shapes, data types, batching behavior, and hardware placement (e.g., GPU).

You can technically serve ONNX models directly with Triton, but using TensorRT-optimized `.plan` files usually provides significant speedups, especially on NVIDIA hardware.

### Preparing the Triton Model Directory

Let’s now organize our exported ONNX models into the directory layout expected by Triton.

Run the following script:

```bash
python ./scripts/move_onnx_models_to_triton.py
```

This script copies each ONNX model from `data/models/` into `triton/models/[model_name]/1/model.onnx`, and also creates a `2/` folder for the optimized TensorRT engine (`model.plan`), which we’ll generate in the next step.

### Triton Model Versioning
Triton supports model versioning natively. Inside each model directory (e.g., `unet/`, `text_encoder/`), you can include multiple numbered subfolders such as `1/`, `2/`, etc.

Each subfolder represents a specific version of the model. Triton allows you to control which version is active via the `version_policy` field in `config.pbtxt`.

In our case, we’ll configure Triton to use the TensorRT-optimized version in `2/model.plan`.

Next, we’ll convert the exported ONNX files into optimized `.plan` files using TensorRT.

## Step 4 – Model Compilation with TensorRT

We’ll use a PyTorch-based Docker container (which includes Python and commonly used ML libraries) to compile our ONNX models into optimized TensorRT engines. This step is performed inside the container to avoid installing TensorRT and its dependencies locally.

### Running the PyTorch Container

Start by launching the container:

```bash
docker run -it --ipc host -v ./triton:/workspace/triton --gpus=all pytorch:latest bash
```

#### Explanation of flags:

- `-it`: interactive mode with a shell
- `--ipc host`: allows shared memory (needed for TensorRT)
- `-v ./triton:/workspace/triton`: mounts your local `./triton` directory into the container
- `--gpus=all`: enables GPU access
- `pytorch:latest`: or any other base image you use that has PyTorch + TensorRT installed

### Compiling the models

Once inside the container, navigate to the model directory:

```bash
cd /triton/models
```

You’ll now convert each ONNX model into a TensorRT engine (`.plan` file) optimized for inference.

#### Static Shapes vs Dynamic Shapes

#### Static shapes:
If your model was exported with fixed input dimensions, you can run a simple `trtexec` command without specifying shape ranges. This yields the best performance but requires fixed inputs.

#### Dynamic shapes:
If your model was exported with dynamic axes (e.g., variable batch sizes or spatial dimensions), you must specify minimum, optimal, and maximum input shapes using flags like `--minShapes`, `--optShapes`, and `--maxShapes`. This allows TensorRT to generate an engine that supports a range of input sizes.

### Basic Conversion (Static Shapes Only)

If your model uses fixed input shapes:

```bash
trtexec --onnx=./1/model.onnx \
        --saveEngine=./2/model.plan \
        --fp16
```

This compiles the ONNX model into a `.plan` file using FP16 precision, which reduces memory usage and improves inference speed. TensorRT automatically infers the input shape from the ONNX graph.

### Custom Shape Configuration (For Dynamic Axes)

If your model includes dynamic shapes, you'll need to specify shape ranges during compilation:

```bash
--minShapes=input_name:min_dims \
--optShapes=input_name:opt_dims \
--maxShapes=input_name:max_dims
```

Let’s now compile our models one by one.

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
- **TensorRT-optimized models are hardware-dependent**. The `.plan` files generated by TensorRT are highly optimized for the **specific GPU architecture** used during compilation.

  For example:
  - If you optimize the model on an **NVIDIA A100**, you **must** deploy it on another A100.
  - To deploy on a different GPU (e.g., T4, V100, or RTX 4090), you’ll need to **recompile the engine** using `trtexec` on that target GPU.

### Results

When you run `trtexec` on a model, TensorRT will not only optimize and export the `.plan` file, it will also benchmark the model's GPU inference performance. After a short warmup and a few runs, you'll see output like:

#### Example Benchmark Results on A100 40G

For the UNet:

```bash
[I] GPU Compute Time: min = 14.7405 ms, max = 14.7978 ms, mean = 14.7559 ms, ...
```

This means that a single forward pass through the UNet takes ~14.76 ms on average.

For the Text Encoder:

```bash
[I] GPU Compute Time: mean = 0.855457 ms
```

This is quite fast, less than 1 millisecond. It’s only run once at the beginning to encode the text prompt.

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

#### How to Interpret the Metrics

Each result gives:

- min: Fastest single run
- max: Slowest single run
- mean: Average time across all runs
- median: Middle value
- percentile(90/95/99%): Time under which that % of runs complete (e.g., 99% of runs were faster than 14.79 ms)

These numbers help you understand latency stability and identify performance bottlenecks.

### Total Inference Time for 50 Diffusion Steps (Theoretical)

In a typical Stable Diffusion pipeline:

- The UNet runs once per denoising step: 50 steps = 50 UNet calls
- The Text Encoder runs once at the beginning
- The VAE Decoder runs once at the end

So, in an ideal scenario (excluding overhead):

```bash
Total ≈ (50 × UNet time) + Text Encoder time + VAE Decoder time
      ≈ (50 × 14.76 ms) + 0.86 ms + 17.05 ms
      ≈ 738 ms + 0.86 ms + 17.05 ms
      ≈ ~756.9 ms
```

That’s approximately 1.32 frames per second, which is fast for image generation, thanks to TensorRT’s low-latency execution on high-performance GPUs like the A100.

For more details on TensorRT capabilities, optimization techniques, and deployment workflows, check out the official NVIDIA TensorRT documentation: https://docs.nvidia.com/deeplearning/tensorrt
You can also explore the source code, samples, and OSS contributions on GitHub: https://github.com/NVIDIA/TensorRT

## Step 5 – Triton Inference Server

Triton Inference Server is an inference serving platform developed by NVIDIA. It streamlines the deployment of machine learning models at scale, supporting features like dynamic batching, model versioning, concurrent execution, and multi-framework support (PyTorch, TensorFlow, ONNX, TensorRT, etc.).

### Model Config Explanation (`config.pbtxt`)

This document explains how to define a standard model configuration for use with Triton Inference Server, specifically for models optimized with TensorRT.

Each model directory in `triton/models/` must include a `config.pbtxt` file describing how the model should be served, including input/output definitions, supported versions, batching settings, and backend type.

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

### Field-by-Field Breakdown
#### `name`
The name of the model. This must match the folder name under `triton/models/`.  
Example: for `triton/models/unet/`, the name must be `"unet"`.

#### `backend`
Specifies the backend used to run the model.  
Use `"tensorrt"` for TensorRT-optimized models.

#### `platform`
Set to `"tensorrt_plan"` when using serialized TensorRT engines (`model.plan` files).

#### `max_batch_size`
Defines the maximum supported batch size.  
Set to `0` to disable batching. Use a positive integer if your model supports it.

#### `version_policy`
Specifies which version(s) of the model Triton should load.  
For example, to load only version 2:

```protobuf
version_policy: { specific { versions: [2] } }
```

This would load the file at `unet/2/model.plan`.

### Input and Output Declarations
All input and output fields must match:
- The names used during ONNX export
- The shapes used during TensorRT optimization
- The data types used during `trtexec` (e.g., `--fp16` ⇒ `TYPE_FP16`)

#### Inputs

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

#### Outputs

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

#### Instance Group

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

To control GPU selection and parallelism. For more on this, see https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/examples/jetson/concurrency_and_dynamic_batching/README.html 

### Notes
- Triton will strictly validate that incoming inference requests match the declared input/output names, types, and shapes.
- If anything is misaligned, Triton will raise an error at runtime and refuse to execute the model.
- If your ONNX export used `--dynamic_axes`, you must reflect that with dynamic dimensions (e.g. `-1`).
- If your TensorRT optimization used `--fp16`, you must set data_type: `TYPE_FP16`.
- Use `trtexec` logs to inspect actual tensor names and shapes to verify alignment.

## Step 7 - Launching the Triton Inference Server

Once all your models are in place under the `triton/models/` directory (with the correct `config.pbtxt` and serialized `model.plan` files), you can launch the Triton Inference Server using Docker:

```bash
docker run --gpus all --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  --shm-size=10g --ipc=host --pid=host \
  -v ./triton/models/:/models \
  triton:latest \
  tritonserver --model-repository=/models
```

**Explanation of flags:**

- `--gpus all`: Enables access to all available GPUs
- `--shm-size=10g`: Increases shared memory to avoid out-of-memory errors for large tensors
- `-v ./triton/models/:/models`: Mounts your local model repository into the container
- `tritonserver --model-repository=/models`: Launches the Triton server using the mounted model directory

## Step 8 - Querying Individual Models in Python

Once the server is running, you can send inference requests using the Triton Python client.

Here’s a basic example for querying a single model (e.g., `vae_decoder`, `unet`, or `text_encoder`):

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

### Key Components

- `InferInput("sample", shape, dtype)`: Must match the name and shape declared in `config.pbtxt`.
- `.set_data_from_numpy(...)`: Loads the NumPy tensor into the request.
- `client.infer(...)`: Sends the inference request to a specific model on the server.
- `.as_numpy(...)`: Extracts the model's output as a NumPy array.

Repeat the same logic for other models:

- **Text encoder:** Input → `input_ids`, Output → `last_hidden_state`
- **UNet:** Inputs → `sample`, `timestep`, `encoder_hidden_states`
- **VAE decoder:** Input → `sample`, Output → `image`

## Step 9 – Stitching the Pipeline Together with BLS

Now that each component (text encoder, UNet, VAE encoder/decoder) is individually deployable and callable, we’re ready to link them into a single pipeline using **BLS (Business Logic Scripting)**.

BLS allows you to define a custom Python backend model that orchestrates multiple sub-model calls directly inside Triton, no need for the client to manage them manually.

We’ll define a Python model under `triton/models/pipeline_text_to_image/` and script the full inference logic using Triton's Python backend API.

---

### A Minimal Stable Diffusion Pipeline (Client-Side)

Before implementing the full pipeline using BLS (which can be harder to iterate on), we recommend starting with a simple **client-side pipeline**. This makes testing and debugging much easier.

#### Why?

- Faster iteration and easier debugging
- Full control over the pipeline logic in Python
- Ensures all inputs and outputs are aligned
- Makes the later transition to BLS safer and more reliable

#### What This Looks Like

Rather than building a BLS backend immediately, we call each model individually from the client:

1. **Text Encoder**  
   → Call the `text_encoder` model to generate `encoder_hidden_states`

2. **UNet Loop**  
   → For each denoising timestep, call the `unet` with the latent + conditioning

3. **VAE Decoder**  
   → Call the `vae_decoder` once at the end to decode the latent into an image

This structure mirrors Hugging Face’s `StableDiffusionPipeline`, but each step is routed through Triton.

#### Minimal Implementation

We provide a minimal version of the pipeline that interacts with Triton, inspired by 🤗 `diffusers`, but stripped to the essentials.

Implementation:

```bash
pipelines/ensemble.py
```

To test it end-to-end:

```bash
./scripts/run_ensemble_pipeline.py
```

This script will:

- Load a prompt
- Encode it with the `text_encoder`
- Run the UNet loop over multiple timesteps
- Decode the final latent using `vae_decoder`
- Output the generated image

Once this works, you're ready to port the logic into a server-side pipeline using BLS.

### Triton BLS (Business Logic Scripting) with Python Backend

Once your client-side pipeline works, you can move the logic server-side using Triton’s Python backend. This allows you to define the full Stable Diffusion pipeline (text → latent → image) as a single model.

#### 1. `config.pbtxt` for the BLS Model

To enable a Python backend, you only need a minimal config:

```protobuf
name: "pipeline_text_to_image"
backend: "python"
max_batch_size: 0
```

Triton expects the Python script inside:

```bash
triton/models/pipeline_text_to_image/
├── 1/
│   └── model.py
└── config.pbtxt
```

#### 2. Basics of model.py
Triton requires you to define a `TritonPythonModel` class with two methods:

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

Each request is an instance of `InferenceRequest`, and must return a list of `InferenceResponse` objects.

#### 3. Using `pb_utils`

Triton provides a helper module, `triton_python_backend_utils` (imported as `pb_utils`), to:

- Parse input tensors from requests
- Call other Triton models (like `unet`, `text_encoder`, etc.)
- Build and return output tensors

Example: extracting parameters from the request

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

#### 4. DLPack and Why We Use It

Triton supports DLPack for **zero-copy memory sharing** between PyTorch tensors and Triton’s internal format.

Sending tensors to submodels:

```python
inference_request = pb_utils.InferenceRequest(
    model_name = "unet",
    inputs = [
        pb_utils.Tensor.from_dlpack("sample"               , torch.to_dlpack(latent_model_input)),
        pb_utils.Tensor.from_dlpack("timestep"             , torch.to_dlpack(timestep)),
        pb_utils.Tensor.from_dlpack("encoder_hidden_states", torch.to_dlpack(prompt_embeds)),
    ],
    requested_output_names = ["outputs"]
)
```

Receiving output from a submodel:

```python
inference_response = inference_request.exec()
if inference_response.has_error():
    raise pb_utils.TritonModelException(inference_response.error().message())

output_tensor = pb_utils.get_output_tensor_by_name(inference_response, "outputs")
output_tensor = torch.from_dlpack(output_tensor.to_dlpack())
```

Using DLPack avoids unnecessary CPU transfers and improves performance when chaining models.

### Running the Final BLS Pipeline

Once the `pipeline_text_to_image` model is implemented with `model.py`, your client no longer needs to handle each step of the pipeline. Triton will run the full inference graph server-side.

You can find the minimal client for calling this final BLS pipeline here:

```bash
pipelines/bls.py
```

To test it end-to-end, simply run:

```bash
scripts/run_bls_pipeline.py
```

This script sends a request to `pipeline_text_to_image` with high-level inputs like `prompt`, `num_inference_steps`, and `guidance_scale`. Triton takes care of the rest:

- Encodes the prompt
- Runs the UNet loop internally
- Decodes the final image

This marks the transition from a **client-driven pipeline** to a **fully self-contained server-side inference system**.

## Step 10 - Benchmarking the Three Pipelines

You can run the following script to benchmark and compare the three pipelines:

```bash
scripts/benchmark_pipelines.py
```

### This script benchmarks:

- **Hugging Face Diffusers** (pure PyTorch)
- **Triton Ensemble Pipeline** (client-side orchestration)
- **Triton BLS Pipeline** (server-side orchestration via Python backend)

### Results (for 20 runs)

```less
Hugging Face Diffusers  Average Time: 2.35 s ± 0.20 s  
Triton Ensemble         Average Time: 0.97 s ± 0.01 s  
Triton BLS              Average Time: 0.91 s ± 0.01 s 
```

These results demonstrate the performance improvements gained by offloading computation to Triton and reducing client-side logic. The **Triton BLS** version is the fastest, with everything handled server-side, including orchestration, latent processing, and memory management.

### Why the BLS Pipeline Is Fastest

By eliminating some Python overhead and leveraging optimized TensorRT engines, inference time drops by more than half compared to the Hugging Face implementation.

However, some latency remains due to:

- Python execution inside `model.py` (control flow, tensor handling)
- gRPC overhead (sending the prompt and receiving the final image)

### Theoretical Minimum

As estimated earlier:

- 50 UNet steps × 14.76 ms = 738 ms
- + Text encoder (~0.86 ms)
- + VAE decoder (~17.05 ms)

Total theoretical compute time: **~756.9 ms**

The Triton BLS pipeline comes close to this limit, with most of the remaining gap explained by I/O and the overhead of lightweight Python glue logic.
