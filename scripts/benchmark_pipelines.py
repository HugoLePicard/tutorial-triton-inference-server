import inspect
import torch
import numpy
import time

from diffusers          import StableDiffusionPipeline
from pipelines.ensemble import StableDiffusionPipelineEnsemble
from pipelines.bls      import StableDiffusionPipelineBLS

from utils import *

def benchmark_pipeline(pipeline, label, prompt="a robot", generator=None, nb_warmup_runs=5, nb_benchmark_runs=20):
    print(f"Running {label} benchmark: {nb_warmup_runs} warmup(s), {nb_benchmark_runs} timed run(s)")
    
    def _run_once():
        if "generator" in inspect.signature(pipeline.__call__).parameters:
            return pipeline(prompt, generator=generator)
        else:
            return pipeline(prompt)
    
    for _ in range(nb_warmup_runs):
        _run_once()

    durations = []
    for _ in range(nb_benchmark_runs):
        start = time.time()
        _run_once()
        durations.append(time.time() - start)

    time_average = numpy.mean(durations)
    time_std     = numpy.std(durations)
    print(f"{label} Average Time: {time_average:.2f} s ± {time_std:.2f} s over {nb_benchmark_runs} runs \n")
    return time_average

if __name__ == "__main__":
    prompt    = "a robot"
    generator = torch.Generator("cuda").manual_seed(0)

    print("Loading diffusers pipeline...")
    pipe_diffusers = StableDiffusionPipeline.from_pretrained(
        DIFFUSION_MODEL_ID,
        torch_dtype=torch.float16
    ).to("cuda")

    print("Loading Triton Ensemble pipeline...")
    pipe_triton = StableDiffusionPipelineEnsemble()

    print("Loading Triton BLS pipeline...")
    pipe_bls = StableDiffusionPipelineBLS()

    benchmark_pipeline(pipe_diffusers, "HuggingFace Diffusers", prompt, generator)
    benchmark_pipeline(pipe_triton   , "Triton Ensemble"      , prompt, generator)
    benchmark_pipeline(pipe_bls      , "Triton BLS"           , prompt)