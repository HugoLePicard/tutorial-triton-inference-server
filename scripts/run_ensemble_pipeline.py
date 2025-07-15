from pipelines.ensemble import StableDiffusionPipelineEnsemble

if __name__ == "__main__":
    pipe = StableDiffusionPipelineEnsemble()
    image = pipe(prompt="a robot", seed=42)
    print("Image shape:", image.shape)