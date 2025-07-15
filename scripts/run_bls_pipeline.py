from pipelines.bls import StableDiffusionPipelineBLS

if __name__ == "__main__":
    pipe = StableDiffusionPipelineBLS()
    image = pipe(prompt="a robot", seed=42)
    print("Image shape:", image.shape)