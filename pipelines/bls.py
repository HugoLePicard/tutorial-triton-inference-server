import numpy
import tritonclient.grpc as grpcclient

class StableDiffusionPipelineBLS:
    def __init__(self, url="localhost:8001"):
        self.client = grpcclient.InferenceServerClient(url=url)

    def __call__(
        self,
        prompt,
        guidance_scale      = None,
        num_inference_steps = None,
        seed                = None,
        negative_prompt     = None,
    ):
        inputs = []

        prompt_numpy = numpy.array([prompt], dtype=object)
        prompt_input = grpcclient.InferInput("prompt", prompt_numpy.shape, "BYTES")
        prompt_input.set_data_from_numpy(prompt_numpy)
        inputs.append(prompt_input)

        if guidance_scale is not None:
            guidance_scale_numpy = numpy.array([guidance_scale], dtype=numpy.float32)
            guidance_scale_input = grpcclient.InferInput("guidance_scale", guidance_scale_numpy.shape, "FP32")
            guidance_scale_input.set_data_from_numpy(guidance_scale_numpy)
            inputs.append(guidance_scale_input)

        if num_inference_steps is not None:
            steps_numpy = numpy.array([num_inference_steps], dtype=numpy.int32)
            steps_input = grpcclient.InferInput("num_inference_steps", steps_numpy.shape, "INT32")
            steps_input.set_data_from_numpy(steps_numpy)
            inputs.append(steps_input)

        if seed is not None:
            seed_numpy = numpy.array([seed], dtype=numpy.int32)
            seed_input = grpcclient.InferInput("seed", seed_numpy.shape, "INT32")
            seed_input.set_data_from_numpy(seed_numpy)
            inputs.append(seed_input)

        if negative_prompt is not None:
            neg_prompt_numpy = numpy.array([negative_prompt], dtype=object)
            neg_prompt_input = grpcclient.InferInput("negative_prompt", neg_prompt_numpy.shape, "BYTES")
            neg_prompt_input.set_data_from_numpy(neg_prompt_numpy)
            inputs.append(neg_prompt_input)

        result = self.client.infer(
            model_name="pipeline_text_to_image",
            inputs=inputs,
            outputs=[grpcclient.InferRequestedOutput("image")]
        )

        image = result.as_numpy("image")[0]
        return image
