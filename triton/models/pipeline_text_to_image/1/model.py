import inspect
import torch
import numpy 
import triton_python_backend_utils as pb_utils

from transformers                import CLIPTokenizer
from typing                      import List, Optional, Union
from diffusers.schedulers        import PNDMScheduler
from diffusers.utils.torch_utils import randn_tensor
from tqdm.auto                   import tqdm

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int]                      = None,
    device             : Optional[Union[str, torch.device]] = None,
    timesteps          : Optional[List[int]]                = None,
    sigmas             : Optional[List[float]]              = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

class TritonPythonModel:
    
    def initialize(self, args):
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.scheduler = PNDMScheduler(
            num_train_timesteps = 1000,
            beta_start          = 0.00085,
            beta_end            = 0.012,
            beta_schedule       = "scaled_linear",
            trained_betas       = None,
            skip_prk_steps      = True,
            set_alpha_to_one    = False,
            prediction_type     = "epsilon",
            timestep_spacing    = "leading",
            steps_offset        = 1,
        )
        self.vae_scale_factor = 8
        self.sigmas           = None
        self.timesteps        = None
        self.height           = 512
        self.width            = 512

    def encode_prompt(
        self,
        prompt,
        device,
        do_classifier_free_guidance,
        negative_prompt = None,
    ):
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        text_inputs = self.tokenizer(
            prompt,
            padding        = "max_length",
            max_length     = self.tokenizer.model_max_length,
            truncation     = True,
            return_tensors = "pt",
        )
        text_input_ids = text_inputs.input_ids

        inference_request = pb_utils.InferenceRequest(
            model_name = "text_encoder",
            inputs = [
                pb_utils.Tensor.from_dlpack("tokens", torch.to_dlpack(text_input_ids.to(torch.int64))),
            ],
            requested_output_names = ["last_hidden_state"]
        )

        inference_response = inference_request.exec()
        if inference_response.has_error():
            raise pb_utils.TritonModelException(
                inference_response.error().message()
            )

        last_hidden_state_tensor = pb_utils.get_output_tensor_by_name(inference_response, "last_hidden_state")
        prompt_embeds = torch.from_dlpack(last_hidden_state_tensor.to_dlpack())
        prompt_embeds = prompt_embeds.to(dtype=torch.float16, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds        = prompt_embeds.repeat(1, 1, 1)
        prompt_embeds        = prompt_embeds.view(bs_embed * 1, seq_len, -1)

        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding        = "max_length",
                max_length     = max_length,
                truncation     = True,
                return_tensors = "pt",
            )

            inference_request = pb_utils.InferenceRequest(
                model_name = "text_encoder",
                inputs = [
                    pb_utils.Tensor.from_dlpack("tokens", torch.to_dlpack(uncond_input.input_ids.to(torch.int64))),
                ],
                requested_output_names = ["last_hidden_state"]
            )

            inference_response = inference_request.exec()
            if inference_response.has_error():
                raise pb_utils.TritonModelException(
                    inference_response.error().message()
                )

            last_hidden_state_tensor = pb_utils.get_output_tensor_by_name(inference_response, "last_hidden_state")
            negative_prompt_embeds   = torch.from_dlpack(last_hidden_state_tensor.to_dlpack())
            negative_prompt_embeds   = negative_prompt_embeds.to(dtype=torch.float16, device=device)

        if do_classifier_free_guidance:
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=torch.float16, device=device)
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, 1, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * 1, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator):
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width)  // self.vae_scale_factor,
        )
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    def execute(self, requests):
        responses = []

        for request in requests:
            with torch.no_grad():
                if not (prompt_tensor := pb_utils.get_input_tensor_by_name(request, "prompt")):
                    raise ValueError("prompt is required")
                prompt: Union[str, List[str]] = [
                    t.decode("utf-8") for t in prompt_tensor.as_numpy().tolist()
                ]

                guidance_scale: float = 7.5
                if (guidance_scale_tensor := pb_utils.get_input_tensor_by_name(request, "guidance_scale")):
                    guidance_scale = guidance_scale_tensor.as_numpy().tolist()[0]

                num_inference_steps: int = 50
                if (steps_tensor := pb_utils.get_input_tensor_by_name(request, "num_inference_steps")):
                    num_inference_steps = steps_tensor.as_numpy().tolist()[0]

                seed: int = 0
                if (seed_tensor := pb_utils.get_input_tensor_by_name(request, "seed")):
                    seed = seed_tensor.as_numpy().tolist()[0]

                negative_prompt: Union[str, List[str]] = None
                if (negative_prompt_tensor := pb_utils.get_input_tensor_by_name(request, "negative_prompt")):
                    negative_prompt = [
                        t.decode("utf-8") for t in negative_prompt_tensor.as_numpy().tolist()
                    ]

                generator = torch.Generator("cuda").manual_seed(seed)

                self._guidance_scale = guidance_scale

                if prompt is not None and isinstance(prompt, str):
                    batch_size = 1
                elif prompt is not None and isinstance(prompt, list):
                    batch_size = len(prompt)
                else:
                    batch_size = prompt_embeds.shape[0]

                device = torch.device("cuda")

                prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                    prompt,
                    device,
                    self.do_classifier_free_guidance,
                    negative_prompt,
                )

                if self.do_classifier_free_guidance:
                    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

                timesteps, num_inference_steps = retrieve_timesteps(
                    self.scheduler, num_inference_steps, device, self.timesteps, self.sigmas
                )

                num_channels_latents = 4
                latents = self.prepare_latents(
                    batch_size,
                    num_channels_latents,
                    self.height,
                    self.width,
                    prompt_embeds.dtype,
                    device,
                    generator,
                )

                num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
                self._num_timesteps = len(timesteps)
                with tqdm(total=num_inference_steps) as progress_bar:
                    for i, t in enumerate(timesteps):
                        latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                        timestep           = torch.tensor([t], device = device, dtype=torch.float16)

                        inference_request = pb_utils.InferenceRequest(
                            model_name = "unet",
                            inputs = [
                                pb_utils.Tensor.from_dlpack("sample"               , torch.to_dlpack(latent_model_input)),
                                pb_utils.Tensor.from_dlpack("timestep"             , torch.to_dlpack(timestep)),
                                pb_utils.Tensor.from_dlpack("encoder_hidden_states", torch.to_dlpack(prompt_embeds)),
                            ],
                            requested_output_names = ["outputs"]
                        )
                        inference_response = inference_request.exec()
                        if inference_response.has_error():
                            raise pb_utils.TritonModelException(
                                inference_response.error().message()
                            )

                        noise_pred = pb_utils.get_output_tensor_by_name(inference_response, "outputs")
                        noise_pred = torch.from_dlpack(noise_pred.to_dlpack())

                        if self.do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                        latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                        if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                            progress_bar.update()

                latents = latents / 0.18215

                inference_request = pb_utils.InferenceRequest(
                    model_name = "vae_decoder",
                    inputs = [pb_utils.Tensor.from_dlpack("sample", torch.to_dlpack(latents.contiguous()))],
                    requested_output_names = ["image"]
                )
                inference_response = inference_request.exec()
                if inference_response.has_error():
                    raise pb_utils.TritonModelException(
                        inference_response.error().message()
                    )
                image = pb_utils.get_output_tensor_by_name(inference_response, "image")
                image = torch.from_dlpack(image.to_dlpack()).to(torch.float16)
                image = (image / 2 + 0.5).clamp(0, 1) * 255
                image = image.cpu().permute(0, 2, 3, 1).float().numpy().astype(numpy.uint8)

                tensor_output = [pb_utils.Tensor("image", image)]
                responses.append(pb_utils.InferenceResponse(tensor_output))

        return responses