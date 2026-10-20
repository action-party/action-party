from typing import Tuple
from einops import rearrange
from torch import nn
import torch.distributed as dist
import torch

from pipeline import SelfForcingTrainingPipeline
from utils.loss import get_denoising_loss
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


class BaseModel(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self._initialize_models(args, device)

        self.device = device
        self.args = args
        self.dtype = torch.bfloat16 if args.mixed_precision else torch.float32
        if hasattr(args, "denoising_step_list"):
            self.denoising_step_list = torch.tensor(args.denoising_step_list, dtype=torch.long)
            if args.warp_denoising_step:
                timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
                self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

    def _initialize_models(self, args, device):
        self.real_model_name = getattr(args, "real_name", "Wan2.1-T2V-1.3B")
        self.fake_model_name = getattr(args, "fake_name", "Wan2.1-T2V-1.3B")

        self.generator = WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=True)
        self.generator.model.requires_grad_(True)

        # Only load real_score and fake_score if needed (for DMD/SiD/GAN, not for basic diffusion)
        if getattr(args, "load_score_models", True):
            self.real_score = WanDiffusionWrapper(model_name=self.real_model_name, is_causal=False)
            self.real_score.model.requires_grad_(False)

            self.fake_score = WanDiffusionWrapper(model_name=self.fake_model_name, is_causal=False)
            self.fake_score.model.requires_grad_(True)
        else:
            self.real_score = None
            self.fake_score = None

        # Only load text encoder if needed
        if getattr(args, "load_text_encoder", True):
            self.text_encoder = WanTextEncoder()
            self.text_encoder.requires_grad_(False)
        else:
            self.text_encoder = None

        # Only load VAE if needed (for raw video encoding or visualization)
        if getattr(args, "load_vae", True):
            self.vae = WanVAEWrapper()
            self.vae.requires_grad_(False)
        else:
            self.vae = None

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    def _get_timestep(
            self,
            min_timestep: int,
            max_timestep: int,
            batch_size: int,
            num_frame: int,
            num_frame_per_block: int,
            uniform_timestep: bool = False
    ) -> torch.Tensor:
        """
        Randomly generate a timestep tensor based on the generator's task type. It uniformly samples a timestep
        from the range [min_timestep, max_timestep], and returns a tensor of shape [batch_size, num_frame].
        - If uniform_timestep, it will use the same timestep for all frames.
        - If not uniform_timestep, it will use a different timestep for each block.
        """
        if uniform_timestep:
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, 1],
                device=self.device,
                dtype=torch.long
            ).repeat(1, num_frame)
            return timestep
        else:
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, num_frame],
                device=self.device,
                dtype=torch.long
            )
            # make the noise level the same within every block
            if self.independent_first_frame:
                # the first frame is always kept the same
                timestep_from_second = timestep[:, 1:]
                # For i2v, the remaining frames after the first should use (num_frame - 1) as block size
                effective_block_size = min(num_frame_per_block, timestep_from_second.shape[1])
                timestep_from_second = timestep_from_second.reshape(
                    timestep_from_second.shape[0], -1, effective_block_size)
                timestep_from_second[:, :, 1:] = timestep_from_second[:, :, 0:1]
                timestep_from_second = timestep_from_second.reshape(
                    timestep_from_second.shape[0], -1)
                timestep = torch.cat([timestep[:, 0:1], timestep_from_second], dim=1)
            else:
                timestep = timestep.reshape(
                    timestep.shape[0], -1, num_frame_per_block)
                timestep[:, :, 1:] = timestep[:, :, 0:1]
                timestep = timestep.reshape(timestep.shape[0], -1)
            return timestep


class SelfForcingModel(BaseModel):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.denoising_loss_func = get_denoising_loss(args.denoising_loss_type)()

    def _run_generator(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        initial_latent: torch.tensor = None,
        clean_latent: torch.tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optionally simulate the generator's input from noise using backward simulation
        and then run the generator for one-step.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - initial_latent: a tensor containing the initial latents [B, 1, C, H, W] for i2v mode.
            - clean_latent: a tensor containing the clean ground truth latents [B, F, C, H, W] for teacher forcing.
        Output:
            - pred_image: a tensor with shape [B, F, C, H, W].
            - gradient_mask: a tensor or None
            - denoised_timestep_from: an integer
            - denoised_timestep_to: an integer
        """
        # Step 1: Sample noise and backward simulate the generator's input
        assert getattr(self.args, "backward_simulation", True), "Backward simulation needs to be enabled"
        if initial_latent is not None:
            conditional_dict["initial_latent"] = initial_latent
        if self.args.i2v:
            noise_shape = [image_or_video_shape[0], image_or_video_shape[1] - 1, *image_or_video_shape[2:]]
        else:
            noise_shape = image_or_video_shape.copy()

        # During training, the number of generated frames should be uniformly sampled from
        # [21, self.num_training_frames], but still being a multiple of self.num_frame_per_block
        min_num_frames = 20 if self.args.independent_first_frame else 21
        max_num_frames = self.num_training_frames - 1 if self.args.independent_first_frame else self.num_training_frames
        assert max_num_frames % self.num_frame_per_block == 0
        assert min_num_frames % self.num_frame_per_block == 0
        max_num_blocks = max_num_frames // self.num_frame_per_block
        min_num_blocks = min_num_frames // self.num_frame_per_block
        num_generated_blocks = torch.randint(min_num_blocks, max_num_blocks + 1, (1,), device=self.device)
        dist.broadcast(num_generated_blocks, src=0)
        num_generated_blocks = num_generated_blocks.item()
        num_generated_frames = num_generated_blocks * self.num_frame_per_block
        if self.args.independent_first_frame and initial_latent is None:
            num_generated_frames += 1
            min_num_frames += 1
        # Sync num_generated_frames across all processes
        noise_shape[1] = num_generated_frames

        pred_image_or_video, denoised_timestep_from, denoised_timestep_to = self._consistency_backward_simulation(
            noise=torch.randn(noise_shape,
                              device=self.device, dtype=self.dtype),
            clean_latent=clean_latent,  # Pass ground truth for teacher forcing
            **conditional_dict,
        )
        # Slice last num_training_frames
        # If we generated more than num_training_frames and have a VAE, re-encode the first frame
        # to get a clean latent for the first frame (needed for proper loss computation)
        if pred_image_or_video.shape[1] > self.num_training_frames and self.vae is not None:
            with torch.no_grad():
                # Reencode to get image latent
                latent_to_decode = pred_image_or_video[:, :-(self.num_training_frames - 1), ...]
                # Deccode to video
                pixels = self.vae.decode_to_pixel(latent_to_decode)
                frame = pixels[:, -1:, ...].to(self.dtype)
                frame = rearrange(frame, "b t c h w -> b c t h w")
                # Encode frame to get image latent
                image_latent = self.vae.encode_to_latent(frame).to(self.dtype)
            pred_image_or_video_last_N = torch.cat([image_latent, pred_image_or_video[:, -(self.num_training_frames - 1):, ...]], dim=1)
        elif pred_image_or_video.shape[1] > self.num_training_frames:
            # If no VAE available, just take the last num_training_frames
            pred_image_or_video_last_N = pred_image_or_video[:, -self.num_training_frames:]
        else:
            pred_image_or_video_last_N = pred_image_or_video

        if num_generated_frames != min_num_frames:
            # Currently, we do not use gradient for the first chunk, since it contains image latents
            gradient_mask = torch.ones_like(pred_image_or_video_last_N, dtype=torch.bool)
            if self.args.independent_first_frame:
                gradient_mask[:, :1] = False
            else:
                gradient_mask[:, :self.num_frame_per_block] = False
        else:
            gradient_mask = None

        pred_image_or_video_last_N = pred_image_or_video_last_N.to(self.dtype)
        return pred_image_or_video_last_N, gradient_mask, denoised_timestep_from, denoised_timestep_to

    def _consistency_backward_simulation(
        self,
        noise: torch.Tensor,
        clean_latent: torch.Tensor = None,
        **conditional_dict: dict
    ) -> torch.Tensor:
        """
        Simulate the generator's input from noise to avoid training/inference mismatch.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Here we use the consistency sampler (https://arxiv.org/abs/2303.01469)
        Input:
            - noise: a tensor sampled from N(0, 1) with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - clean_latent: optional ground truth frames [B, F, C, H, W] for teacher forcing during training.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
        Output:
            - output: a tensor with shape [B, T, F, C, H, W].
            T is the total number of timesteps. output[0] is a pure noise and output[i] and i>0
            represents the x0 prediction at each timestep.
        """
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()

        return self.inference_pipeline.inference_with_trajectory(
            noise=noise, clean_latent=clean_latent, **conditional_dict
        )

    def _initialize_inference_pipeline(self):
        """
        Lazy initialize the inference pipeline during the first backward simulation run.
        Here we encapsulate the inference code with a model-dependent outside function.
        We pass our FSDP-wrapped modules into the pipeline to save memory.
        """
        self.inference_pipeline = SelfForcingTrainingPipeline(
            denoising_step_list=self.denoising_step_list,
            scheduler=self.scheduler,
            generator=self.generator,
            num_frame_per_block=self.num_frame_per_block,
            independent_first_frame=self.args.independent_first_frame,
            same_step_across_blocks=self.args.same_step_across_blocks,
            last_step_only=self.args.last_step_only,
            num_max_frames=self.num_training_frames,
            context_noise=self.args.context_noise,
            i2v=self.args.i2v
        )
