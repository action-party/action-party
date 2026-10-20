"""Bidirectional diffusion inference (``sample_solver: unipc`` / ``dpm``).

This is the validation / EMA-preview pipeline used by the diffusion trainer
in :mod:`trainer.diffusion`. The coord-token branch in ``inference_sample``
mirrors the forward path in :class:`utils.wan_wrapper.WanDiffusionWrapper`.
"""
from tqdm import tqdm
from typing import List
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


class BidirectionalDiffusionInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        super().__init__()
        # Step 1: Initialize all models
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=False) if generator is None else generator
        # Keep text_encoder as-is (can be None for unconditional models)
        self.text_encoder = text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        # Step 2: Initialize scheduler
        self.num_train_timesteps = args.num_train_timestep
        self.sampling_steps = getattr(args, 'sampling_steps', 50)  # Configurable inference steps
        self.sample_solver = getattr(args, 'sample_solver', 'unipc')  # Also make solver configurable
        # Use timestep_shift from config (should match training!)
        self.shift = getattr(args, 'timestep_shift', 8.0)
        print(f"Inference config: sampling_steps={self.sampling_steps}, solver={self.sample_solver}, shift={self.shift}")
        
        self.device = device

        # I2V latent cross-attention setup (copy patch_embedding to avoid FSDP overhead)
        self.i2v_cross_attention_latent = getattr(args, "i2v_cross_attention_latent", False)
        self.latent_patch_embedding = None
        if self.i2v_cross_attention_latent:
            self._initialize_latent_patch_embedding()
            print(f"✓ Initialized latent patch embedding (no FSDP overhead)")

        self.args = args
    
    def _initialize_latent_patch_embedding(self):
        """Copy patch_embedding weights to avoid FSDP overhead during inference."""
        if isinstance(self.generator, FSDP):
            with FSDP.summon_full_params(self.generator, writeback=False, recurse=True):
                patch_emb = self.generator.model.patch_embedding
                self.latent_patch_embedding = torch.nn.Conv3d(
                    in_channels=patch_emb.in_channels,
                    out_channels=patch_emb.out_channels,
                    kernel_size=patch_emb.kernel_size,
                    stride=patch_emb.stride,
                    padding=patch_emb.padding,
                    bias=patch_emb.bias is not None
                ).to(device=self.device, dtype=patch_emb.weight.dtype)
                
                self.latent_patch_embedding.weight.data.copy_(patch_emb.weight.data)
                if patch_emb.bias is not None:
                    self.latent_patch_embedding.bias.data.copy_(patch_emb.bias.data)
        else:
            patch_emb = self.generator.model.patch_embedding
            self.latent_patch_embedding = torch.nn.Conv3d(
                in_channels=patch_emb.in_channels,
                out_channels=patch_emb.out_channels,
                kernel_size=patch_emb.kernel_size,
                stride=patch_emb.stride,
                padding=patch_emb.padding,
                bias=patch_emb.bias is not None
            ).to(device=self.device, dtype=patch_emb.weight.dtype)
            
            self.latent_patch_embedding.weight.data.copy_(patch_emb.weight.data)
            if patch_emb.bias is not None:
                self.latent_patch_embedding.bias.data.copy_(patch_emb.bias.data)
        
        self.latent_patch_embedding.eval()
        self.latent_patch_embedding.requires_grad_(False)

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        return_latents=False,
        initial_frame=None
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_frame (torch.Tensor, optional): Fixed first frame for i2v conditioning.
                Shape: (batch_size, 1, num_channels, height, width).
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_frames, num_channels, height, width). It is normalized to be in the range [0, 1].
        """

        # Handle text conditioning
        if self.text_encoder is not None:
            conditional_dict = self.text_encoder(
                text_prompts=text_prompts
            )
        else:
            # Unconditional: create zero embeddings (same as training)
            batch_size = len(text_prompts)
            text_dim = self.generator.model.text_dim
            text_len = self.generator.model.text_len
            dummy_prompt_embeds = torch.zeros(
                [batch_size, text_len, text_dim],
                device=noise.device,
                dtype=noise.dtype
            )
            conditional_dict = {
                "prompt_embeds": dummy_prompt_embeds
            }
        
        # Only compute unconditional if guidance_scale != 1.0 (CFG is actually being used)
        use_cfg = abs(self.args.guidance_scale - 1.0) > 1e-5
        if use_cfg:
            if self.text_encoder is not None:
                unconditional_dict = self.text_encoder(
                    text_prompts=[self.args.negative_prompt] * len(text_prompts)
                )
            else:
                # Reuse the same zero embeddings for unconditional
                unconditional_dict = conditional_dict

        # Add latent cross-attention if enabled
        if self.i2v_cross_attention_latent and initial_frame is not None:
            # Use our separate patch_embedding copy (no FSDP overhead)
            # initial_frame: [B, 1, C, H, W] -> need [B, C, 1, H, W] for Conv3d
            first_frame = initial_frame.permute(0, 2, 1, 3, 4)  # [B, C, 1, H, W]
            
            # Apply patch embedding (using our separate copy)
            # patch_embedding is Conv3d with kernel=(1,2,2), stride=(1,2,2)
            latent_tokens = self.latent_patch_embedding(first_frame)  # [B, model_dim, 1, H', W']
            
            # Flatten to tokens: [B, model_dim, 1, H', W'] -> [B, num_patches, model_dim]
            latent_tokens = latent_tokens.squeeze(2)  # [B, model_dim, H', W']
            latent_tokens = latent_tokens.flatten(2).transpose(1, 2)  # [B, num_patches, model_dim]
            
            # Add to conditional_dict by setting 'text_emb' key
            # This matches training: latent tokens are in model dimension (1536-dim)
            # The wrapper will prefer 'text_emb' over 'prompt_embeds'
            if 'text_emb' in conditional_dict:
                # text_emb is already in model dimension, safe to concatenate
                conditional_dict['text_emb'] = torch.cat([latent_tokens, conditional_dict['text_emb']], dim=1)
            else:
                # Just use latent tokens as conditioning (unconditional generation)
                conditional_dict['text_emb'] = latent_tokens
            
            # Also add to unconditional_dict if using CFG
            if use_cfg:
                if 'text_emb' in unconditional_dict:
                    unconditional_dict['text_emb'] = torch.cat([latent_tokens, unconditional_dict['text_emb']], dim=1)
                else:
                    unconditional_dict['text_emb'] = latent_tokens

        latents = noise
        
        # For noised conditioning: store the noise for the first frame to use consistently
        i2v_noised_conditioning = getattr(self.args, 'i2v_noised_conditioning', False)
        first_frame_noise = None
        if initial_frame is not None and i2v_noised_conditioning:
            first_frame_noise = noise[:, :1].clone()  # Store the initial noise for first frame

        sample_scheduler = self._initialize_sample_scheduler(noise)
        for i, t in enumerate(tqdm(sample_scheduler.timesteps)):
            latent_model_input = latents
            timestep = t * torch.ones([latents.shape[0], latents.shape[1]], device=noise.device, dtype=torch.float32)
            
            # For i2v with clean conditioning: set first frame timestep to 0
            if initial_frame is not None and not i2v_noised_conditioning:
                timestep[:, :1] = 0

            flow_pred_cond, _ = self.generator(latent_model_input, conditional_dict, timestep)
            
            if use_cfg:
                # Apply classifier-free guidance
                flow_pred_uncond, _ = self.generator(latent_model_input, unconditional_dict, timestep)
                flow_pred = flow_pred_uncond + self.args.guidance_scale * (
                    flow_pred_cond - flow_pred_uncond)
            else:
                # No CFG: just use conditional prediction (matches training)
                flow_pred = flow_pred_cond

            temp_x0 = sample_scheduler.step(
                flow_pred.unsqueeze(0),
                t,
                latents.unsqueeze(0),
                return_dict=False)[0]
            latents = temp_x0.squeeze(0)
            
            # For i2v: restore the first frame after each denoising step
            # (ONLY if not using cross-attention - cross-attn handles conditioning differently)
            if initial_frame is not None and not self.i2v_cross_attention_latent:
                if i2v_noised_conditioning:
                    # Noised conditioning: noise the clean frame to match current timestep
                    # using the SAME noise throughout denoising for consistency
                    # t is already a scalar tensor, we need to create a tensor with proper shape
                    timestep_tensor = torch.full(
                        (initial_frame.flatten(0, 1).shape[0],), 
                        t.item() if isinstance(t, torch.Tensor) else t,
                        device=self.device, 
                        dtype=torch.long
                    )
                    noised_first_frame = sample_scheduler.add_noise(
                        initial_frame.flatten(0, 1),
                        first_frame_noise.flatten(0, 1),
                        timestep_tensor
                    ).unflatten(0, initial_frame.shape[:2])
                    latents[:, :1] = noised_first_frame
                else:
                    # Clean conditioning: restore clean frame
                    latents[:, :1] = initial_frame
        
        x0 = latents
        video = self.vae.decode_to_pixel(x0)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        del sample_scheduler

        if return_latents:
            return video, latents
        else:
            return video

    def _initialize_sample_scheduler(self, noise):
        if self.sample_solver == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(
                self.sampling_steps, device=noise.device, shift=self.shift)
            self.timesteps = sample_scheduler.timesteps
        elif self.sample_solver == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(self.sampling_steps, self.shift)
            self.timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=noise.device,
                sigmas=sampling_sigmas)
        else:
            raise NotImplementedError("Unsupported solver.")
        return sample_scheduler
