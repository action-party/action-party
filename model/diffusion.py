"""Training-time diffusion model wrapper used by :mod:`trainer.diffusion`.

:class:`CausalDiffusion` is the object the trainer interacts with: it owns the
DiT generator, text encoder and VAE (via :class:`utils.wan_wrapper.WanDiffusionWrapper`),
and dispatches to the right forward path based on the config
(``coord_token_mode`` / ``videojam_mode`` / ``position_token_mode`` / pure
video generation). It also constructs the conditioning modules imported below.
"""
from typing import Tuple
import torch

from model.base import BaseModel
from utils.wan_wrapper import WanDiffusionWrapper
from model.multi_subject import MultiSubjectConditioner
from model.spatial_conditioner import SpatialMultiSubjectConditioner
from model.action_only_conditioner import ActionOnlySpatialConditioner
from model.single_action_conditioner import SingleActionPerFrameConditioner
from utils.position_maps import generate_position_maps_batch


class CausalDiffusion(BaseModel):
    def __init__(self, args, device):
        """
        Initialize the Diffusion loss module.
        """
        # Initialize training_timesteps before super().__init__() to avoid AttributeError
        self.training_timesteps = None
        
        # Initialize coord_only_no_video before super().__init__() 
        # because _initialize_models accesses it
        self.coord_only_no_video = getattr(args, "coord_only_no_video", False)
        
        super().__init__(args, device)
        
        # Frame-level settings
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block
        
        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True

        # Enable gradient checkpointing if requested
        if getattr(args, "gradient_checkpointing", False):
            self.generator.enable_gradient_checkpointing()

        # Training hyperparameters
        self.num_train_timestep = args.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        self.guidance_scale = args.guidance_scale
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)
        self.teacher_forcing = getattr(args, "teacher_forcing", False)
        # Noise augmentation in teacher forcing, we add small noise to clean context latents
        self.noise_augmentation_max_timestep = getattr(args, "noise_augmentation_max_timestep", 0)
        # I2V noised conditioning: first frame gets same noise level as other frames
        self.i2v_noised_conditioning = getattr(args, "i2v_noised_conditioning", False)
        # I2V latent cross-attention: first frame as cross-attention context
        self.i2v_cross_attention_latent = getattr(args, "i2v_cross_attention_latent", False)
        
        # AR training mode: variable number of clean context frames
        # Each step, num_clean is randomly sampled from 1 to (num_frames - 1)
        # Frame at index num_clean gets random noise (diffusion loss)
        # Frames after that get full noise (t=max)
        self.ar_training_mode = getattr(args, "ar_training_mode", False)
        
        # Joint position diffusion (VideoJAM-style)
        # Jointly diffuse video latents (16ch) + position maps (2ch)
        self.joint_position_diffusion = getattr(args, "joint_position_diffusion", False)
        self.position_loss_weight = getattr(args, "position_loss_weight", 1.0)
        self.position_sigma = getattr(args, "position_sigma", 1.0)  # Gaussian sigma in latent pixels
        self.position_channels = getattr(args, "joint_position_channels", 2) if self.joint_position_diffusion else 0
        # Position loss type: "mse" (default), "weighted_mse", "bce", "focal"
        self.position_loss_type = getattr(args, "position_loss_type", "mse")
        # Weight multiplier for non-zero regions in weighted_mse
        self.position_foreground_weight = getattr(args, "position_foreground_weight", 100.0)
        
        # True VideoJAM mode: position latents are VAE-encoded (16ch), loaded from dataset
        # Uses same patch_embedding + Win+/Wout+ projections
        self.videojam_mode = getattr(args, "videojam_mode", False)
        if self.videojam_mode:
            # In VideoJAM mode, position_channels = 16 (same as video from VAE)
            self.position_channels = 16
            print(f"Using TRUE VIDEOJAM MODE: VAE-encoded position latents (16ch), "
                  f"Win+/Wout+ projections, loss_weight={self.position_loss_weight}")
        elif self.joint_position_diffusion:
            pass  # handled below
        
        # Coordinate token mode: (x, y) per subject as sequence tokens
        self.coord_token_mode = getattr(args, "coord_token_mode", False)
        self.coord_diffusion = getattr(args, "coord_diffusion", False)  # Diffuse coords like video
        self.coord_temporal_rope = getattr(args, "coord_temporal_rope", False)
        self.coord_spatial_rope = getattr(args, "coord_spatial_rope", False)
        self.coord_layernorm = getattr(args, "coord_layernorm", False)
        self.coord_fourier_encoding = getattr(args, "coord_fourier_encoding", False)
        self.coord_fourier_frequencies = getattr(args, "coord_fourier_frequencies", 10)
        self.coord_direct_embedding = getattr(args, "coord_direct_embedding", False)
        self.coord_no_encoder_decoder = getattr(args, "coord_no_encoder_decoder", False)
        self.coord_scale_factor = getattr(args, "coord_scale_factor", 1.0)  # Scale [0,1] coords to match noise
        self.num_subjects = getattr(args, "num_subjects", 2)
        self.coord_loss_weight = getattr(args, "coord_loss_weight", 1.0)
        self.coord_only_mode = getattr(args, "coord_only_mode", False)  # Only train coord decoder, freeze video
        self.coord_only_no_video = getattr(args, "coord_only_no_video", False)  # Pass zeros for video input
        self.coord_uniform_weight = getattr(args, "coord_uniform_weight", False)  # Disable bell-curve weighting for coords
        self.log_per_subject_loss = getattr(args, "log_per_subject_loss", False)  # Log loss per subject for debugging
        if self.coord_token_mode:
            features = []
            if self.coord_diffusion:
                features.append("DIFFUSION")  # Key feature - diffuse coords like video
            if self.coord_temporal_rope:
                features.append("temporal RoPE")
            if self.coord_spatial_rope:
                features.append("spatial RoPE")
            if self.coord_layernorm:
                features.append("LayerNorm")
            features_str = f" with {', '.join(features)}" if features else ""
            print(f"Using COORD TOKEN MODE: {self.num_subjects} subjects, "
                  f"coordinates as sequence tokens{features_str}, loss_weight={self.coord_loss_weight}")
        
        # Position token mode: diffuse per-subject position maps as sequential tokens
        self.position_token_mode = getattr(args, "position_token_mode", False)
        if self.position_token_mode:
            self.position_loss_weight = getattr(args, "position_loss_weight", 1.0)
            print(f"Using POSITION TOKEN MODE: Position maps as sequential tokens with 3D RoPE, "
                  f"loss_weight={self.position_loss_weight}")
        
        # Pixel position mode: diffuse raw pixel-space position maps (not VAE encoded)
        self.pixel_position_mode = getattr(args, "pixel_position_mode", False)
        if self.pixel_position_mode:
            self.position_loss_weight = getattr(args, "position_loss_weight", 1.0)
            self.pixel_position_num_subjects = getattr(args, "pixel_position_num_subjects", 2)
            print(f"Using PIXEL POSITION MODE: {self.pixel_position_num_subjects} subjects, "
                  f"diffusing raw pixel position maps, loss_weight={self.position_loss_weight}")
        
        if self.joint_position_diffusion and not self.videojam_mode and not self.coord_token_mode:
            print(f"Using JOINT POSITION DIFFUSION: {self.position_channels} position channels, "
                  f"sigma={self.position_sigma}, loss_weight={self.position_loss_weight}, "
                  f"loss_type={self.position_loss_type}")
        
        # I2V latent cross-attention: will be initialized after checkpoint load
        # (needs to be after _initialize_models and checkpoint loading)
        
        # Optional: Train on specific timesteps instead of random sampling
        self.train_on_denoising_steps = getattr(args, "train_on_denoising_steps", False)
        if self.train_on_denoising_steps and hasattr(args, 'denoising_step_list'):
            self.training_timesteps = torch.tensor(args.denoising_step_list, dtype=torch.long, device=device)
            # Apply warp if configured
            if getattr(args, 'warp_denoising_step', False):
                timesteps_all = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
                self.training_timesteps = torch.tensor(
                    [timesteps_all[1000 - t].item() for t in self.training_timesteps],
                    dtype=torch.float32,
                    device=device
                )
            print(f"Training on specific timesteps: {self.training_timesteps.tolist()}")
        
        # Handle alphas_cumprod if it exists (for some scheduler types)
        if getattr(self.scheduler, "alphas_cumprod", None) is not None:
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        else:
            self.scheduler.alphas_cumprod = None

    def _initialize_models(self, args, device):
        # Infer is_causal from num_frame_per_block (if it's < num_frames, we need causal masking)
        num_frames = args.image_or_video_shape[1]
        num_frame_per_block = getattr(args, "num_frame_per_block", num_frames)
        is_causal = num_frame_per_block < num_frames
        
        # Get model kwargs and add joint position diffusion settings if enabled
        model_kwargs = dict(getattr(args, "model_kwargs", {}))
        
        # True VideoJAM mode: position latents are VAE-encoded (16ch)
        videojam_mode = getattr(args, "videojam_mode", False)
        coord_token_mode = getattr(args, "coord_token_mode", False)
        num_subjects = getattr(args, "num_subjects", 2)
        
        # Position token mode: position maps as sequential tokens with 3D RoPE
        position_token_mode = getattr(args, "position_token_mode", False)
        
        if position_token_mode:
            model_kwargs["position_token_mode"] = True
            model_kwargs["num_subjects"] = num_subjects
            position_temporal_offset = getattr(args, "position_temporal_offset", 0.0)
            model_kwargs["position_temporal_offset"] = position_temporal_offset
            subject_isolation_self_attn = getattr(args, "subject_isolation_self_attn", False)
            model_kwargs["subject_isolation_self_attn"] = subject_isolation_self_attn
            position_no_video_attn = getattr(args, "position_no_video_attn", False)
            model_kwargs["position_no_video_attn"] = position_no_video_attn
            position_skip_self_attn = getattr(args, "position_skip_self_attn", False)
            model_kwargs["position_skip_self_attn"] = position_skip_self_attn
            position_no_self_attn_update = getattr(args, "position_no_self_attn_update", False)
            model_kwargs["position_no_self_attn_update"] = position_no_self_attn_update
            use_flex_attention = getattr(args, "use_flex_attention", True)  # Default to True
            model_kwargs["use_flex_attention"] = use_flex_attention
            separate_position_modulation = getattr(args, "separate_position_modulation", False)
            model_kwargs["separate_position_modulation"] = separate_position_modulation
            cross_attn_rope = getattr(args, "cross_attn_rope", False)
            model_kwargs["cross_attn_rope"] = cross_attn_rope
            per_frame_cross_attn = getattr(args, "per_frame_cross_attn", False)
            model_kwargs["per_frame_cross_attn"] = per_frame_cross_attn
            isolation_str = ", subject_isolation=True" if subject_isolation_self_attn else ""
            no_video_str = ", pos_no_video=True" if position_no_video_attn else ""
            skip_self_attn_str = ", pos_skip_self_attn=True" if position_skip_self_attn else ""
            no_update_str = ", pos_no_update=True" if position_no_self_attn_update else ""
            flex_str = ", flex_attn=True" if use_flex_attention else ", flex_attn=False"
            sep_mod_str = ", separate_modulation=True" if separate_position_modulation else ""
            cross_rope_str = ", cross_attn_rope=True" if cross_attn_rope else ""
            per_frame_str = ", per_frame_cross_attn=True" if per_frame_cross_attn else ""
            print(f"Initializing model in POSITION TOKEN MODE ({num_subjects} subjects, temporal_offset={position_temporal_offset}{isolation_str}{no_video_str}{skip_self_attn_str}{no_update_str}{flex_str}{sep_mod_str}{cross_rope_str}{per_frame_str})")
        
        # Pixel position mode: diffuse raw pixel-space position maps
        pixel_position_mode = getattr(args, "pixel_position_mode", False)
        if pixel_position_mode:
            pixel_position_num_subjects = getattr(args, "pixel_position_num_subjects", 2)
            model_kwargs["pixel_position_mode"] = True
            model_kwargs["pixel_position_num_subjects"] = pixel_position_num_subjects
            use_flex_attention = getattr(args, "use_flex_attention", True)
            model_kwargs["use_flex_attention"] = use_flex_attention
            print(f"Initializing model in PIXEL POSITION MODE ({pixel_position_num_subjects} subjects)")
        
        if coord_token_mode:
            model_kwargs["coord_token_mode"] = True
            model_kwargs["num_subjects"] = num_subjects
            coord_temporal_rope = getattr(args, "coord_temporal_rope", False)
            coord_spatial_rope = getattr(args, "coord_spatial_rope", False)
            coord_layernorm = getattr(args, "coord_layernorm", False)
            coord_fourier_encoding = getattr(args, "coord_fourier_encoding", False)
            coord_fourier_frequencies = getattr(args, "coord_fourier_frequencies", 10)
            model_kwargs["coord_temporal_rope"] = coord_temporal_rope
            model_kwargs["coord_spatial_rope"] = coord_spatial_rope
            model_kwargs["coord_layernorm"] = coord_layernorm
            model_kwargs["coord_fourier_encoding"] = coord_fourier_encoding
            model_kwargs["coord_fourier_frequencies"] = coord_fourier_frequencies
            coord_direct_embedding = getattr(args, "coord_direct_embedding", False)
            model_kwargs["coord_direct_embedding"] = coord_direct_embedding
            coord_no_encoder_decoder = getattr(args, "coord_no_encoder_decoder", False)
            model_kwargs["coord_no_encoder_decoder"] = coord_no_encoder_decoder
            coord_scale_factor = getattr(args, "coord_scale_factor", 1.0)
            model_kwargs["coord_scale_factor"] = coord_scale_factor
            # Per-frame timesteps for AR training with different noise levels per frame
            per_frame_timesteps = getattr(args, "per_frame_timesteps", False)
            model_kwargs["per_frame_timesteps"] = per_frame_timesteps
            # Subject isolation: coord tokens from different subjects cannot attend to each other
            subject_isolation_self_attn = getattr(args, "subject_isolation_self_attn", False)
            model_kwargs["subject_isolation_self_attn"] = subject_isolation_self_attn
            # Cross-attention subject isolation: coord tokens attend only to their own subject's actions
            cross_attn_subject_isolation = getattr(args, "cross_attn_subject_isolation", True)
            model_kwargs["cross_attn_subject_isolation"] = cross_attn_subject_isolation
            # Cross-attention RoPE: apply 3D RoPE to coord tokens and action embeddings
            cross_attn_rope = getattr(args, "cross_attn_rope", False)
            model_kwargs["cross_attn_rope"] = cross_attn_rope
            # Per-frame cross-attention: coord token at frame t only attends to action[t-1]
            per_frame_cross_attn = getattr(args, "per_frame_cross_attn", False)
            model_kwargs["per_frame_cross_attn"] = per_frame_cross_attn
            features = []
            if coord_no_encoder_decoder:
                features.append("pure slice (no enc/dec)")
            elif coord_direct_embedding:
                features.append("direct")
            elif coord_fourier_encoding:
                features.append(f"Fourier({coord_fourier_frequencies})")
            if coord_temporal_rope:
                features.append("temporal RoPE")
            if coord_spatial_rope:
                features.append("spatial RoPE")
            if coord_layernorm:
                features.append("LayerNorm")
            if subject_isolation_self_attn:
                features.append("self-attn isolation")
            if not cross_attn_subject_isolation:
                features.append("NO cross-attn isolation")
            if cross_attn_rope:
                features.append("cross-attn RoPE")
            if per_frame_cross_attn:
                features.append("per-frame cross-attn")
            features_str = f" with {', '.join(features)}" if features else ""
            print(f"Initializing model in coord_token_mode ({num_subjects} subjects{features_str})")
        elif videojam_mode:
            model_kwargs["videojam_mode"] = True
            # Identity init: [I,0] for Win+/Wout+ (cleaner, no residuals)
            videojam_identity_init = getattr(args, "videojam_identity_init", False)
            model_kwargs["videojam_identity_init"] = videojam_identity_init
            # Position residual option (only used when not using identity init)
            videojam_position_residual = getattr(args, "videojam_position_residual", True)
            model_kwargs["videojam_position_residual"] = videojam_position_residual
            # Don't set position_channels - VideoJAM uses same in_dim (16) for both video and position
            if videojam_identity_init:
                print(f"Initializing model in VideoJAM mode with [I,0] identity init (no residuals)")
            else:
                residual_str = "with" if videojam_position_residual else "without"
                print(f"Initializing model in VideoJAM mode (zero-init) {residual_str} position residual")
        else:
            # Legacy mode: raw position maps with specified channels
            joint_position_channels = getattr(args, "joint_position_channels", 0)
            if getattr(args, "joint_position_diffusion", False) and joint_position_channels > 0:
                model_kwargs["position_channels"] = joint_position_channels
                print(f"Initializing model with position_channels={joint_position_channels}")
        
        # Per-frame timesteps: always check, not just for coord_token_mode
        # (For video-only AR training with different noise levels per frame)
        if not coord_token_mode:
            per_frame_timesteps = getattr(args, "per_frame_timesteps", False)
            if per_frame_timesteps:
                model_kwargs["per_frame_timesteps"] = per_frame_timesteps
                print(f"[VIDEO-ONLY] Per-frame timesteps enabled for AR training")
        
        # Pass image_or_video_shape for seq_len calculation
        model_kwargs["image_or_video_shape"] = list(args.image_or_video_shape)
        
        self.generator = WanDiffusionWrapper(**model_kwargs, is_causal=is_causal)
        self.generator.model.requires_grad_(True)
        
        # Set coord_only_no_video flag on the model
        if self.coord_only_no_video:
            self.generator.model.coord_only_no_video = True
            print("COORD ONLY NO VIDEO: Skipping video patches entirely, only coordinate tokens in sequence")
        
        # Convert to correct device and dtype before FSDP wrapping
        # This ensures all parameters match the training dtype (FSDP expects this)
        dtype = torch.bfloat16 if args.mixed_precision else torch.float32
        self.generator = self.generator.to(device=device, dtype=dtype)
        
        # Initialize latent_patch_embedding for i2v cross-attention (part of generator.model)
        # This will be initialized in the trainer AFTER checkpoint loading
        if getattr(args, "i2v_cross_attention_latent", False):
            self.i2v_cross_attention_latent = True
        else:
            self.i2v_cross_attention_latent = False
        
        # Initialize action embedding for action-conditional generation
        # This is created INSIDE the model (not added afterward) for proper FSDP integration
        self.use_action_conditioning = getattr(args, "use_action_conditioning", False)
        if self.use_action_conditioning:
            num_action_classes = getattr(args, "num_action_classes", 5)
            
            # Option to use direct embeddings (skip text_embedding MLP projection)
            # This gives stronger gradients and avoids bottleneck
            use_direct_action_embeddings = getattr(args, "use_direct_action_embeddings", False)
            
            if use_direct_action_embeddings:
                # Use model.dim (1536) directly - skip text_embedding projection
                action_embed_dim = self.generator.model.dim
                print(f"Using direct action embeddings: {num_action_classes} classes × {action_embed_dim} dims")
            else:
                # Use model.text_dim (4096) - will be projected by text_embedding MLP
                action_embed_dim = self.generator.model.text_dim
                print(f"Using standard action embeddings: {num_action_classes} classes × {action_embed_dim} dims (projected to {self.generator.model.dim})")
            
            # Create action embedding layer and register it properly as a submodule
            action_embedding = torch.nn.Embedding(
                num_embeddings=num_action_classes,
                embedding_dim=action_embed_dim
            )
            # Initialize with small random values (before moving to device)
            torch.nn.init.normal_(action_embedding.weight, mean=0.0, std=0.02)
            
            # Option to freeze action embeddings (let k/v projections do all the learning)
            freeze_action_embeddings = getattr(args, "freeze_action_embeddings", False)
            if freeze_action_embeddings:
                action_embedding.weight.requires_grad = False
                print("Action embeddings FROZEN - only k/v projections will learn")
            
            # Move to device/dtype
            action_embedding = action_embedding.to(device=device, dtype=dtype)
            # Register as a submodule (ensures FSDP wraps it correctly)
            self.generator.model.add_module('action_embedding', action_embedding)
            
            # Store flag for inference
            self.generator.model.use_direct_action_embeddings = use_direct_action_embeddings
            
            # Replace text_embedding with identity if using direct embeddings
            # This skips the 4096→1536 projection, giving direct gradient path
            if use_direct_action_embeddings:
                print("Replacing text_embedding MLP with identity (direct gradient path)")
                self.generator.model.text_embedding = torch.nn.Identity()

        # Single-subject modulated conditioning (FiLM-style for single subject, e.g., blue square)
        self.use_single_subject_modulated = getattr(args, "use_single_subject_modulated", False)
        
        if self.use_single_subject_modulated:
            from model.multi_subject_simple import SingleSubjectModulatedConditioner
            
            num_action_classes = getattr(args, "num_action_classes", 5)
            action_embed_dim = getattr(args, "action_embed_dim", 256)
            freeze_base_embedding = getattr(args, "freeze_base_embedding", False)
            
            self.single_subject_conditioner = SingleSubjectModulatedConditioner(
                num_action_classes=num_action_classes,
                model_dim=self.generator.model.dim,  # 1536
                action_embed_dim=action_embed_dim,
                freeze_base_embedding=freeze_base_embedding,
            )
            self.single_subject_conditioner = self.single_subject_conditioner.to(device=device, dtype=dtype)
            
            print(f"Using SINGLE-SUBJECT modulated conditioning (FiLM): "
                  f"{num_action_classes} actions, action_embed_dim={action_embed_dim}")
            
            # Replace text_embedding with identity
            print("Replacing text_embedding MLP with identity (single-subject modulated mode)")
            self.generator.model.text_embedding = torch.nn.Identity()

        # Initialize multi-subject conditioner (combines subject features with action embeddings)
        self.use_multi_subject_conditioning = getattr(args, "use_multi_subject_conditioning", False)
        self.use_simple_multi_subject_conditioning = getattr(args, "use_simple_multi_subject_conditioning", False)
        self.use_modulated_multi_subject_conditioning = getattr(args, "use_modulated_multi_subject_conditioning", False)
        
        if self.use_modulated_multi_subject_conditioning:
            # Modulated conditioner: Actions modulate subject embeddings via FiLM (scale/shift)
            from model.multi_subject_simple import ModulatedMultiSubjectConditioner
            
            num_action_classes = getattr(args, "num_action_classes", 5)
            num_subjects_total = getattr(args, "num_subjects_total", 6)
            num_subjects_per_video = getattr(args, "num_subjects_per_video", 2)
            action_embed_dim = getattr(args, "action_embed_dim", 256)
            freeze_subject_embeddings = getattr(args, "freeze_subject_embeddings", True)
            
            self.multi_subject_conditioner = ModulatedMultiSubjectConditioner(
                num_action_classes=num_action_classes,
                num_subjects=num_subjects_total,
                model_dim=self.generator.model.dim,  # 1536
                action_embed_dim=action_embed_dim,
                freeze_subject_embeddings=freeze_subject_embeddings,
            )
            self.multi_subject_conditioner = self.multi_subject_conditioner.to(device=device, dtype=dtype)
            
            # Store for cross-attention
            self.generator.model.num_subjects = num_subjects_per_video
            
            print(f"Using MODULATED multi-subject conditioning (FiLM): {num_subjects_total} subjects, "
                  f"{num_action_classes} actions, action_embed_dim={action_embed_dim}")
            
            # Replace text_embedding with identity (we're using direct embeddings)
            print("Replacing text_embedding MLP with identity (modulated multi-subject mode)")
            self.generator.model.text_embedding = torch.nn.Identity()
            
        elif self.use_simple_multi_subject_conditioning:
            # Simple conditioner: learned embeddings for both subjects and actions
            from model.multi_subject_simple import SimpleMultiSubjectConditioner
            
            num_action_classes = getattr(args, "num_action_classes", 5)
            num_subjects_total = getattr(args, "num_subjects_total", 6)
            num_subjects_per_video = getattr(args, "num_subjects_per_video", 2)
            combine_mode = getattr(args, "combine_mode", "add")
            
            freeze_subject_embeddings = getattr(args, "freeze_subject_embeddings", False)
            freeze_action_embeddings = getattr(args, "freeze_action_embeddings", False)
            
            self.multi_subject_conditioner = SimpleMultiSubjectConditioner(
                num_action_classes=num_action_classes,
                num_subjects=num_subjects_total,
                model_dim=self.generator.model.dim,  # 1536
                combine_mode=combine_mode,
                freeze_subject_embeddings=freeze_subject_embeddings,
                freeze_action_embeddings=freeze_action_embeddings,
            )
            self.multi_subject_conditioner = self.multi_subject_conditioner.to(device=device, dtype=dtype)
            
            # Store for cross-attention
            self.generator.model.num_subjects = num_subjects_per_video
            
            print(f"Using SIMPLE multi-subject conditioning: {num_subjects_total} subjects, "
                  f"{num_action_classes} actions, combine_mode={combine_mode}")
            
            # Replace text_embedding with identity (we're using direct embeddings)
            print("Replacing text_embedding MLP with identity (simple multi-subject mode)")
            self.generator.model.text_embedding = torch.nn.Identity()
        
        # Spatial RoPE conditioning (3D RoPE in cross-attention for explicit spatial correspondence)
        self.use_spatial_rope_conditioning = getattr(args, "use_spatial_rope_conditioning", False)
        self.use_action_only_conditioning = getattr(args, "use_action_only_conditioning", False)
        self.use_single_action_conditioning = getattr(args, "use_single_action_conditioning", False)
        
        if self.use_spatial_rope_conditioning:
            num_action_classes = getattr(args, "num_action_classes", 5)
            num_subjects_total = getattr(args, "num_subjects_total", 2)
            num_subjects_per_video = getattr(args, "num_subjects_per_video", 2)
            speed = getattr(args, "position_speed", 6)
            freeze_embeddings = getattr(args, "freeze_embeddings", True)
            patch_size = getattr(args, "latent_patch_size", 8)
            
            if self.use_single_action_conditioning:
                # Single action per frame mode: one action token per frame, no subject ambiguity
                self.multi_subject_conditioner = SingleActionPerFrameConditioner(
                    num_action_classes=num_action_classes,
                    model_dim=self.generator.model.dim,  # 1536
                    patch_size=patch_size,
                    speed=speed,
                )
                print(f"Using SINGLE ACTION PER FRAME conditioning: "
                      f"{num_action_classes} actions, speed={speed}px/frame, patch_size={patch_size}")
                print("  (One action per frame - position specifies which subject)")
            elif self.use_action_only_conditioning:
                # Action-only mode: no subject embedding, just action + spatial RoPE
                zero_position_for_masked = getattr(args, "zero_position_for_masked", False)
                position_before_action = getattr(args, "position_before_action", False)
                self.zero_position_for_masked = zero_position_for_masked  # Store for coord token spatial RoPE
                image_size = float(getattr(args, 'synthetic_height', 256))
                self.multi_subject_conditioner = ActionOnlySpatialConditioner(
                    num_action_classes=num_action_classes,
                    num_subjects=num_subjects_total,
                    model_dim=self.generator.model.dim,  # 1536
                    patch_size=patch_size,
                    speed=speed,
                    zero_position_for_masked=zero_position_for_masked,
                    position_before_action=position_before_action,
                    image_size=image_size,
                )
                self.use_gt_positions_for_rope = getattr(args, "use_gt_positions_for_rope", False)
                print(f"Using ACTION-ONLY SPATIAL ROPE conditioning: "
                      f"{num_action_classes} actions, speed={speed}px/frame, patch_size={patch_size}")
                if self.use_gt_positions_for_rope:
                    print("  Using GT frame_coords for spatial RoPE (relative action space)")
                print("  (No subject embeddings - subject identity comes from spatial RoPE only)")
            else:
                # Full spatial RoPE: action + subject embeddings
                self.multi_subject_conditioner = SpatialMultiSubjectConditioner(
                    num_action_classes=num_action_classes,
                    num_subjects=num_subjects_total,
                    model_dim=self.generator.model.dim,  # 1536
                    patch_size=patch_size,
                    speed=speed,
                    freeze_embeddings=freeze_embeddings,
                )
                print(f"Using SPATIAL ROPE multi-subject conditioning: {num_subjects_total} subjects, "
                      f"{num_action_classes} actions, speed={speed}px/frame, patch_size={patch_size}")
            
            self.multi_subject_conditioner = self.multi_subject_conditioner.to(device=device, dtype=dtype)
            
            # Store for cross-attention
            self.generator.model.num_subjects = num_subjects_per_video
            
            # Replace text_embedding with identity ONLY if not using text conditioning
            # Keep the real projection (4096 -> 1536) when:
            #   - fixed_prompt is set (single-game text conditioning)
            #   - cross_attn_type is masked_position_cross_attn (video tokens attend to text in cross-attn)
            # The dimension check in WanModel._forward (context.size(-1) != self.dim) ensures
            # action embeddings (already 1536) skip the projection automatically.
            fixed_prompt = getattr(args, "fixed_prompt", None)
            cross_attn_type = getattr(args, 'model_kwargs', {}).get('cross_attn_type', None)
            needs_text_projection = (
                fixed_prompt or 
                cross_attn_type in ('masked_position_cross_attn', 'dual_stream_cross_attn')
            )
            if not needs_text_projection:
                print("Replacing text_embedding MLP with identity (spatial rope mode, no text conditioning)")
                self.generator.model.text_embedding = torch.nn.Identity()
            else:
                reason = f"fixed_prompt='{fixed_prompt}'" if fixed_prompt else f"cross_attn_type={cross_attn_type}"
                print(f"Keeping text_embedding projection (4096->1536) for video text conditioning ({reason})")
                if fixed_prompt:
                    # Mark that we need to update cross-attn embedding after text encoder is loaded
                    self._pending_fixed_prompt_setup = fixed_prompt
        
        # Position-aware multi-subject conditioning (adds x,y position encoding)
        self.use_position_aware_conditioning = getattr(args, "use_position_aware_conditioning", False)
        if self.use_position_aware_conditioning:
            from model.multi_subject_position import PositionAwareMultiSubjectConditioner
            
            num_action_classes = getattr(args, "num_action_classes", 5)
            num_subjects_total = getattr(args, "num_subjects_total", 2)
            num_subjects_per_video = getattr(args, "num_subjects_per_video", 2)
            image_size = getattr(args, "position_image_size", 256)
            speed = getattr(args, "position_speed", 2)
            freeze_embeddings = getattr(args, "freeze_embeddings", True)
            
            self.multi_subject_conditioner = PositionAwareMultiSubjectConditioner(
                num_action_classes=num_action_classes,
                num_subjects=num_subjects_total,
                model_dim=self.generator.model.dim,  # 1536
                image_size=image_size,
                speed=speed,
                freeze_embeddings=freeze_embeddings,
            )
            self.multi_subject_conditioner = self.multi_subject_conditioner.to(device=device, dtype=dtype)
            
            # Store for cross-attention
            self.generator.model.num_subjects = num_subjects_per_video
            
            print(f"Using POSITION-AWARE multi-subject conditioning: {num_subjects_total} subjects, "
                  f"{num_action_classes} actions, speed={speed}px/frame")
            
            # Replace text_embedding with identity (we're using direct embeddings)
            print("Replacing text_embedding MLP with identity (position-aware mode)")
            self.generator.model.text_embedding = torch.nn.Identity()
            
        elif self.use_multi_subject_conditioning:
            # Original conditioner: pooled image features for subjects
            num_action_classes = getattr(args, "num_action_classes", 5)
            action_dim = getattr(args, "action_dim", 512)
            subject_dim = getattr(args, "subject_dim", 1024)
            num_subjects = getattr(args, "num_subjects", 2)
            
            self.multi_subject_conditioner = MultiSubjectConditioner(
                latent_channels=getattr(args, "latent_channels", 16),
                model_dim=self.generator.model.dim,  # 1536
                patch_size=tuple(getattr(args, "patch_size", (1, 2, 2))),
                num_action_classes=num_action_classes,
                action_dim=action_dim,
                subject_dim=subject_dim,
                num_pool_heads=getattr(args, "num_pool_heads", 8),
            )
            self.multi_subject_conditioner = self.multi_subject_conditioner.to(device=device, dtype=dtype)
            
            # Store for cross-attention
            self.generator.model.num_subjects = num_subjects
            
            print(f"Using multi-subject conditioning (image pooling): {num_subjects} subjects, "
                  f"action_dim={action_dim}, subject_dim={subject_dim}")
            
            # Replace text_embedding with identity (we're using direct embeddings)
            print("Replacing text_embedding MLP with identity (multi-subject mode)")
            self.generator.model.text_embedding = torch.nn.Identity()

        # Only load text encoder if needed
        if getattr(args, "load_text_encoder", True):
            from utils.wan_wrapper import WanTextEncoder
            self.text_encoder = WanTextEncoder()
            self.text_encoder.requires_grad_(False)
            
            # If we have a pending fixed_prompt setup, update cross-attention embedding now
            if hasattr(self, '_pending_fixed_prompt_setup') and self._pending_fixed_prompt_setup:
                fixed_prompt = self._pending_fixed_prompt_setup
                if hasattr(self.generator, 'set_video_cross_attn_embedding'):
                    success = self.generator.set_video_cross_attn_embedding(fixed_prompt, text_encoder=self.text_encoder)
                    if success:
                        print(f"  Updated video cross-attn embedding to fixed_prompt: \"{fixed_prompt}\"")
                    else:
                        print(f"  WARNING: Failed to set fixed_prompt embedding, using default null embedding")
                delattr(self, '_pending_fixed_prompt_setup')
                
                # Free text encoder memory since we only needed it for the fixed_prompt embedding
                print("  Freeing text encoder memory (fixed_prompt embedding cached in model)")
                del self.text_encoder
                self.text_encoder = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            self.text_encoder = None

        # Only load VAE if needed (for raw video encoding or visualization)
        if getattr(args, "load_vae", True):
            from utils.wan_wrapper import WanVAEWrapper
            self.vae = WanVAEWrapper()
            self.vae.requires_grad_(False)
        else:
            self.vae = None

        # Initialize scheduler from generator
        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)
    
    def initialize_multi_subject_patch_embedding(self):
        """Copy patch_embedding weights to multi-subject conditioner after checkpoint load."""
        if hasattr(self, 'multi_subject_conditioner'):
            self.multi_subject_conditioner.initialize_from_patch_embedding(
                self.generator.model.patch_embedding
            )
            print("  ✓ Initialized multi-subject patch embedding from model patch_embedding")

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
        action_labels: torch.Tensor = None,
        subject_prototypes: torch.Tensor = None,
        subject_ids: torch.Tensor = None,
        initial_positions: torch.Tensor = None,  # [B, S, 2] for position-aware conditioning
        position_latents: torch.Tensor = None,  # [B, T, C, H, W] VAE-encoded position maps (VideoJAM mode)
        frame_coords: torch.Tensor = None,  # [B, T, S, 2] per-frame coordinates for coord_token_mode
        position_pixels: torch.Tensor = None,  # [B, S, T, H, W] raw pixel position maps (pixel_position_mode)
        subject_mask: torch.Tensor = None,  # [B, S] boolean, True for real subjects (multi-game)
        video_text_embeds: torch.Tensor = None,  # [B, 512, 4096] pre-encoded game text for video cross-attn
        pre_sampled_num_clean: torch.Tensor = None,  # [B] optional pre-sampled num_clean for text action conditioning
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and compute the DMD loss.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - generator_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        noise = torch.randn_like(clean_latent)
        batch_size, num_frame = image_or_video_shape[:2]
        
        # Position token mode: position latents as separate sequential tokens
        # Generate noise for position latents separately (same timestep as video)
        noisy_position_latents = None
        position_noise = None
        clean_position_latents = None
        num_position_subjects = 1
        
        # Pixel position mode: raw pixel position maps as separate tokens
        noisy_position_pixels = None
        position_pixel_noise = None
        clean_position_pixels = None
        if getattr(self, 'pixel_position_mode', False) and position_pixels is not None:
            # position_pixels: [B, S, T, H, W]
            clean_position_pixels = position_pixels
            position_pixel_noise = torch.randn_like(position_pixels)
            num_position_subjects = position_pixels.shape[1]
            # Note: noisy_position_pixels will be computed after timestep sampling
        
        if getattr(self, 'position_token_mode', False) and position_latents is not None:
            # position_latents can be:
            # - [B, S, T, C, H, W] for multi-subject
            # - [B, T, C, H, W] for single subject
            if position_latents.dim() == 6:
                # Multi-subject: [B, S, T, C, H, W]
                num_position_subjects = position_latents.shape[1]
            clean_position_latents = position_latents
            position_noise = torch.randn_like(position_latents)
            # Note: noisy_position_latents will be computed after timestep sampling
        
        # VideoJAM mode: use VAE-encoded position latents from dataset
        position_maps = None
        if getattr(self, 'videojam_mode', False) and position_latents is not None:
            # position_latents: [B, T, C, H, W] - VAE-encoded position maps (16 channels)
            # Concatenate with video latent: [B, T, 32, H, W]
            clean_latent_with_pos = torch.cat([clean_latent, position_latents], dim=2)
            
            # Generate noise for both video and position
            noise = torch.randn_like(clean_latent_with_pos)
            
            # Store original clean_latent for logging
            clean_latent_video = clean_latent
            clean_latent = clean_latent_with_pos
            # Store position_latents for loss computation
            position_maps = position_latents
        
        # Legacy joint position diffusion: generate position maps on-the-fly
        elif getattr(self, 'joint_position_diffusion', False) and action_labels is not None and initial_positions is not None:
            # Generate position maps on-the-fly from GT data
            # action_labels: [B, S, T] - per-subject action sequences
            # initial_positions: [B, S, 2] - initial (y, x) in pixels
            _, H, W = clean_latent.shape[2:]  # [B, T, C, H, W]
            speed = getattr(self, 'multi_subject_conditioner', None)
            speed = speed.speed if speed is not None else getattr(self, 'position_speed', 16)
            
            # Generate Gaussian position maps: [B, T, S, H, W]
            position_maps = generate_position_maps_batch(
                initial_positions=initial_positions,
                actions=action_labels,
                H=H, W=W,
                sigma=self.position_sigma,
                speed=speed,
                pixel_to_latent_scale=8.0
            ).to(clean_latent.device, dtype=clean_latent.dtype)
            
            # Concatenate with video latent: [B, T, C+S, H, W]
            clean_latent_with_pos = torch.cat([clean_latent, position_maps], dim=2)
            
            # Generate noise for both video and position
            noise = torch.randn_like(clean_latent_with_pos)
            
            # Store original clean_latent for logging
            clean_latent_video = clean_latent
            clean_latent = clean_latent_with_pos
        
        # Handle i2v mode: use first frame from clean_latent if no initial_latent provided
        if initial_latent is None and getattr(self, 'independent_first_frame', False):
            # Extract first frame as the conditioning frame for i2v
            initial_latent = clean_latent[:, :1].clone()
        
        # AR training mode: variable number of clean context frames
        # - Frames 0 to (num_clean-1): clean (noise=0, t=0)
        # - Frame num_clean: target frame (random noise, random t) - loss computed here
        # - Frames (num_clean+1) to end: full noise (t=max)
        ar_training_mode = getattr(self, 'ar_training_mode', False)
        num_clean = None  # Will be set if ar_training_mode is True
        if ar_training_mode:
            if pre_sampled_num_clean is not None:
                num_clean = pre_sampled_num_clean
            else:
                # Randomly sample num_clean from 1 to (num_frame - 1) for each batch element
                # This ensures at least 1 clean frame and at least 1 frame for diffusion loss
                num_clean = torch.randint(1, num_frame, (batch_size,), device=self.device)
            
            # Set initial_latent to the clean context frames (use max num_clean for shape)
            max_num_clean = num_clean.max().item()
            initial_latent = clean_latent[:, :max_num_clean].clone()
            
            # Set noise to 0 for clean context frames (per-batch varying)
            for b in range(batch_size):
                noise[b, :num_clean[b]] = 0.0
            # Keep random noise for target frame (index num_clean[b])
            # Set full noise (will be handled via timestep=max) for future frames
            
            # Joint position diffusion: zero out position maps for masked (future) frames
            if getattr(self, 'joint_position_diffusion', False) and position_maps is not None:
                video_channels = clean_latent.shape[2] - self.position_channels
                for b in range(batch_size):
                    nc = num_clean[b].item()
                    # Zero out position channels for frames after target (unknown positions)
                    if nc + 1 < num_frame:
                        clean_latent[b, nc + 1:, video_channels:] = 0.0
            
            # Position token mode: set position noise to 0 for clean context frames
            if getattr(self, 'position_token_mode', False) and position_noise is not None:
                for b in range(batch_size):
                    position_noise[b, :num_clean[b]] = 0.0
        
        # Handle SINGLE-SUBJECT modulated conditioning (FiLM-style for single subject)
        if getattr(self, 'use_single_subject_modulated', False) and action_labels is not None:
            # action_labels: [B, T] - per-frame action sequence
            
            # For AR training: mask future actions with "still" (0)
            if ar_training_mode and num_clean is not None:
                masked_action_labels = action_labels.clone()
                for b in range(batch_size):
                    nc = num_clean[b].item()
                    if nc < action_labels.shape[1]:
                        masked_action_labels[b, nc:] = 0
                action_embeds = self.single_subject_conditioner(masked_action_labels)
            else:
                action_embeds = self.single_subject_conditioner(action_labels)
            
            # Ensure dtype matches model
            action_embeds = action_embeds.to(dtype=clean_latent.dtype)
            
            # Replace conditional_dict
            conditional_dict = {"prompt_embeds": action_embeds}
            unconditional_dict = conditional_dict
        
        # Handle action conditioning: replace text embeddings with action embeddings
        # NOTE: This must come AFTER num_clean is determined for AR mode masking
        elif self.use_action_conditioning and action_labels is not None:
            # Detect if single action per video [B] or per-frame actions [B, N-1]
            if action_labels.dim() == 1:
                # Single action per video: [batch_size] -> [batch_size, 1, dim]
                action_embeds = self.generator.model.action_embedding(action_labels)
                action_embeds = action_embeds.unsqueeze(1)
                
                # For temporal_action_cross_attn: repeat action N times (one per frame)
                if getattr(self.generator.model, 'cross_attn_type', None) == 'temporal_action_cross_attn':
                    action_embeds = action_embeds.repeat(1, num_frame, 1)  # [batch_size, num_frame, dim]
            else:
                # Per-frame actions: [batch_size, N-1] -> [batch_size, N-1, dim]
                # action_labels[b, i] = action that produces frame i+1 (transition from frame i to i+1)
                # This is "action-as-result" semantics: action[i] is associated with the frame it produces
                
                # For AR training: mask future actions with "still" (0) for noise frames
                # This prevents information leakage from unknown future actions
                # When denoising frame nc, we only know actions that produced frames 1..nc
                # That's action[0]..action[nc-1]. Mask action[nc] onwards.
                if ar_training_mode and num_clean is not None:
                    # Clone action_labels to avoid modifying the original
                    masked_action_labels = action_labels.clone()
                    for b in range(batch_size):
                        nc = num_clean[b].item()
                        # action[nc-1] produces frame nc (the target frame we're denoising)
                        # action[nc] produces frame nc+1 (unknown future) - mask it
                        # Note: nc >= 1 always (frame 0 is always clean)
                        if nc < action_labels.shape[1]:  # action_labels is [B, N-1]
                            masked_action_labels[b, nc:] = 0
                    action_embeds = self.generator.model.action_embedding(masked_action_labels)
                else:
                    action_embeds = self.generator.model.action_embedding(action_labels)
                
                # For temporal_action_cross_attn: keep N-1 actions (no padding needed)
                # action[i] produces frame i+1, so it gets RoPE position i+1
                # This alignment is handled in WanTemporalActionCrossAttention with position offset
                # Frame 0 has no corresponding action (it's the initial frame)
                # Frames 1..N-1 attend to actions 0..N-2 via RoPE alignment
            
            # Replace conditional_dict with action embeddings
            conditional_dict = {"prompt_embeds": action_embeds}
            # For classifier-free guidance, we'd use a null class embedding
            # For now, use same embeddings (no CFG for action conditioning)
            unconditional_dict = conditional_dict
        
        # Handle SPATIAL ROPE multi-subject conditioning (3D RoPE in cross-attention)
        if getattr(self, 'use_spatial_rope_conditioning', False) and action_labels is not None and subject_ids is not None and initial_positions is not None:
            # action_labels: [B, S, T] - per-subject action sequences
            # subject_ids: [B, S] - subject indices
            # initial_positions: [B, S, 2] - initial (x, y) for each subject
            
            # For AR training: mask future actions
            if ar_training_mode and num_clean is not None:
                B, S, T = action_labels.shape
                action_mask = torch.ones_like(action_labels, dtype=torch.float32)
                for b in range(batch_size):
                    nc = num_clean[b].item()
                    if nc < T:
                        action_mask[b, :, nc:] = 0
            else:
                action_mask = None
            
            # Get embeddings AND positions: [B, S*T, model_dim], [B, S*T, 3]
            # For relative action spaces (e.g., 7 actions with orientation), pass frame_coords
            # so the conditioner uses GT positions instead of computing from action deltas
            conditioner_kwargs = dict(
                action_labels=action_labels,
                subject_ids=subject_ids,
                initial_positions=initial_positions,
                action_mask=action_mask,
            )
            if getattr(self, 'use_gt_positions_for_rope', False) and frame_coords is not None:
                conditioner_kwargs['frame_coords'] = frame_coords
            action_embeds, context_positions = self.multi_subject_conditioner(**conditioner_kwargs)
            
            action_embeds = action_embeds.to(dtype=clean_latent.dtype)
            context_positions = context_positions.to(dtype=clean_latent.dtype)
            
            # Pass BOTH embeddings and positions to conditional_dict
            conditional_dict = {
                "prompt_embeds": action_embeds,
                "context_positions": context_positions,
            }
            # Pass pre-encoded game text for video token cross-attention
            if video_text_embeds is not None:
                conditional_dict["video_text_embeds"] = video_text_embeds
            unconditional_dict = conditional_dict
        
        # Handle POSITION-AWARE multi-subject conditioning (with x,y position encoding)
        elif getattr(self, 'use_position_aware_conditioning', False) and action_labels is not None and subject_ids is not None and initial_positions is not None:
            # action_labels: [B, S, T] - per-subject action sequences
            # subject_ids: [B, S] - subject indices
            # initial_positions: [B, S, 2] - initial (x, y) for each subject
            
            # For AR training: mask future actions
            if ar_training_mode and num_clean is not None:
                B, S, T = action_labels.shape
                action_mask = torch.ones_like(action_labels, dtype=torch.float32)
                for b in range(batch_size):
                    nc = num_clean[b].item()
                    if nc < T:
                        action_mask[b, :, nc:] = 0
            else:
                action_mask = None
            
            # Get position-aware action embeddings: [B, S*T, model_dim]
            action_embeds = self.multi_subject_conditioner(
                action_labels=action_labels,
                subject_ids=subject_ids,
                initial_positions=initial_positions,
                action_mask=action_mask
            )
            
            action_embeds = action_embeds.to(dtype=clean_latent.dtype)
            conditional_dict = {"prompt_embeds": action_embeds}
            unconditional_dict = conditional_dict
        
        # Handle SIMPLE or MODULATED multi-subject conditioning (learned embeddings for subjects)
        elif (getattr(self, 'use_simple_multi_subject_conditioning', False) or 
            getattr(self, 'use_modulated_multi_subject_conditioning', False)) and action_labels is not None and subject_ids is not None:
            # action_labels: [B, S, T] - per-subject action sequences
            # subject_ids: [B, S] - subject indices for each subject in the video
            
            # For AR training: mask future actions
            if ar_training_mode and num_clean is not None:
                # Create mask: [B, S, T] where 1=valid, 0=masked
                B, S, T = action_labels.shape
                action_mask = torch.ones_like(action_labels, dtype=torch.float32)
                for b in range(batch_size):
                    nc = num_clean[b].item()
                    # Mask actions that produce frames beyond the target
                    if nc < T:
                        action_mask[b, :, nc:] = 0
            else:
                action_mask = None
            
            # Get multi-subject action embeddings: [B, S*T, model_dim]
            action_embeds = self.multi_subject_conditioner(
                action_labels=action_labels,
                subject_ids=subject_ids,
                action_mask=action_mask
            )
            
            # Ensure dtype matches model (bfloat16 for mixed precision)
            action_embeds = action_embeds.to(dtype=clean_latent.dtype)
            
            # Replace conditional_dict
            conditional_dict = {"prompt_embeds": action_embeds}
            unconditional_dict = conditional_dict
        
        # Handle multi-subject conditioning with image pooling (original approach)
        elif getattr(self, 'use_multi_subject_conditioning', False) and action_labels is not None and subject_prototypes is not None:
            # action_labels: [B, S, T] - per-subject action sequences
            # subject_prototypes: [B, S, C, H, W] - VAE-encoded prototype images
            
            # For AR training: mask future actions
            if ar_training_mode and num_clean is not None:
                # Create mask: [B, S, T] where 1=valid, 0=masked
                B, S, T = action_labels.shape
                action_mask = torch.ones_like(action_labels, dtype=torch.float32)
                for b in range(batch_size):
                    nc = num_clean[b].item()
                    # Mask actions that produce frames beyond the target
                    if nc < T:
                        action_mask[b, :, nc:] = 0
            else:
                action_mask = None
            
            # Get multi-subject action embeddings: [B, S*T, model_dim]
            action_embeds = self.multi_subject_conditioner(
                subject_prototypes=subject_prototypes,
                action_labels=action_labels,
                action_mask=action_mask
            )
            
            # Ensure dtype matches model (bfloat16 for mixed precision)
            action_embeds = action_embeds.to(dtype=clean_latent.dtype)
            
            # Replace conditional_dict
            conditional_dict = {"prompt_embeds": action_embeds}
            unconditional_dict = conditional_dict
        
        if not ar_training_mode and initial_latent is not None:
            # For i2v (non-AR): handle noise on first frame based on conditioning mode
            i2v_noised_conditioning = getattr(self, 'i2v_noised_conditioning', False)
            if not i2v_noised_conditioning and not self.i2v_cross_attention_latent:
                # Clean conditioning mode (latent replacement): first frame has no noise
                # Cross-attention mode: let first frame be noised naturally, conditioning via attention
                noise[:, :1] = 0.0
        
        # Step 2: Sample a timestep and add noise to denoiser inputs
        if self.training_timesteps is not None:
            # Sample from specific timesteps
            num_timesteps = len(self.training_timesteps)
            index = torch.randint(0, num_timesteps, (image_or_video_shape[0], image_or_video_shape[1]), device=self.device)
            # NOTE: Use float32 for timesteps to avoid bfloat16 precision issues (e.g., 999 -> 1000)
            timestep = self.training_timesteps[index].to(dtype=torch.float32, device=self.device)
        else:
            # Random sampling from full range
            index = self._get_timestep(
                0,
                self.scheduler.num_train_timesteps,
                image_or_video_shape[0],
                image_or_video_shape[1],
                self.num_frame_per_block,
                uniform_timestep=False
            )
            # NOTE: Use float32 for timesteps to avoid bfloat16 precision issues
            timestep = self.scheduler.timesteps[index].to(dtype=torch.float32, device=self.device)
        
        # AR training mode: set timesteps for clean/target/future frames (per-batch varying)
        if ar_training_mode:
            max_timestep = self.scheduler.timesteps[0].item()  # First timestep is the max (e.g., 1000)
            
            for b in range(batch_size):
                nc = num_clean[b].item()
                # Clean context frames: t=0
                timestep[b, :nc] = 0
                # Target frame (index nc): keep the randomly sampled timestep
                # Future frames: t=max (full noise)
                if nc + 1 < num_frame:
                    timestep[b, nc + 1:] = max_timestep
        elif initial_latent is not None:
            # For i2v with clean conditioning (latent replacement): set first frame timestep to 0 (consistent with noise=0)
            # Cross-attention mode: let first frame have normal timestep, conditioning via attention
            i2v_noised_conditioning = getattr(self, 'i2v_noised_conditioning', False)
            if not i2v_noised_conditioning and not self.i2v_cross_attention_latent:
                timestep[:, :1] = 0
        
        noisy_latents = self.scheduler.add_noise(
            clean_latent.flatten(0, 1),
            noise.flatten(0, 1),
            timestep.flatten(0, 1)
        ).unflatten(0, (batch_size, num_frame))
        training_target = self.scheduler.training_target(clean_latent, noise, timestep)
        
        # Position token mode: add noise to position latents (same timestep as video)
        position_training_target = None
        if getattr(self, 'position_token_mode', False) and clean_position_latents is not None:
            if clean_position_latents.dim() == 6:
                # Multi-subject: [B, S, T, C, H, W]
                B_pos, S_pos, T_pos, C_pos, H_pos, W_pos = clean_position_latents.shape
                # Flatten B, S, T together for noise addition
                clean_flat = clean_position_latents.reshape(B_pos * S_pos * T_pos, C_pos, H_pos, W_pos)
                noise_flat = position_noise.reshape(B_pos * S_pos * T_pos, C_pos, H_pos, W_pos)
                # Expand timestep for all subjects: [B, T] -> [B*S, T] -> [B*S*T]
                timestep_expanded = timestep.unsqueeze(1).expand(-1, S_pos, -1).reshape(B_pos * S_pos, T_pos)
                timestep_flat = timestep_expanded.flatten(0, 1)
                
                noisy_flat = self.scheduler.add_noise(clean_flat, noise_flat, timestep_flat)
                noisy_position_latents = noisy_flat.reshape(B_pos, S_pos, T_pos, C_pos, H_pos, W_pos)
                
                position_training_target = self.scheduler.training_target(
                    clean_position_latents, position_noise, 
                    timestep.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, S_pos, T_pos, 1, 1, 1)
                )
            else:
                # Single subject: [B, T, C, H, W]
                noisy_position_latents = self.scheduler.add_noise(
                    clean_position_latents.flatten(0, 1),
                    position_noise.flatten(0, 1),
                    timestep.flatten(0, 1)
                ).unflatten(0, (batch_size, num_frame))
                position_training_target = self.scheduler.training_target(
                    clean_position_latents, position_noise, timestep
                )
        
        # Pixel position mode: add noise to position pixels (same timestep as video)
        position_pixel_training_target = None
        if getattr(self, 'pixel_position_mode', False) and clean_position_pixels is not None:
            # position_pixels: [B, S, T, H, W]
            B_pos, S_pos, T_pos, H_pos, W_pos = clean_position_pixels.shape
            # Add channel dim for scheduler: [B, S, T, H, W] -> [B*S*T, 1, H, W]
            clean_flat = clean_position_pixels.reshape(B_pos * S_pos * T_pos, 1, H_pos, W_pos)
            noise_flat = position_pixel_noise.reshape(B_pos * S_pos * T_pos, 1, H_pos, W_pos)
            # Expand timestep for all subjects: [B, T] -> [B*S, T] -> [B*S*T]
            timestep_expanded = timestep.unsqueeze(1).expand(-1, S_pos, -1).reshape(B_pos * S_pos, T_pos)
            timestep_flat = timestep_expanded.flatten(0, 1)
            
            noisy_flat = self.scheduler.add_noise(clean_flat, noise_flat, timestep_flat)
            noisy_position_pixels = noisy_flat.reshape(B_pos, S_pos, T_pos, H_pos, W_pos)
            
            position_pixel_training_target = self.scheduler.training_target(
                clean_position_pixels.reshape(B_pos * S_pos * T_pos, 1, H_pos, W_pos),
                position_pixel_noise.reshape(B_pos * S_pos * T_pos, 1, H_pos, W_pos),
                timestep_flat
            ).reshape(B_pos, S_pos, T_pos, H_pos, W_pos)

        # Step 3: Noise augmentation, also add small noise to clean context latents
        # (Only used for teacher_forcing mode, not for i2v causal diffusion)
        if self.noise_augmentation_max_timestep > 0:
            index_clean_aug = self._get_timestep(
                0,
                self.noise_augmentation_max_timestep,
                image_or_video_shape[0],
                image_or_video_shape[1],
                self.num_frame_per_block,
                uniform_timestep=False
            )
            timestep_clean_aug = self.scheduler.timesteps[index_clean_aug].to(dtype=self.dtype, device=self.device)
            clean_latent_aug = self.scheduler.add_noise(
                clean_latent.flatten(0, 1),
                noise.flatten(0, 1),
                timestep_clean_aug.flatten(0, 1)
            ).unflatten(0, (batch_size, num_frame))
        else:
            clean_latent_aug = clean_latent
            timestep_clean_aug = None

        # Pass initial_latent to generator for i2v cross-attention
        # The model will process it internally if i2v_cross_attention_latent is enabled
        first_frame_latent = None
        if self.i2v_cross_attention_latent and initial_latent is not None:
            # Extract first frame: [B, num_frames, C, H, W] -> [B, 1, C, H, W]
            first_frame_latent = initial_latent[:, :1]
            # Permute to Conv3d format: [B, 1, C, H, W] -> [B, C, 1, H, W]
            first_frame_latent = first_frame_latent.permute(0, 2, 1, 3, 4)

        # Compute loss
        coord_pred = None
        coord_flow_target = None  # For coord diffusion mode
        if getattr(self, 'coord_token_mode', False) and frame_coords is not None:
            # frame_coords: [B, T, S, 2] - GT coordinates normalized [0, 1]
            B_coord, T_coord, S_coord, _ = frame_coords.shape
            
            if getattr(self, 'coord_diffusion', False):
                # COORD DIFFUSION MODE: Add noise to target coords, predict flow
                # Same as video: context is clean, target is noisy
                # Use the SAME scheduler.add_noise as video for consistency!
                
                # Scale coordinates to match noise distribution
                # [0, 1] -> centered at 0 with scale matching N(0, 1)
                coord_scale = getattr(self, 'coord_scale_factor', 1.0)
                if coord_scale != 1.0:
                    frame_coords_scaled = (frame_coords - 0.5) * coord_scale
                else:
                    frame_coords_scaled = frame_coords
                
                noisy_coords = frame_coords_scaled.clone()
                coord_noise = torch.randn_like(frame_coords_scaled)
                
                if ar_training_mode and num_clean is not None:
                    # Add noise only to target frame (same timestep as video)
                    for b in range(batch_size):
                        nc = num_clean[b].item()
                        if nc < num_frame:
                            # Get timestep for target frame
                            t_target = timestep[b, nc:nc+1]  # Keep as tensor [1]
                            clean_coord = frame_coords_scaled[b, nc:nc+1]  # [1, S, 2] - SCALED
                            noise_coord = coord_noise[b, nc:nc+1]   # [1, S, 2]
                            
                            # Use scheduler.add_noise for EXACT same sigma as video
                            # Reshape for scheduler: [1, S, 2] -> [1, 2] per subject, then combine
                            for s in range(S_coord):
                                clean_s = clean_coord[:, s:s+1, :]  # [1, 1, 2]
                                noise_s = noise_coord[:, s:s+1, :]  # [1, 1, 2]
                                # add_noise expects [B, C, H, W], fake it as [1, 2, 1, 1]
                                clean_flat = clean_s.view(1, 2, 1, 1)
                                noise_flat = noise_s.view(1, 2, 1, 1)
                                noisy_flat = self.scheduler.add_noise(clean_flat, noise_flat, t_target)
                                noisy_coords[b, nc, s] = noisy_flat.view(2)
                            
                            # Future frames: pure noise
                            if nc + 1 < num_frame:
                                noisy_coords[b, nc+1:] = coord_noise[b, nc+1:]
                
                # Compute flow target: noise - clean_scaled (in scaled space)
                coord_flow_target = coord_noise - frame_coords_scaled  # [B, T, S, 2]
                
                gen_output = self.generator(
                    noisy_image_or_video=noisy_latents,  # Model skips video if coord_only_no_video
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    clean_x=clean_latent_aug if self.teacher_forcing else None,
                    aug_t=timestep_clean_aug if self.teacher_forcing else None,
                    first_frame_latent=first_frame_latent,
                    subject_coords=noisy_coords,  # [B, T, S, 2] with target noisy
                    coord_num_clean=num_clean,  # For spatial RoPE: use last clean position for noisy frames
                    zero_position_for_masked=getattr(self, 'zero_position_for_masked', False),
                )
            else:
                # DIRECT PREDICTION MODE: Mask coords for target, predict directly
                masked_coords = frame_coords.clone()
                if ar_training_mode and num_clean is not None:
                    for b in range(batch_size):
                        nc = num_clean[b].item()
                        if nc < num_frame:
                            masked_coords[b, nc:] = 0.0
                
                gen_output = self.generator(
                    noisy_image_or_video=noisy_latents,  # Model skips video if coord_only_no_video
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    clean_x=clean_latent_aug if self.teacher_forcing else None,
                    aug_t=timestep_clean_aug if self.teacher_forcing else None,
                    first_frame_latent=first_frame_latent,
                    subject_coords=masked_coords,  # [B, T, S, 2] with future coords masked
                    coord_num_clean=num_clean,  # For spatial RoPE: use last clean position for noisy frames
                    zero_position_for_masked=getattr(self, 'zero_position_for_masked', False),
                )
            
            # Unpack: (flow_pred, x0_pred, coord_pred)
            if len(gen_output) == 3:
                flow_pred, x0_pred, coord_pred = gen_output
            else:
                flow_pred, x0_pred = gen_output
        # Position token mode: pass noisy position latents to generator
        elif getattr(self, 'position_token_mode', False) and noisy_position_latents is not None:
            # noisy_position_latents can be:
            # - [B, S, T, C, H, W] for multi-subject
            # - [B, T, C, H, W] for single subject
            gen_output = self.generator(
                noisy_image_or_video=noisy_latents,
                conditional_dict=conditional_dict,
                timestep=timestep,
                clean_x=clean_latent_aug if self.teacher_forcing else None,
                aug_t=timestep_clean_aug if self.teacher_forcing else None,
                first_frame_latent=first_frame_latent,
                position_latents=noisy_position_latents,
            )
            # Unpack: (flow_pred_video, x0_pred_video, flow_pred_position, x0_pred_position)
            if len(gen_output) == 4:
                flow_pred, x0_pred, position_flow_pred, position_x0_pred = gen_output
            else:
                # Fallback if model doesn't return position outputs
                flow_pred, x0_pred = gen_output[:2]
                position_flow_pred = None
                position_x0_pred = None
        # Pixel position mode: pass noisy position pixels to generator
        elif getattr(self, 'pixel_position_mode', False) and noisy_position_pixels is not None:
            # noisy_position_pixels: [B, S, T, H, W]
            gen_output = self.generator(
                noisy_image_or_video=noisy_latents,
                conditional_dict=conditional_dict,
                timestep=timestep,
                clean_x=clean_latent_aug if self.teacher_forcing else None,
                aug_t=timestep_clean_aug if self.teacher_forcing else None,
                first_frame_latent=first_frame_latent,
                position_pixels=noisy_position_pixels,
            )
            # Unpack: (flow_pred_video, x0_pred_video, flow_pred_position, x0_pred_position)
            if len(gen_output) == 4:
                flow_pred, x0_pred, position_flow_pred, position_x0_pred = gen_output
            else:
                # Fallback if model doesn't return position outputs
                flow_pred, x0_pred = gen_output[:2]
                position_flow_pred = None
                position_x0_pred = None
        else:
            flow_pred, x0_pred = self.generator(
                noisy_image_or_video=noisy_latents,
                conditional_dict=conditional_dict,
                timestep=timestep,
                clean_x=clean_latent_aug if self.teacher_forcing else None,
                aug_t=timestep_clean_aug if self.teacher_forcing else None,
                first_frame_latent=first_frame_latent  # For i2v cross-attention
            )
            position_flow_pred = None
            position_x0_pred = None
        
        # Position token mode: compute position loss
        loss_position_token = None
        if getattr(self, 'position_token_mode', False) and position_flow_pred is not None and position_training_target is not None:
            # Flow matching MSE loss on position tokens (same as video)
            # Handle both single subject [B, T, C, H, W] and multi-subject [B, S, T, C, H, W]
            if position_flow_pred.dim() == 6:
                # Multi-subject: [B, S, T, C, H, W]
                # Mean over C, H, W dimensions
                loss_position_token = torch.nn.functional.mse_loss(
                    position_flow_pred.float(), position_training_target.float(), reduction='none'
                ).mean(dim=(3, 4, 5))  # Mean over C, H, W -> [B, S, T]
                # Mean over subjects to get [B, T]
                loss_position_token = loss_position_token.mean(dim=1)
            else:
                # Single subject: [B, T, C, H, W]
                loss_position_token = torch.nn.functional.mse_loss(
                    position_flow_pred.float(), position_training_target.float(), reduction='none'
                ).mean(dim=(2, 3, 4))  # Mean over C, H, W -> [B, T]
            
            loss_position_token = loss_position_token * self.scheduler.training_weight(timestep).unflatten(0, (batch_size, num_frame))
        
        # Pixel position mode: compute position loss
        if getattr(self, 'pixel_position_mode', False) and position_flow_pred is not None and position_pixel_training_target is not None:
            # Flow matching MSE loss on pixel position maps
            # position_flow_pred: [B, S, T, H, W]
            # position_pixel_training_target: [B, S, T, H, W]
            loss_position_token = torch.nn.functional.mse_loss(
                position_flow_pred.float(), position_pixel_training_target.float(), reduction='none'
            ).mean(dim=(3, 4))  # Mean over H, W -> [B, S, T]
            # Mean over subjects to get [B, T]
            loss_position_token = loss_position_token.mean(dim=1)
            loss_position_token = loss_position_token * self.scheduler.training_weight(timestep).unflatten(0, (batch_size, num_frame))
        
        # VideoJAM mode: standard flow matching loss on both video and position latents
        if getattr(self, 'videojam_mode', False) and position_maps is not None:
            video_channels = self.generator.model.in_dim  # Original video channels (16)
            
            # Split predictions (both are 16 channels in VAE latent space)
            flow_pred_video = flow_pred[:, :, :video_channels]
            flow_pred_position = flow_pred[:, :, video_channels:]
            
            # Split targets
            target_video = training_target[:, :, :video_channels]
            target_position = training_target[:, :, video_channels:]
            
            # Compute video loss (standard flow matching MSE)
            loss_video = torch.nn.functional.mse_loss(
                flow_pred_video.float(), target_video.float(), reduction='none'
            ).mean(dim=(2, 3, 4))
            loss_video = loss_video * self.scheduler.training_weight(timestep).unflatten(0, (batch_size, num_frame))
            
            # Compute position loss (standard flow matching MSE - same loss as video)
            # Both are in the same VAE latent space
            loss_position = torch.nn.functional.mse_loss(
                flow_pred_position.float(), target_position.float(), reduction='none'
            ).mean(dim=(2, 3, 4))
            loss_position = loss_position * self.scheduler.training_weight(timestep).unflatten(0, (batch_size, num_frame))
        
        # Legacy joint position diffusion: segmentation-style losses on raw position maps
        elif getattr(self, 'joint_position_diffusion', False) and position_maps is not None:
            video_channels = self.generator.model.in_dim  # Original video channels (16)
            
            # Split predictions
            flow_pred_video = flow_pred[:, :, :video_channels]
            flow_pred_position = flow_pred[:, :, video_channels:]
            
            # Split targets
            target_video = training_target[:, :, :video_channels]
            target_position = training_target[:, :, video_channels:]
            
            # Compute video loss
            loss_video = torch.nn.functional.mse_loss(
                flow_pred_video.float(), target_video.float(), reduction='none'
            ).mean(dim=(2, 3, 4))
            loss_video = loss_video * self.scheduler.training_weight(timestep).unflatten(0, (batch_size, num_frame))
            
            # Compute position loss - use x0_pred (reconstructed) instead of flow for better loss
            # Extract predicted and GT position maps
            x0_pred_position = x0_pred[:, :, video_channels:]  # Predicted clean position maps
            gt_position_maps = position_maps  # GT position maps [B, T, S, H, W]
            
            # All segmentation losses use sigmoid and treat output as logits
            # At inference, apply sigmoid to position channels!
            gt = gt_position_maps.float().clamp(0, 1)
            pred_logits = x0_pred_position.float()
            
            if self.position_loss_type == "dice":
                # Dice Loss - gold standard for sparse segmentation
                # Dice = 2 * intersection / (pred + gt)
                pred_prob = torch.sigmoid(pred_logits)
                
                pred_flat = pred_prob.view(batch_size, num_frame, self.position_channels, -1)
                gt_flat = gt.view(batch_size, num_frame, self.position_channels, -1)
                
                intersection = (pred_flat * gt_flat).sum(dim=-1)
                union = pred_flat.sum(dim=-1) + gt_flat.sum(dim=-1)
                smooth = 1e-5
                dice = (2.0 * intersection + smooth) / (union + smooth)
                loss_position = (1.0 - dice).mean(dim=2)
                
            elif self.position_loss_type == "dice_bce":
                # Dice + BCE combo - best of both worlds
                pred_prob = torch.sigmoid(pred_logits)
                
                # Dice component
                pred_flat = pred_prob.view(batch_size, num_frame, self.position_channels, -1)
                gt_flat = gt.view(batch_size, num_frame, self.position_channels, -1)
                intersection = (pred_flat * gt_flat).sum(dim=-1)
                union = pred_flat.sum(dim=-1) + gt_flat.sum(dim=-1)
                smooth = 1e-5
                dice = (2.0 * intersection + smooth) / (union + smooth)
                dice_loss = (1.0 - dice).mean(dim=2)
                
                # BCE with pos_weight (100x for blob region given ~1% positive pixels)
                pos_weight = torch.ones_like(gt) * self.position_foreground_weight
                bce = torch.nn.functional.binary_cross_entropy_with_logits(
                    pred_logits, gt, pos_weight=pos_weight, reduction='none'
                ).mean(dim=(2, 3, 4))
                
                loss_position = dice_loss + bce
                
            elif self.position_loss_type == "focal":
                # Focal loss: downweight easy zeros, upweight hard examples
                gamma = 2.0
                alpha = 0.75  # Higher weight for positive class
                pred_prob = torch.sigmoid(pred_logits)
                eps = 1e-7
                pred_prob = pred_prob.clamp(eps, 1 - eps)
                
                # Focal loss with alpha balancing
                bce = -gt * torch.log(pred_prob) - (1 - gt) * torch.log(1 - pred_prob)
                p_t = pred_prob * gt + (1 - pred_prob) * (1 - gt)
                alpha_t = alpha * gt + (1 - alpha) * (1 - gt)
                focal_weight = alpha_t * (1 - p_t) ** gamma
                loss_position = (focal_weight * bce).mean(dim=(2, 3, 4))
                
            elif self.position_loss_type == "bce":
                # BCE loss with positive class weight
                pos_weight = torch.ones_like(gt) * self.position_foreground_weight
                loss_position = torch.nn.functional.binary_cross_entropy_with_logits(
                    pred_logits, gt, pos_weight=pos_weight, reduction='none'
                ).mean(dim=(2, 3, 4))
                
            elif self.position_loss_type == "weighted_mse":
                # Weighted MSE: upweight the blob region
                threshold = 0.1
                foreground_mask = (gt_position_maps > threshold).float()
                weight = 1.0 + (self.position_foreground_weight - 1.0) * foreground_mask
                squared_error = (x0_pred_position.float() - gt_position_maps.float()) ** 2
                weighted_error = squared_error * weight
                loss_position = weighted_error.mean(dim=(2, 3, 4))
                
            else:
                # Default MSE on reconstructed position maps
                loss_position = torch.nn.functional.mse_loss(
                    x0_pred_position.float(), gt_position_maps.float(), reduction='none'
                ).mean(dim=(2, 3, 4))
            
            loss_position = loss_position * self.scheduler.training_weight(timestep).unflatten(0, (batch_size, num_frame))
        else:
            # Standard loss computation
            if getattr(self, 'coord_only_mode', False):
                # COORD ONLY MODE: Skip video loss entirely (we're passing zeros anyway)
                loss_video = torch.zeros(batch_size, num_frame, device=self.device)
            else:
                loss_video = torch.nn.functional.mse_loss(
                    flow_pred.float(), training_target.float(), reduction='none'
                ).mean(dim=(2, 3, 4))
                loss_video = loss_video * self.scheduler.training_weight(timestep).unflatten(0, (batch_size, num_frame))
            loss_position = None
        
        # Coordinate token mode: compute loss on predicted coordinates
        loss_coord = None
        if getattr(self, 'coord_token_mode', False) and coord_pred is not None and frame_coords is not None:
            # coord_pred: [B, T*S, 2] - predicted coordinates (or flow if diffusion mode)
            # frame_coords: [B, T, S, 2] - GT coordinates
            B, T, S, _ = frame_coords.shape
            
            if getattr(self, 'coord_diffusion', False) and coord_flow_target is not None:
                # FLOW MATCHING LOSS: predict flow (noise - clean)
                # coord_pred is the predicted flow, coord_flow_target is GT flow
                gt_flow = coord_flow_target.view(B, T * S, 2)
                loss_coord = torch.nn.functional.mse_loss(
                    coord_pred.float(), gt_flow.float(), reduction='none'
                ).mean(dim=-1)  # [B, T*S]
            else:
                # DIRECT PREDICTION: MSE on coordinates
                gt_coords = frame_coords.view(B, T * S, 2)
                loss_coord = torch.nn.functional.mse_loss(
                    coord_pred.float(), gt_coords.float(), reduction='none'
                ).mean(dim=-1)  # [B, T*S]
            
            # Reshape to [B, T, S] for per-subject loss logging
            loss_coord_per_subject = loss_coord.view(B, T, S)  # [B, T, S]
            
            # Store per-subject loss before averaging (for logging)
            self._loss_coord_per_subject = loss_coord_per_subject.detach()
            
            # Average over subjects for total loss, masking padded subjects if applicable
            if subject_mask is not None:
                # subject_mask: [B, S] boolean, True for real subjects
                mask_float = subject_mask.float().unsqueeze(1)  # [B, 1, S]
                loss_coord_per_subject = loss_coord_per_subject * mask_float
                num_real = mask_float.sum(dim=-1).clamp(min=1)  # [B, 1]
                loss_coord = loss_coord_per_subject.sum(dim=-1) / num_real  # [B, T] / [B, 1] broadcasts correctly
            else:
                loss_coord = loss_coord_per_subject.mean(dim=-1)  # [B, T]
            
            # Apply training weight (same as video) for diffusion mode
            # Can be disabled with coord_uniform_weight: true (better for coordinates)
            if getattr(self, 'coord_diffusion', False) and not getattr(self, 'coord_uniform_weight', False):
                training_weights = self.scheduler.training_weight(timestep).unflatten(0, (batch_size, num_frame))
                loss_coord = loss_coord * training_weights
        
        # AR training mode: only compute loss on the target frame (index num_clean[b] per batch)
        if ar_training_mode:
            # Create gradient mask: only target frame (index num_clean[b]) has loss
            gradient_mask = torch.zeros((batch_size, num_frame), device=self.device, dtype=torch.float32)
            for b in range(batch_size):
                gradient_mask[b, num_clean[b]] = 1.0
            loss_video = loss_video * gradient_mask
            if loss_position is not None:
                loss_position = loss_position * gradient_mask
            if loss_coord is not None:
                loss_coord = loss_coord * gradient_mask
            if loss_position_token is not None:
                loss_position_token = loss_position_token * gradient_mask
        elif initial_latent is not None and not self.i2v_cross_attention_latent:
            # Mask loss for first frame in i2v mode (ONLY for latent replacement modes)
            # Cross-attention mode: train on all frames including first
            # Create gradient mask: False for first frame, True for rest
            gradient_mask = torch.ones((batch_size, num_frame), device=self.device, dtype=torch.bool)
            gradient_mask[:, :1] = False
            loss_video = loss_video * gradient_mask.float()
            if loss_position is not None:
                loss_position = loss_position * gradient_mask.float()
            if loss_coord is not None:
                loss_coord = loss_coord * gradient_mask.float()
            if loss_position_token is not None:
                loss_position_token = loss_position_token * gradient_mask.float()
        
        # Combine losses
        if getattr(self, 'coord_only_mode', False) and loss_coord is not None:
            # COORD ONLY MODE: Skip video loss, only train coordinate decoder
            loss = self.coord_loss_weight * loss_coord.mean()
        else:
            loss = loss_video.mean()
            if loss_position is not None:
                loss = loss + self.position_loss_weight * loss_position.mean()
            if loss_coord is not None:
                loss = loss + self.coord_loss_weight * loss_coord.mean()
            if loss_position_token is not None:
                loss = loss + self.position_loss_weight * loss_position_token.mean()

        # For logging, use video-only clean_latent if position maps were used
        log_clean_latent = clean_latent_video if getattr(self, 'joint_position_diffusion', False) and 'clean_latent_video' in dir() else clean_latent
        log_x0_pred = x0_pred[:, :, :self.generator.model.in_dim] if getattr(self, 'joint_position_diffusion', False) else x0_pred

        log_dict = {
            "x0": log_clean_latent.detach() if 'clean_latent_video' in locals() else clean_latent.detach(),
            "x0_pred": log_x0_pred.detach() if getattr(self, 'joint_position_diffusion', False) else x0_pred.detach()
        }
        
        # Always log video loss unless in coord_only_mode
        if not getattr(self, 'coord_only_mode', False):
            log_dict["loss_video"] = loss_video.mean().item()
        
        # Log additional losses if present
        if loss_position is not None:
            log_dict["loss_position"] = loss_position.mean().item()
        if loss_position_token is not None:
            log_dict["loss_position_token"] = loss_position_token.mean().item()
        if loss_coord is not None:
            log_dict["loss_coord"] = loss_coord.mean().item()
            
            # Log per-subject loss if enabled
            if getattr(self, 'log_per_subject_loss', False) and hasattr(self, '_loss_coord_per_subject'):
                loss_per_subj = self._loss_coord_per_subject  # [B, T, S]
                S = loss_per_subj.shape[-1]
                for s in range(S):
                    log_dict[f"loss_coord_s{s}"] = loss_per_subj[:, :, s].mean().item()
        
        # Log timestep for the target frame (for debugging)
        if ar_training_mode and num_clean is not None:
            # Get timestep of target frame for first batch item
            target_t = timestep[0, num_clean[0].item()].item()
            log_dict["timestep"] = target_t
        
        return loss, log_dict
