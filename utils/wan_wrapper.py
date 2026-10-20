"""High-level wrappers around the Wan DiT, VAE and T5 text encoder.

The main additions on top of the upstream Self-Forcing wrappers are the
coord-token specific branches in :class:`WanDiffusionWrapper`: passing
coord tokens through to ``WanModel._forward``, optional spatial-RoPE
cross-attention, and subject-isolation masks in self-attention. Inference helpers live in the
same class so that ``scripts/inference/inference_all_games_coord.py`` can run
the model end-to-end on a Melting Pot scene.
"""
import types
from typing import List, Optional
import torch
from torch import nn

from utils.scheduler import SchedulerInterface, FlowMatchScheduler
from wan.modules.tokenizers import HuggingfaceTokenizer
from wan.modules.model import WanModel, RegisterTokens, GanAttentionBlock
from wan.modules.vae import _video_vae
from wan.modules.t5 import umt5_xxl
from wan.modules.causal_model import CausalWanModel


class WanTextEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.text_encoder = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=torch.float32,
            device=torch.device('cpu')
        ).eval().requires_grad_(False)
        self.text_encoder.load_state_dict(
            torch.load("wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
                       map_location='cpu', weights_only=False)
        )

        self.tokenizer = HuggingfaceTokenizer(
            name="wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl/", seq_len=512, clean='whitespace')

    @property
    def device(self):
        # Assume we are always on GPU
        return torch.cuda.current_device()

    def forward(self, text_prompts: List[str]) -> dict:
        ids, mask = self.tokenizer(
            text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)

        for u, v in zip(context, seq_lens):
            u[v:] = 0.0  # set padding to 0.0

        return {
            "prompt_embeds": context
        }


class WanVAEWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

        # init model
        self.model = _video_vae(
            pretrained_path="wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
            z_dim=16,
        ).eval().requires_grad_(False)

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        # pixel: [batch_size, num_channels, num_frames, height, width]
        device, dtype = pixel.device, pixel.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        # Process entire batch at once (much faster than looping!)
        output = self.model.encode(pixel, scale).float()
        
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        # from [batch_size, num_frames, num_channels, height, width]
        # to [batch_size, num_channels, num_frames, height, width]
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        if use_cache:
            decode_function = self.model.cached_decode
        else:
            decode_function = self.model.decode

        output = []
        for u in zs:
            output.append(decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0))
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output


class WanDiffusionWrapper(torch.nn.Module):
    def __init__(
            self,
            model_name="Wan2.1-T2V-1.3B",
            timestep_shift=8.0,
            is_causal=False,
            local_attn_size=-1,
            sink_size=0,
            random_init=False,
            num_layers=None,
            dim=None,
            ffn_dim=None,
            num_heads=None,
            in_dim=None,
            out_dim=None,
            text_dim=None,
            cross_attn_type=None,  # Cross-attention type: None, 't2v_cross_attn', 'i2v_cross_attn', 'temporal_action_cross_attn'
            position_channels=0,  # Joint position diffusion: number of position map channels
            videojam_mode=False,  # True VideoJAM mode: VAE-encoded position latents (16ch)
            videojam_position_residual=True,  # Add position embeddings as residual at output (zero-init mode)
            videojam_identity_init=False,  # Use [I,0] init for Win+/Wout+ (no residuals needed)
            coord_token_mode=False,  # Coordinate token mode: (x,y) per subject as sequence tokens
            coord_temporal_rope=False,  # Apply temporal RoPE to coordinate tokens
            coord_spatial_rope=False,  # Apply spatial RoPE based on actual coordinate position
            coord_layernorm=False,  # Apply LayerNorm after coord_encoder
            coord_fourier_encoding=False,  # Use fixed Fourier encoding instead of learned linear
            coord_fourier_frequencies=10,  # Number of frequency bands for Fourier encoding
            coord_direct_embedding=False,  # Direct embedding: pad (x,y) to dim, no encoder/decoder
            coord_no_encoder_decoder=False,  # Pure slicing: no learned encoder/decoder
            coord_scale_factor=1.0,  # Scale factor for coord centering/normalization
            coord_only_no_video=False,  # Coord-only mode: no video input, only coordinate tokens
            num_subjects=2,  # Number of subjects for coord_token_mode
            per_frame_timesteps=False,  # Enable per-frame timestep modulation for AR training
            position_token_mode=False,  # Position token mode: position maps as sequential tokens with 3D RoPE
            position_temporal_offset=0.0,  # Temporal RoPE offset for position tokens (0.1 recommended to distinguish from video)
            subject_isolation_self_attn=False,  # If True, isolate position tokens per subject in self-attention
            position_no_video_attn=False,  # If True, position tokens cannot attend to video (only own subject's position)
            position_skip_self_attn=False,  # If True, position tokens skip self-attention entirely (pure residual)
            position_no_self_attn_update=False,  # If True, position tokens don't update from self-attn but video can see them
            use_flex_attention=True,  # If True, use FlexAttention for masked self-attention (faster)
            pixel_position_mode=False,  # If True, diffuse pixel-space position maps (not VAE encoded)
            pixel_position_num_subjects=2,  # Number of subjects/channels for pixel position maps
            separate_position_modulation=False,  # If True, use separate adaLN modulation for position tokens
            cross_attn_rope=False,  # If True, apply 3D RoPE in cross-attention for spatial correspondence
            per_frame_cross_attn=False,  # If True, position patches at frame t only see action[t] (not all frames)
            cross_attn_subject_isolation=True,  # If True, coord tokens attend only to their own subject's actions in cross-attn
            image_or_video_shape=None,  # [B, T, C, H, W] for seq_len calculation
            **kwargs  # Capture any extra kwargs
    ):
        super().__init__()
        
        # Store image_or_video_shape for seq_len calculation
        self.image_or_video_shape = image_or_video_shape
        
        # Store position_channels for later use
        self.position_channels = position_channels
        self.videojam_mode = videojam_mode
        self.videojam_position_residual = videojam_position_residual
        self.videojam_identity_init = videojam_identity_init
        self.coord_token_mode = coord_token_mode
        self.coord_temporal_rope = coord_temporal_rope
        self.coord_spatial_rope = coord_spatial_rope
        self.coord_layernorm = coord_layernorm
        self.coord_fourier_encoding = coord_fourier_encoding
        self.coord_fourier_frequencies = coord_fourier_frequencies
        self.coord_direct_embedding = coord_direct_embedding
        self.coord_no_encoder_decoder = coord_no_encoder_decoder
        self.coord_only_no_video = coord_only_no_video
        self.num_subjects = num_subjects
        self.per_frame_timesteps = per_frame_timesteps
        self.position_token_mode = position_token_mode
        self.position_temporal_offset = position_temporal_offset
        self.subject_isolation_self_attn = subject_isolation_self_attn
        self.position_no_video_attn = position_no_video_attn
        self.position_skip_self_attn = position_skip_self_attn
        self.position_no_self_attn_update = position_no_self_attn_update
        self.use_flex_attention = use_flex_attention
        self.pixel_position_mode = pixel_position_mode
        self.pixel_position_num_subjects = pixel_position_num_subjects
        self.separate_position_modulation = separate_position_modulation
        self.cross_attn_rope = cross_attn_rope
        self.per_frame_cross_attn = per_frame_cross_attn
        self.cross_attn_subject_isolation = cross_attn_subject_isolation
        
        if kwargs:
            print(f"Warning: Unexpected kwargs passed to WanDiffusionWrapper: {kwargs}")

        if random_init:
            # Random initialization - create model directly with specified parameters (no downloads needed)
            # Use user-provided params or defaults
            final_num_layers = num_layers if num_layers is not None else 30
            final_dim = dim if dim is not None else 1536
            final_ffn_dim = ffn_dim if ffn_dim is not None else 8960
            final_num_heads = num_heads if num_heads is not None else 12
            final_in_dim = in_dim if in_dim is not None else 16
            final_out_dim = out_dim if out_dim is not None else 16
            final_text_dim = text_dim if text_dim is not None else final_dim  # Default to dim if not specified
            
            print(f"Creating model with: num_layers={final_num_layers}, dim={final_dim}, ffn_dim={final_ffn_dim}, num_heads={final_num_heads}, in_dim={final_in_dim}, out_dim={final_out_dim}, text_dim={final_text_dim}")
            
            # Create model directly - no config file needed
            if is_causal:
                self.model = CausalWanModel(
                    model_type='t2v',
                    patch_size=(1, 2, 2),
                    text_len=512,
                    in_dim=final_in_dim,
                    dim=final_dim,
                    ffn_dim=final_ffn_dim,
                    freq_dim=256,
                    text_dim=final_text_dim,  # Use configurable text_dim
                    out_dim=final_out_dim,
                    num_heads=final_num_heads,
                    num_layers=final_num_layers,
                    local_attn_size=local_attn_size,
                    sink_size=sink_size,
                    qk_norm=True,
                    cross_attn_norm=True,
                    eps=1e-6
                )
            else:
                self.model = WanModel(
                    model_type='t2v',
                    patch_size=(1, 2, 2),
                    text_len=512,
                    in_dim=final_in_dim,
                    dim=final_dim,
                    ffn_dim=final_ffn_dim,
                    freq_dim=256,
                    text_dim=final_text_dim,  # Use configurable text_dim
                    out_dim=final_out_dim,
                    num_heads=final_num_heads,
                    num_layers=final_num_layers,
                    window_size=(-1, -1),
                    qk_norm=True,
                    cross_attn_norm=True,
                    cross_attn_type=cross_attn_type,  # Custom cross-attention type
                    eps=1e-6,
                    position_token_mode=position_token_mode,
                    num_subjects=num_subjects,
                    cross_attn_rope=cross_attn_rope,
                    per_frame_cross_attn=per_frame_cross_attn,
                    cross_attn_subject_isolation=cross_attn_subject_isolation,
                )
            print(f"Initialized with RANDOM weights (no pretrained model needed)")
            print(f"  num_layers={final_num_layers}, dim={final_dim}, ffn_dim={final_ffn_dim}, num_heads={final_num_heads}, in_dim={final_in_dim}, out_dim={final_out_dim}, text_dim={final_text_dim}")
            if cross_attn_type:
                print(f"  cross_attn_type={cross_attn_type}")
        else:
            # Load pretrained weights
            # Support both local paths (wan_models/...) and HuggingFace repos (Wan-AI/...)
            import os
            if os.path.exists(f"wan_models/{model_name}"):
                model_path = f"wan_models/{model_name}/"
            elif "/" in model_name:  # Assume it's a HF repo like "Wan-AI/Wan2.1-T2V-1.3B"
                model_path = model_name
            else:
                model_path = f"wan_models/{model_name}/"
            
            if is_causal:
                self.model = CausalWanModel.from_pretrained(
                    model_path, local_attn_size=local_attn_size, sink_size=sink_size)
            else:
                # If custom cross_attn_type is specified, create model with that type
                # then load weights (with strict=False for cross-attention mismatches)
                if cross_attn_type is not None or position_channels > 0 or videojam_mode or coord_token_mode or position_token_mode or pixel_position_mode:
                    # Load config to get architecture params
                    import json
                    config_path = os.path.join(model_path, "config.json")
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    
                    # Create model with custom cross_attn_type and/or position_channels
                    self.model = WanModel(
                        model_type=config.get('model_type', 't2v'),
                        patch_size=tuple(config.get('patch_size', [1, 2, 2])),
                        text_len=config.get('text_len', 512),
                        in_dim=config.get('in_dim', 16),
                        dim=config.get('dim', 1536),
                        ffn_dim=config.get('ffn_dim', 8960),
                        freq_dim=config.get('freq_dim', 256),
                        text_dim=config.get('text_dim', 4096),
                        out_dim=config.get('out_dim', 16),
                        num_heads=config.get('num_heads', 12),
                        num_layers=config.get('num_layers', 30),
                        window_size=tuple(config.get('window_size', [-1, -1])),
                        qk_norm=config.get('qk_norm', True),
                        cross_attn_norm=config.get('cross_attn_norm', True),
                        cross_attn_type=cross_attn_type,
                        eps=config.get('eps', 1e-6),
                        position_channels=position_channels,  # Joint position diffusion (legacy)
                        videojam_mode=videojam_mode,  # True VideoJAM mode
                        videojam_position_residual=videojam_position_residual,  # Position residual fix
                        videojam_identity_init=videojam_identity_init,  # [I,0] initialization
                        coord_token_mode=coord_token_mode,  # Coordinate token mode
                        coord_temporal_rope=coord_temporal_rope,  # Temporal RoPE for coord tokens
                        coord_spatial_rope=coord_spatial_rope,  # Spatial RoPE for coord tokens
                        coord_layernorm=coord_layernorm,  # LayerNorm for coord tokens
                        coord_fourier_encoding=coord_fourier_encoding,  # Fourier encoding
                        coord_fourier_frequencies=coord_fourier_frequencies,  # Frequency bands
                        coord_direct_embedding=coord_direct_embedding,  # Direct embedding
                        coord_no_encoder_decoder=coord_no_encoder_decoder,  # Pure slicing
                        coord_scale_factor=coord_scale_factor,  # Scale factor for RoPE position computation
                        num_subjects=num_subjects,  # Number of subjects
                        per_frame_timesteps=per_frame_timesteps,  # Per-frame timestep modulation
                        position_token_mode=position_token_mode,  # Position token mode
                        position_temporal_offset=position_temporal_offset,  # Temporal RoPE offset for position tokens
                        subject_isolation_self_attn=subject_isolation_self_attn,  # Subject isolation in self-attention
                        position_no_video_attn=position_no_video_attn,  # Position tokens don't attend to video
                        position_skip_self_attn=position_skip_self_attn,  # Position tokens skip self-attention entirely
                        position_no_self_attn_update=position_no_self_attn_update,  # Position tokens don't update but video can see them
                        use_flex_attention=use_flex_attention,  # Use FlexAttention for masked self-attention
                        pixel_position_mode=pixel_position_mode,  # Pixel-space position diffusion
                        pixel_position_num_subjects=pixel_position_num_subjects,  # Number of subjects for pixel positions
                        separate_position_modulation=separate_position_modulation,  # Separate adaLN for position tokens
                        cross_attn_rope=cross_attn_rope,  # 3D RoPE in cross-attention for spatial correspondence
                        per_frame_cross_attn=per_frame_cross_attn,  # Per-frame masking in cross-attention
                        cross_attn_subject_isolation=cross_attn_subject_isolation,
                    )
                    # Set coord_only_no_video after creation
                    if coord_only_no_video:
                        self.model.coord_only_no_video = True
                        # Store latent dimensions for spatial RoPE (used when grid_sizes is 0,0,0)
                        self.model._coord_only_latent_h = 32
                        self.model._coord_only_latent_w = 32
                    if coord_token_mode:
                        features = []
                        if coord_temporal_rope:
                            features.append("temporal RoPE")
                        if coord_spatial_rope:
                            features.append("spatial RoPE")
                        if coord_layernorm:
                            features.append("LayerNorm")
                        if coord_fourier_encoding:
                            features.append(f"Fourier({coord_fourier_frequencies})")
                        if coord_direct_embedding:
                            features.append("direct")
                        features_str = f" with {', '.join(features)}" if features else ""
                        print(f"  Using coord_token_mode: {num_subjects} subjects as sequence tokens{features_str}")
                    elif position_token_mode:
                        sep_mod_str = ", SEPARATE modulation" if separate_position_modulation else ""
                        cross_rope_str = ", CROSS-ATTN RoPE" if cross_attn_rope else ""
                        per_frame_str = ", PER-FRAME cross-attn" if per_frame_cross_attn else ""
                        print(f"  Using POSITION TOKEN MODE: {num_subjects} subjects, 3D RoPE{sep_mod_str}{cross_rope_str}{per_frame_str}")
                    elif pixel_position_mode:
                        print(f"  Using PIXEL POSITION MODE: {pixel_position_num_subjects} subjects, diffusing pixel-space position maps")
                    elif videojam_mode:
                        if videojam_identity_init:
                            print(f"  Using VideoJAM mode with [I,0] identity initialization (no residuals)")
                        else:
                            residual_str = "with" if videojam_position_residual else "without"
                            print(f"  Using VideoJAM mode: zero-init {residual_str} position residual")
                    elif position_channels > 0:
                        print(f"  Using position_channels={position_channels} for joint position diffusion")
                    
                    # Load pretrained weights (strict=False because cross-attention layers differ)
                    from safetensors.torch import load_file
                    weights_path = os.path.join(model_path, "diffusion_pytorch_model.safetensors")
                    if os.path.exists(weights_path):
                        state_dict = load_file(weights_path)
                    else:
                        weights_path = os.path.join(model_path, "diffusion_pytorch_model.bin")
                        state_dict = torch.load(weights_path, map_location="cpu")
                    
                    missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
                    if missing:
                        print(f"  Note: {len(missing)} missing keys (expected for custom cross_attn_type)")
                    print(f"  Using custom cross_attn_type: {cross_attn_type}")
                else:
                    self.model = WanModel.from_pretrained(model_path)
            print(f"Loaded pretrained model from: {model_path}")
            
            # Set per_frame_timesteps on the model after loading pretrained weights
            # (from_pretrained doesn't accept this parameter, so we set it afterwards)
            if per_frame_timesteps:
                self.model.per_frame_timesteps = per_frame_timesteps
            # Note: Cannot override architecture params when loading pretrained weights
            if any(p is not None for p in [num_layers, dim, ffn_dim, num_heads]):
                print("  Warning: Architecture parameters ignored when loading pretrained weights")
            
            # Initialize position modulation from video modulation if enabled
            if separate_position_modulation and hasattr(self.model, 'init_position_modulation_from_video'):
                self.model.init_position_modulation_from_video()
                print(f"  Initialized separate position modulation from video modulation weights")
        self.model.eval()
        
        # For masked_position_cross_attn: pre-compute and set the T5-encoded "" null embedding
        if cross_attn_type == 'masked_position_cross_attn':
            self._init_null_embedding_for_masked_cross_attn()

        # For non-causal diffusion, all frames share the same timestep
        self.uniform_timestep = not is_causal

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        # Compute seq_len from image_or_video_shape if available
        # seq_len = num_frames * (H // patch_h) * (W // patch_w)
        # For 256x256 video: 9 frames * 16 * 16 = 2304 tokens
        # For 720p video: 21 frames * 60 * 104 / 4 = 32760 tokens
        if hasattr(self, 'image_or_video_shape') and self.image_or_video_shape is not None:
            # image_or_video_shape: [B, T, C, H, W] or similar
            shape = self.image_or_video_shape
            if len(shape) >= 4:
                T = shape[1]  # num frames (latent space)
                H = shape[-2]  # height (latent space)
                W = shape[-1]  # width (latent space)
                # Patch size is 2x2 for temporal, 2x2 for spatial
                self.seq_len = T * (H // 2) * (W // 2)
                print(f"  Computed seq_len: {self.seq_len} ({T} frames × {H//2}×{W//2} patches)")
            else:
                self.seq_len = 32760  # Default for 720p
        else:
            self.seq_len = 32760  # Default for 720p [1, 21, 16, 60, 104]
        self.post_init()

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()
    
    def _init_null_embedding_for_masked_cross_attn(self) -> None:
        """
        Pre-compute the T5-encoded "" null embedding for masked cross-attention.
        This is the same null embedding used in CFG (classifier-free guidance).
        Video tokens will attend to this instead of zeros.
        """
        try:
            # Load T5 encoder and compute embedding for ""
            text_encoder = WanTextEncoder()
            text_encoder = text_encoder.cuda() if torch.cuda.is_available() else text_encoder
            
            with torch.no_grad():
                null_embed_dict = text_encoder([''])
                null_embed = null_embed_dict['prompt_embeds']  # [1, 512, 4096]
                
            # Move to CPU for storage (will be moved to device when used)
            null_embed = null_embed.cpu()
            
            # Set it for all masked cross-attention blocks
            if hasattr(self.model, 'set_null_embedding_for_cross_attn'):
                self.model.set_null_embedding_for_cross_attn(null_embed)
            else:
                print("  Warning: Model doesn't have set_null_embedding_for_cross_attn method")
                
            # Clean up text encoder
            del text_encoder
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"  Warning: Could not init null embedding, using zeros fallback: {e}")
    
    def set_video_cross_attn_embedding(self, prompt: str, text_encoder=None) -> bool:
        """
        Set the cross-attention embedding for video tokens to a specific prompt.
        This allows video tokens to attend to a meaningful text embedding instead of null.
        
        Args:
            prompt: The text prompt to use for video token cross-attention
            text_encoder: Optional text encoder (if None, will load temporarily)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load text encoder if not provided
            cleanup_encoder = False
            if text_encoder is None:
                text_encoder = WanTextEncoder()
                cleanup_encoder = True
            
            # Move text encoder to same device as model for encoding
            model_device = next(self.model.parameters()).device
            text_encoder = text_encoder.to(model_device)
            
            with torch.no_grad():
                embed_dict = text_encoder([prompt])
                embed = embed_dict['prompt_embeds']  # [1, 512, 4096]
            
            # Set it for all masked cross-attention blocks (model handles device internally)
            if hasattr(self.model, 'set_null_embedding_for_cross_attn'):
                self.model.set_null_embedding_for_cross_attn(embed)
                print(f"  Set video cross-attn embedding to: \"{prompt[:50]}...\"" if len(prompt) > 50 else f"  Set video cross-attn embedding to: \"{prompt}\"")
            else:
                print("  Warning: Model doesn't have set_null_embedding_for_cross_attn method")
                return False
            
            # Clean up if we loaded the encoder
            if cleanup_encoder:
                del text_encoder
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            return True
                    
        except Exception as e:
            print(f"  Warning: Could not set video cross-attn embedding: {e}")
            return False

    def adding_cls_branch(self, atten_dim=1536, num_class=4, time_embed_dim=0) -> None:
        # NOTE: This is hard coded for WAN2.1-T2V-1.3B for now!!!!!!!!!!!!!!!!!!!!
        self._cls_pred_branch = nn.Sequential(
            # Input: [B, 384, 21, 60, 104]
            nn.LayerNorm(atten_dim * 3 + time_embed_dim),
            nn.Linear(atten_dim * 3 + time_embed_dim, 1536),
            nn.SiLU(),
            nn.Linear(atten_dim, num_class)
        )
        self._cls_pred_branch.requires_grad_(True)
        num_registers = 3
        self._register_tokens = RegisterTokens(num_registers=num_registers, dim=atten_dim)
        self._register_tokens.requires_grad_(True)

        gan_ca_blocks = []
        for _ in range(num_registers):
            block = GanAttentionBlock()
            gan_ca_blocks.append(block)
        self._gan_ca_blocks = nn.ModuleList(gan_ca_blocks)
        self._gan_ca_blocks.requires_grad_(True)
        # self.has_cls_branch = True

    def _permute_position_latents(self, position_latents: torch.Tensor) -> torch.Tensor:
        """
        Permute position latents from training format to model format.
        
        Input formats:
        - Single subject: [B, T, C, H, W] -> [B, C, T, H, W]
        - Multi-subject: [B, S, T, C, H, W] -> [B, S, C, T, H, W]
        """
        if position_latents.dim() == 5:
            # Single subject: [B, T, C, H, W] -> [B, C, T, H, W]
            return position_latents.permute(0, 2, 1, 3, 4)
        elif position_latents.dim() == 6:
            # Multi-subject: [B, S, T, C, H, W] -> [B, S, C, T, H, W]
            return position_latents.permute(0, 1, 3, 2, 4, 5)
        else:
            raise ValueError(f"Unexpected position_latents dim: {position_latents.dim()}")
    
    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        see derivations https://chatgpt.com/share/67bf8589-3d04-8008-bc6e-4cf1a24e2d0e
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt,
                                                        self.scheduler.sigmas,
                                                        self.scheduler.timesteps]
        )

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert x0 prediction to flow matching's prediction.
        x0_pred: the x0 prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = (x_t - x_0) / sigma_t
        """
        # use higher precision for calculations
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device), [x0_pred, xt,
                                                      scheduler.sigmas,
                                                      scheduler.timesteps]
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    def forward(
        self,
        noisy_image_or_video: torch.Tensor, conditional_dict: dict,
        timestep: torch.Tensor, kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        classify_mode: Optional[bool] = False,
        concat_time_embeddings: Optional[bool] = False,
        clean_x: Optional[torch.Tensor] = None,
        aug_t: Optional[torch.Tensor] = None,
        cache_start: Optional[int] = None,
        first_frame_latent: Optional[torch.Tensor] = None,
        subject_coords: Optional[torch.Tensor] = None,
        coord_num_clean: Optional[torch.Tensor] = None,  # [B] - num clean frames per batch for spatial RoPE
        zero_position_for_masked: bool = False,  # If True, use (0,0) for noisy frame positions
        position_latents: Optional[torch.Tensor] = None,  # [B, T, C, H, W] per-subject position latents (position_token_mode)
        position_pixels: Optional[torch.Tensor] = None,  # [B, S, T, H, W] raw pixel position maps (pixel_position_mode)
    ) -> torch.Tensor:
        prompt_embeds = conditional_dict["prompt_embeds"]
        context_positions = conditional_dict.get("context_positions", None)
        video_text_embeds = conditional_dict.get("video_text_embeds", None)

        # Timestep handling for model
        if self.per_frame_timesteps and timestep.dim() == 2:
            # Per-frame timesteps: pass full [B, T] tensor to model
            # Each frame gets its own timestep modulation
            input_timestep = timestep
            # Debug: print first batch's per-frame timesteps (only once)
            if not hasattr(self, '_printed_per_frame_debug'):
                print(f"[PER-FRAME TIMESTEPS] Using per-frame mode, timestep shape: {timestep.shape}")
                print(f"  First batch timesteps: {timestep[0].tolist()}")
                self._printed_per_frame_debug = True
        elif self.uniform_timestep:
            # Uniform timestep: collapse [B, F] -> [B]
            # For AR training with coord_only, use target frame's timestep, not first frame
            if coord_num_clean is not None and timestep.dim() == 2:
                # coord_num_clean[b] is the target frame index for batch b
                # Use that frame's timestep instead of first frame
                input_timestep = torch.stack([
                    timestep[b, coord_num_clean[b].item()] 
                    for b in range(timestep.shape[0])
                ])
            else:
                input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        logits = None
        coord_pred = None
        # X0 prediction
        if kv_cache is not None:
            flow_pred = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep, context=prompt_embeds,
                seq_len=self.seq_len,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                cache_start=cache_start,
                first_frame_latent=first_frame_latent,
                context_positions=context_positions
            ).permute(0, 2, 1, 3, 4)
        else:
            if clean_x is not None:
                # teacher forcing
                flow_pred = self.model(
                    noisy_image_or_video.permute(0, 2, 1, 3, 4),
                    t=input_timestep, context=prompt_embeds,
                    seq_len=self.seq_len,
                    clean_x=clean_x.permute(0, 2, 1, 3, 4),
                    aug_t=aug_t,
                    first_frame_latent=first_frame_latent,
                    context_positions=context_positions
                ).permute(0, 2, 1, 3, 4)
            else:
                if classify_mode:
                    flow_pred, logits = self.model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep, context=prompt_embeds,
                        seq_len=self.seq_len,
                        classify_mode=True,
                        register_tokens=self._register_tokens,
                        cls_pred_branch=self._cls_pred_branch,
                        gan_ca_blocks=self._gan_ca_blocks,
                        concat_time_embeddings=concat_time_embeddings,
                        first_frame_latent=first_frame_latent,
                        context_positions=context_positions
                    )
                    flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
                else:
                    model_output = self.model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep, context=prompt_embeds,
                        seq_len=self.seq_len,
                        first_frame_latent=first_frame_latent,
                        context_positions=context_positions,
                        subject_coords=subject_coords,
                        coord_num_clean=coord_num_clean,
                        zero_position_for_masked=zero_position_for_masked,
                        # position_latents: [B, T, C, H, W] or [B, S, T, C, H, W]
                        # Model expects: [B, C, T, H, W] or [B, S, C, T, H, W]
                        position_latents=self._permute_position_latents(position_latents) if position_latents is not None else None,
                        # position_pixels: [B, S, T, H, W] for pixel position mode
                        position_pixels=position_pixels,
                        # Pre-encoded game text for video cross-attention [B, 512, 4096]
                        video_text_embeds=video_text_embeds,
                    )
                    # Handle coord_token_mode returns
                    if getattr(self.model, 'coord_only_no_video', False):
                        # COORD ONLY MODE: model returns just coord_pred, no video
                        coord_pred = model_output
                        # Create dummy flow_pred (zeros) for compatibility
                        flow_pred = torch.zeros_like(noisy_image_or_video)
                    elif self.position_token_mode and isinstance(model_output, tuple):
                        # POSITION TOKEN MODE: model returns (video_pred, position_pred)
                        # video_pred: [B, C, T, H, W] -> [B, T, C, H, W]
                        # position_pred: [B, C, T, H, W] or [B, S, C, T, H, W]
                        flow_pred, position_pred = model_output
                        flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
                        # Permute position_pred back to training format
                        if position_pred.dim() == 5:
                            position_pred = position_pred.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
                        elif position_pred.dim() == 6:
                            position_pred = position_pred.permute(0, 1, 3, 2, 4, 5)  # [B, S, T, C, H, W]
                        coord_pred = position_pred  # Reuse coord_pred for position output
                    elif self.pixel_position_mode and isinstance(model_output, tuple):
                        # PIXEL POSITION MODE: model returns (video_pred, position_pred)
                        # video_pred: [B, C, T, H, W] -> [B, T, C, H, W]
                        # position_pred: [B, S, T, H, W] pixel-space position maps
                        flow_pred, position_pred = model_output
                        flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
                        coord_pred = position_pred  # Reuse coord_pred for position output
                    elif self.coord_token_mode and isinstance(model_output, tuple):
                        flow_pred, coord_pred = model_output
                        flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
                    else:
                        flow_pred = model_output.permute(0, 2, 1, 3, 4)
                        coord_pred = None

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1)
        ).unflatten(0, flow_pred.shape[:2])

        if logits is not None:
            return flow_pred, pred_x0, logits
        
        # Position token mode: return position_pred as third element
        if self.position_token_mode and coord_pred is not None:
            # coord_pred here holds position predictions (flow)
            # Convert position flow to position x0
            position_flow = coord_pred
            
            if position_flow.dim() == 6:
                # Multi-subject: [B, S, T, C, H, W]
                B, S, T, C, H, W = position_flow.shape
                # Flatten B, S, T together for conversion
                pos_flow_flat = position_flow.reshape(B * S * T, C, H, W)
                pos_latents_flat = position_latents.reshape(B * S * T, C, H, W)
                # Expand timestep for all subjects
                timestep_expanded = timestep.unsqueeze(1).expand(-1, S, -1).reshape(B * S * T)
                position_x0 = self._convert_flow_pred_to_x0(
                    flow_pred=pos_flow_flat,
                    xt=pos_latents_flat,
                    timestep=timestep_expanded
                ).reshape(B, S, T, C, H, W)
            else:
                # Single subject: [B, T, C, H, W]
                position_x0 = self._convert_flow_pred_to_x0(
                    flow_pred=position_flow.flatten(0, 1),
                    xt=position_latents.flatten(0, 1),
                    timestep=timestep.flatten(0, 1)
                ).unflatten(0, position_flow.shape[:2])
            
            return flow_pred, pred_x0, position_flow, position_x0
        
        # Pixel position mode: return position_pred as third element
        if self.pixel_position_mode and coord_pred is not None:
            # coord_pred here holds pixel-space position predictions (flow)
            # Convert position flow to position x0
            position_flow = coord_pred  # [B, S, T, H, W]
            
            B, S, T, H, W = position_flow.shape
            # Flatten B, S, T together for conversion
            pos_flow_flat = position_flow.reshape(B * S * T, 1, H, W)  # Add dummy channel for conversion
            pos_pixels_flat = position_pixels.reshape(B * S * T, 1, H, W)
            # Expand timestep
            timestep_expanded = timestep[:, :T].reshape(B * T).unsqueeze(1).expand(-1, S).reshape(B * S * T)
            position_x0_flat = self._convert_flow_pred_to_x0(
                flow_pred=pos_flow_flat,
                xt=pos_pixels_flat,
                timestep=timestep_expanded
            )
            position_x0 = position_x0_flat.reshape(B, S, T, H, W)
            
            return flow_pred, pred_x0, position_flow, position_x0
        
        # Coordinate token mode: return coord_pred as third element
        if self.coord_token_mode and coord_pred is not None:
            return flow_pred, pred_x0, coord_pred

        return flow_pred, pred_x0

    def get_scheduler(self) -> SchedulerInterface:
        """
        Update the current scheduler with the interface's static method
        """
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        """
        A few custom initialization steps that should be called after the object is created.
        Currently, the only one we have is to bind a few methods to scheduler.
        We can gradually add more methods here if needed.
        """
        self.get_scheduler()
