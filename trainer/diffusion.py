"""Diffusion trainer (``trainer: diffusion`` in configs).

This is the trainer used by every exp-131/134/135/138/139/141-147 config
in this repo. The core training loop is upstream Self-Forcing; our additions
are:

* dataset selection for the Melting Pot / shapes variants (see
  :mod:`utils.dataset`),
* pre-encoding of per-game text descriptions through T5 (see
  :mod:`utils.game_descriptions`),
* handling of ``coord_token_mode`` / ``coord_diffusion`` configs which run the
  DiT on the joint video + coord-token sequence.

Use :class:`Trainer` through ``train.py``; do not instantiate directly.
"""
import gc
import logging

from model import CausalDiffusion
from utils.dataset import ShardingLMDBDataset, MultiSubjectLMDBDataset, PositionTokenLMDBDataset, PixelPositionLMDBDataset, MultiGameLMDBDataset, cycle
from utils.misc import set_seed
from utils.attention_logger import AttentionLogger
import torch.distributed as dist
from omegaconf import OmegaConf
import torch
import wandb
import time
import os

from utils.distributed import EMA_FSDP, barrier, fsdp_wrap, fsdp_state_dict, launch_distributed_job


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process and not self.disable_wandb:
            wandb.login(host=config.wandb_host, key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )

        self.output_path = config.logdir

        # Step 2: Initialize the model
        self.model = CausalDiffusion(config, device=self.device)
        
        # Load checkpoints BEFORE FSDP wrapping to allow clean initialization
        self._load_checkpoints_before_fsdp(config)
        
        # Initialize multi-subject conditioner's patch embedding from model weights
        # MUST be done BEFORE FSDP wrapping (FSDP flattens weights)
        # Only needed for the original (image pooling) conditioner, not the simple one
        if hasattr(self.model, 'multi_subject_conditioner'):
            # Only initialize patch embedding if it's the original conditioner (has initialize_from_patch_embedding)
            if hasattr(self.model.multi_subject_conditioner, 'initialize_from_patch_embedding'):
                self.model.initialize_multi_subject_patch_embedding()
            self.multi_subject_conditioner = self.model.multi_subject_conditioner
        
        # Store reference to single-subject conditioner (not FSDP-wrapped)
        if hasattr(self.model, 'single_subject_conditioner'):
            self.single_subject_conditioner = self.model.single_subject_conditioner
        
        # Set null embedding AND pre-encode game descriptions for masked cross-attention
        # MUST be done BEFORE FSDP wrapping and AFTER text encoder + model are loaded
        cross_attn_type = getattr(config, 'model_kwargs', {}).get('cross_attn_type', None)
        self.game_text_cache = None  # Pre-encoded game descriptions {prompt_text: [1, 512, 4096]}
        if cross_attn_type in ('masked_position_cross_attn', 'dual_stream_cross_attn'):
            if self.model.text_encoder is not None:
                self.model.text_encoder.to(self.device)
                with torch.no_grad():
                    # 1. Set null embedding (fallback for inference / samples without text)
                    negative_prompt = getattr(config, 'negative_prompt', '')
                    null_dict = self.model.text_encoder(text_prompts=[negative_prompt])
                    null_embed = null_dict['prompt_embeds']  # [1, 512, 4096]
                    self.model.generator.model.set_null_embedding_for_cross_attn(null_embed)
                    if self.is_main_process:
                        print(f"  Set null embedding for masked cross-attention from T5-encoded \"{negative_prompt}\"")
                    
                    # 2. Pre-encode all game descriptions (avoids running T5 every step)
                    if getattr(config, 'use_game_descriptions', False):
                        try:
                            from utils.game_descriptions import GAME_DESCRIPTIONS, get_game_description
                            unique_prompts = set(GAME_DESCRIPTIONS.values())
                            # Also add fallback descriptions for any games without explicit descriptions
                            # (get_game_description generates a fallback from the game name)
                            if self.is_main_process:
                                print(f"  Pre-encoding {len(unique_prompts)} unique game descriptions with T5...")
                            self.game_text_cache = {}
                            # Encode in small batches for efficiency
                            prompt_list = list(unique_prompts)
                            batch_size_encode = 8
                            for i in range(0, len(prompt_list), batch_size_encode):
                                batch_prompts = prompt_list[i:i + batch_size_encode]
                                batch_dict = self.model.text_encoder(text_prompts=batch_prompts)
                                batch_embeds = batch_dict['prompt_embeds']  # [batch, 512, 4096]
                                for j, prompt in enumerate(batch_prompts):
                                    self.game_text_cache[prompt] = batch_embeds[j:j+1].detach().cpu()  # [1, 512, 4096]
                            if self.is_main_process:
                                print(f"  Cached {len(self.game_text_cache)} game text embeddings (T5 dim 4096)")
                        except ImportError:
                            if self.is_main_process:
                                print("  WARNING: utils.game_descriptions not importable, video tokens will use null embedding")
                
                # Free text encoder memory - no longer needed after pre-encoding
                if self.is_main_process:
                    print("  Freeing text encoder memory (all text embeddings pre-cached)")
                del self.model.text_encoder
                self.model.text_encoder = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            elif self.is_main_process:
                print("  WARNING: masked cross-attention but no text encoder loaded - video tokens will attend to zeros!")
        
        # Wrap generator with FSDP
        # If using action conditioning, tell FSDP to NOT shard the action_embedding
        # (Embedding layers need full weight matrix, can't be sharded)
        ignored_modules = []
        if hasattr(self.model.generator.model, 'action_embedding'):
            ignored_modules.append(self.model.generator.model.action_embedding)
        
        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
            ignored_modules=ignored_modules if ignored_modules else None
        )

        # Only wrap text_encoder if it exists
        if self.model.text_encoder is not None:
            self.model.text_encoder = fsdp_wrap(
                self.model.text_encoder,
                sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision,
                wrap_strategy=config.text_encoder_fsdp_wrap_strategy
            )

        # Only move VAE to device if it exists and is needed
        if self.model.vae is not None and (not config.no_visualize or config.load_raw_video):
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        # COORD ONLY MODE: Freeze all parameters except coord encoder/decoder
        # EXCEPTION: If using direct embedding, pure slicing, or coord_train_full_dit, train the full DiT
        coord_only_mode = getattr(config, 'coord_only_mode', False)
        coord_direct_embedding = getattr(config, 'coord_direct_embedding', False)
        coord_no_encoder_decoder = getattr(config, 'coord_no_encoder_decoder', False)
        coord_train_full_dit = getattr(config, 'coord_train_full_dit', False)
        
        if coord_only_mode and not coord_direct_embedding and not coord_no_encoder_decoder and not coord_train_full_dit:
            # Standard coord-only with learned encoder/decoder: freeze everything except coord encoder/decoder
            frozen_count = 0
            trainable_count = 0
            for name, param in self.model.generator.named_parameters():
                # Only keep coord_encoder, coord_decoder, coord_encoder_norm trainable
                if any(x in name for x in ['coord_encoder', 'coord_decoder']):
                    param.requires_grad = True
                    trainable_count += 1
                else:
                    param.requires_grad = False
                    frozen_count += 1
            if self.is_main_process:
                print(f"🧊 COORD ONLY MODE: Frozen {frozen_count} params, trainable {trainable_count} (coord_encoder/decoder only)")
        elif coord_only_mode and (coord_direct_embedding or coord_no_encoder_decoder or coord_train_full_dit):
            # Direct embedding, pure slicing, or explicit full DiT training: train everything
            trainable_count = sum(1 for p in self.model.generator.parameters() if p.requires_grad)
            if coord_train_full_dit:
                mode_name = "LEARNED ENC/DEC + FULL DiT"
            elif coord_no_encoder_decoder:
                mode_name = "PURE SLICING"
            else:
                mode_name = "DIRECT EMBEDDING"
            if self.is_main_process:
                print(f"🧊 COORD ONLY MODE with {mode_name}: Training full DiT ({trainable_count} trainable params)")

        # Create optimizer with potentially different learning rates for different parameter groups
        action_embedding_lr_multiplier = getattr(config, 'action_embedding_lr_multiplier', 1.0)
        cross_attn_lr_multiplier = getattr(config, 'cross_attn_lr_multiplier', 1.0)
        action_cross_attn_lr_multiplier = getattr(config, 'action_cross_attn_lr_multiplier', None)
        position_lr_multiplier = getattr(config, 'position_lr_multiplier', 1.0)
        
        # Build parameter groups based on what's enabled
        has_action_embedding = hasattr(self.model.generator.model, 'action_embedding') and action_embedding_lr_multiplier != 1.0
        has_cross_attn_lr = cross_attn_lr_multiplier != 1.0
        has_action_cross_attn_lr = action_cross_attn_lr_multiplier is not None and action_cross_attn_lr_multiplier != 1.0
        has_position_lr = position_lr_multiplier != 1.0 and (
            hasattr(self.model.generator.model, 'position_embedding') or 
            hasattr(self.model.generator.model, 'win_proj')  # VideoJAM mode
        )
        
        if has_action_embedding or has_cross_attn_lr or has_action_cross_attn_lr or has_position_lr:
            # Build parameter groups with different LRs
            param_groups = []
            assigned_params = set()
            
            # Action embedding params (highest priority)
            if has_action_embedding:
                action_embedding_params = list(self.model.generator.model.action_embedding.parameters())
                for p in action_embedding_params:
                    assigned_params.add(id(p))
                param_groups.append({
                    'params': action_embedding_params, 
                    'lr': config.lr * action_embedding_lr_multiplier,
                    'name': 'action_embedding'
                })
            
            # Action cross-attention params (dual_stream_cross_attn mode)
            # MUST be checked BEFORE general cross_attn to avoid double-matching
            if has_action_cross_attn_lr:
                action_cross_attn_params = []
                for name, param in self.model.generator.named_parameters():
                    if param.requires_grad and id(param) not in assigned_params:
                        # Match blocks.*.cross_attn_action.* (the coord->action cross-attention)
                        if 'cross_attn_action' in name:
                            action_cross_attn_params.append(param)
                            assigned_params.add(id(param))
                
                if action_cross_attn_params:
                    param_groups.append({
                        'params': action_cross_attn_params,
                        'lr': config.lr * action_cross_attn_lr_multiplier,
                        'name': 'action_cross_attn'
                    })
            
            # Cross-attention params (video->null, or shared cross-attn in non-dual modes)
            if has_cross_attn_lr:
                cross_attn_params = []
                for name, param in self.model.generator.named_parameters():
                    if param.requires_grad and id(param) not in assigned_params:
                        # Match cross-attention blocks: blocks.*.cross_attn.*
                        # Note: cross_attn_action params already assigned above
                        if 'cross_attn' in name:
                            cross_attn_params.append(param)
                            assigned_params.add(id(param))
                
                if cross_attn_params:
                    param_groups.append({
                        'params': cross_attn_params,
                        'lr': config.lr * cross_attn_lr_multiplier,
                        'name': 'cross_attn'
                    })
            
            # Position embedding and head params (for joint position diffusion)
            if has_position_lr:
                position_params = []
                for name, param in self.model.generator.named_parameters():
                    if param.requires_grad and id(param) not in assigned_params:
                        # Match position_embedding, position_head, and VideoJAM projections
                        if any(x in name for x in ['position_embedding', 'position_head', 'win_proj', 'wout_proj']):
                            position_params.append(param)
                            assigned_params.add(id(param))
                
                if position_params:
                    param_groups.append({
                        'params': position_params,
                        'lr': config.lr * position_lr_multiplier,
                        'name': 'position'
                    })
            
            # All other params (base LR)
            other_params = [p for p in self.model.generator.parameters() 
                          if p.requires_grad and id(p) not in assigned_params]
            if other_params:
                param_groups.append({
                    'params': other_params,
                    'lr': config.lr,
                    'name': 'other'
                })
            
            if self.is_main_process:
                print(f"Optimizer parameter groups:")
                for group in param_groups:
                    print(f"  {group['name']}: {len(group['params'])} params, LR={group['lr']:.2e}")
            
            self.generator_optimizer = torch.optim.AdamW(
                param_groups,
                betas=(config.beta1, config.beta2),
                weight_decay=config.weight_decay
            )
        else:
            # Standard: single learning rate for all parameters
            all_params = [param for param in self.model.generator.parameters() if param.requires_grad]
            
            self.generator_optimizer = torch.optim.AdamW(
                all_params,
                lr=config.lr,
                betas=(config.beta1, config.beta2),
                weight_decay=config.weight_decay
            )
        
        # Create separate optimizer for multi-subject conditioner (not FSDP-wrapped)
        # Use same 100x LR multiplier as action embeddings in generator
        self.multi_subject_optimizer = None
        if hasattr(self, 'multi_subject_conditioner'):
            multi_subj_params = [p for p in self.multi_subject_conditioner.parameters() if p.requires_grad]
            if multi_subj_params:
                multi_subject_lr = config.lr * action_embedding_lr_multiplier  # 100x LR
                self.multi_subject_optimizer = torch.optim.AdamW(
                    multi_subj_params,
                    lr=multi_subject_lr,
                    betas=(config.beta1, config.beta2),
                    weight_decay=config.weight_decay
                )
                if self.is_main_process:
                    print(f"Created separate optimizer for {len(multi_subj_params)} multi-subject conditioner parameters")
                    print(f"  Multi-subject conditioner LR: {multi_subject_lr} ({action_embedding_lr_multiplier}x)")
        
        # Create separate optimizer for single-subject conditioner (not FSDP-wrapped)
        self.single_subject_optimizer = None
        if hasattr(self, 'single_subject_conditioner'):
            single_subj_params = [p for p in self.single_subject_conditioner.parameters() if p.requires_grad]
            if single_subj_params:
                single_subject_lr = config.lr * action_embedding_lr_multiplier  # 100x LR
                self.single_subject_optimizer = torch.optim.AdamW(
                    single_subj_params,
                    lr=single_subject_lr,
                    betas=(config.beta1, config.beta2),
                    weight_decay=config.weight_decay
                )
                if self.is_main_process:
                    print(f"Created separate optimizer for {len(single_subj_params)} single-subject conditioner parameters")
                    print(f"  Single-subject conditioner LR: {single_subject_lr} ({action_embedding_lr_multiplier}x)")

        # LR Warmup configuration
        self.lr_warmup_steps = getattr(config, 'lr_warmup_steps', 0)
        self.lr_warmup_min_ratio = getattr(config, 'lr_warmup_min_ratio', 0.1)  # Start at 10% of target LR
        self.base_lr = config.lr
        
        # LR Cosine decay configuration
        self.lr_cosine_decay = getattr(config, 'lr_cosine_decay', False)
        self.lr_min_ratio = getattr(config, 'lr_min_ratio', 0.02)  # Decay to 2% of peak LR
        self.num_iter = config.num_iter  # Total training steps
        
        if self.lr_warmup_steps > 0:
            # Store base LR for each param group so we can restore after warmup
            for param_group in self.generator_optimizer.param_groups:
                param_group['_base_lr'] = param_group['lr']
            if self.multi_subject_optimizer is not None:
                for param_group in self.multi_subject_optimizer.param_groups:
                    param_group['_base_lr'] = param_group['lr']
            if self.single_subject_optimizer is not None:
                for param_group in self.single_subject_optimizer.param_groups:
                    param_group['_base_lr'] = param_group['lr']
            if self.is_main_process:
                min_lr = self.base_lr * self.lr_warmup_min_ratio
                if self.lr_cosine_decay:
                    final_lr = self.base_lr * self.lr_min_ratio
                    print(f"Using LR schedule: warmup {self.lr_warmup_steps} steps ({min_lr:.2e} → {self.base_lr:.2e}) then cosine decay → {final_lr:.2e}")
                else:
                    print(f"Using LR warmup: {self.lr_warmup_steps} steps ({min_lr:.2e} → {self.base_lr:.2e})")

        # Step 3: Initialize the dataloader
        load_actions = getattr(config, 'load_actions', False)
        use_multi_subject = getattr(config, 'use_multi_subject_conditioning', False)
        use_simple_multi_subject = getattr(config, 'use_simple_multi_subject_conditioning', False)
        use_modulated_multi_subject = getattr(config, 'use_modulated_multi_subject_conditioning', False)
        use_position_aware = getattr(config, 'use_position_aware_conditioning', False)
        use_spatial_rope = getattr(config, 'use_spatial_rope_conditioning', False)
        
        # Position token mode: uses PositionTokenLMDBDataset with per-subject position latents
        position_token_mode = getattr(config, 'position_token_mode', False)
        # Coord token mode: also uses PositionTokenLMDBDataset (same format, has frame_coords)
        coord_token_mode = getattr(config, 'coord_token_mode', False)
        # Pixel position mode: uses PixelPositionLMDBDataset with raw pixel position maps
        pixel_position_mode = getattr(config, 'pixel_position_mode', False)
        
        if pixel_position_mode:
            dataset = PixelPositionLMDBDataset(config.data_path, max_pair=int(1e8))
            if self.is_main_process:
                print(f"Using PixelPositionLMDBDataset with {dataset.num_subjects} subjects, "
                      f"video shape {dataset.video_latents_shape}, position pixels shape {dataset.position_pixels_shape}")
        elif position_token_mode or coord_token_mode or getattr(config, 'use_position_token_dataset', False):
            # Check if this is a multi-game dataset (has subdirectories with game names)
            multi_game = getattr(config, 'multi_game_dataset', False)
            if multi_game:
                max_subj = getattr(config, 'num_subjects_total', None)
                max_act = getattr(config, 'num_action_classes', None)
                use_desc = getattr(config, 'use_game_descriptions', True)
                dataset = MultiGameLMDBDataset(config.data_path, max_pair=int(1e8),
                                              max_subjects=max_subj, max_action_classes=max_act,
                                              use_game_descriptions=use_desc)
                if self.is_main_process:
                    print(f"Using MultiGameLMDBDataset with {dataset.num_subjects} max subjects, "
                          f"video shape {dataset.video_latents_shape}, "
                          f"{len(dataset.envs)} game shards, {len(dataset)} samples")
            else:
                dataset = PositionTokenLMDBDataset(config.data_path, max_pair=int(1e8))
                if self.is_main_process:
                    mode_str = "position_token_mode" if position_token_mode else "coord_token_mode"
                    print(f"Using PositionTokenLMDBDataset ({mode_str}) with {dataset.num_subjects} subjects, "
                          f"video shape {dataset.video_latents_shape}, position shape {dataset.position_latents_shape}")
        elif use_multi_subject or use_simple_multi_subject or use_modulated_multi_subject or use_position_aware or use_spatial_rope:
            # Multi-subject dataset: loads subject_ids and per-subject actions (prototypes optional)
            dataset = MultiSubjectLMDBDataset(config.data_path, max_pair=int(1e8))
            if self.is_main_process:
                if dataset.subject_prototypes is not None:
                    print(f"Using MultiSubjectLMDBDataset with {dataset.subject_prototypes.shape[0]} subject prototypes")
                else:
                    print(f"Using MultiSubjectLMDBDataset with {dataset.num_subjects_per_video} subjects (learned embeddings)")
        else:
            # Standard dataset
            dataset = ShardingLMDBDataset(config.data_path, max_pair=int(1e8), load_actions=load_actions)
        
        # OVERFIT SINGLE SAMPLE MODE: always return sample 0
        overfit_single_sample = getattr(config, 'overfit_single_sample', False)
        if overfit_single_sample:
            if self.is_main_process:
                print(f"OVERFIT MODE: Training on single sample (index 0) only!")
            # Create a sampler that always returns index 0
            class SingleSampleSampler(torch.utils.data.Sampler):
                def __init__(self, num_samples):
                    self.num_samples = num_samples
                def __iter__(self):
                    return iter([0] * self.num_samples)
                def __len__(self):
                    return self.num_samples
            sampler = SingleSampleSampler(num_samples=10000)
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=config.batch_size,
                sampler=sampler,
                num_workers=0)  # No workers for single sample
        # For tiny datasets, we want to repeat samples within a batch
        # Use RandomSampler with replacement instead of DistributedSampler
        elif len(dataset) < config.batch_size * dist.get_world_size():
            print(f"WARNING: Dataset size ({len(dataset)}) < batch_size * num_GPUs ({config.batch_size * dist.get_world_size()})")
            print(f"Using RandomSampler with replacement to repeat samples in batches")
            sampler = torch.utils.data.RandomSampler(
                dataset, replacement=True, num_samples=config.batch_size * 1000)  # Generate enough samples
            # Don't use DistributedSampler for tiny datasets - let each GPU see all data
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=config.batch_size,
                sampler=sampler,
                num_workers=8)
        else:
            # Standard distributed training
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, shuffle=True, drop_last=True)
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=config.batch_size,
                sampler=sampler,
                num_workers=8)

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        ##############################################################################################################
        # 6. Set up EMA parameter containers
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p
        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0):
            print(f"Setting up EMA with weight {ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)

        ##############################################################################################################
        # 7. Load optimizer/EMA state if resuming (model weights already loaded before FSDP)
        if self.ckpt_data_for_optimizer is not None:
            ckpt_data = self.ckpt_data_for_optimizer
            
            # Load optimizer state using FSDP-compatible API
            if "optimizer" in ckpt_data:
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
                
                try:
                    # Use FSDP's optimizer state loading to handle parameter ID mapping
                    with FSDP.state_dict_type(
                        self.model.generator,
                        StateDictType.FULL_STATE_DICT,
                        FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
                    ):
                        optim_state = FSDP.optim_state_dict_to_load(
                            self.model.generator, self.generator_optimizer, ckpt_data["optimizer"]
                        )
                        self.generator_optimizer.load_state_dict(optim_state)
                    
                    if self.is_main_process:
                        print("  ✓ Loaded optimizer state (FSDP-compatible)")
                except Exception as e:
                    if self.is_main_process:
                        print(f"  ⚠️  Failed to load optimizer state: {e}")
                        print("     Starting with fresh optimizer (momentum reset)")
            
            # Load EMA state
            if "generator_ema" in ckpt_data and self.generator_ema is not None:
                self.generator_ema.load_state_dict(ckpt_data["generator_ema"])
                if self.is_main_process:
                    print("  ✓ Loaded EMA state")
            
            # Load multi-subject conditioner state
            if "multi_subject_conditioner" in ckpt_data and hasattr(self, 'multi_subject_conditioner'):
                self.multi_subject_conditioner.load_state_dict(ckpt_data["multi_subject_conditioner"])
                if self.is_main_process:
                    print("  ✓ Loaded multi-subject conditioner state")
            
            # Load multi-subject optimizer state
            if "multi_subject_optimizer" in ckpt_data and self.multi_subject_optimizer is not None:
                self.multi_subject_optimizer.load_state_dict(ckpt_data["multi_subject_optimizer"])
                if self.is_main_process:
                    print("  ✓ Loaded multi-subject optimizer state")
            
            # Load single-subject conditioner state
            if "single_subject_conditioner" in ckpt_data and hasattr(self, 'single_subject_conditioner'):
                self.single_subject_conditioner.load_state_dict(ckpt_data["single_subject_conditioner"])
                if self.is_main_process:
                    print("  ✓ Loaded single-subject conditioner state")
            
            # Load single-subject optimizer state
            if "single_subject_optimizer" in ckpt_data and self.single_subject_optimizer is not None:
                self.single_subject_optimizer.load_state_dict(ckpt_data["single_subject_optimizer"])
                if self.is_main_process:
                    print("  ✓ Loaded single-subject optimizer state")

        # Let's delete EMA params for early steps to save some computes at training and inference
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm = 10.0
        self.previous_time = None
        
        # Initialize attention logger for monitoring self/cross attention norms
        self.attention_logger = None
        self.attention_log_interval = getattr(config, 'attention_log_interval', 100)  # Log every N steps
        if hasattr(self.model.generator, 'model') and hasattr(self.model.generator.model, 'blocks'):
            self.attention_logger = AttentionLogger(self.model.generator.model.blocks)
            if self.is_main_process:
                print(f"Attention logger initialized (logging every {self.attention_log_interval} steps)")

    def _load_checkpoints_before_fsdp(self, config):
        """
        Load model weights and initialize latent_patch_embedding BEFORE FSDP wrapping.
        This avoids summon_full_params overhead and allows cleaner initialization.
        
        Stores checkpoint data in self.ckpt_data_for_optimizer for later loading.
        """
        self.ckpt_data_for_optimizer = None  # Will store ckpt for optimizer/EMA loading
        
        # Try to find checkpoint in logdir first
        latest_ckpt_path, latest_step = self._find_latest_checkpoint(config.logdir)
        
        if latest_ckpt_path is not None:
            # Resume from training checkpoint
            if self.is_main_process:
                print(f"\n{'='*80}")
                print(f"Found checkpoint in logdir: {latest_ckpt_path}")
                print(f"Auto-resuming from step {latest_step}")
                print(f"{'='*80}\n")
            
            ckpt_data = torch.load(latest_ckpt_path, map_location="cpu")
            self.ckpt_data_for_optimizer = ckpt_data  # Save for later
            self.step = ckpt_data.get("step", latest_step)
            
            # Load generator weights (includes latent_patch_embedding if in checkpoint)
            if "generator" in ckpt_data:
                # Check if latent_patch_embedding is in the checkpoint
                has_latent_in_ckpt = any('latent_patch_embedding' in key for key in ckpt_data["generator"].keys())
                
                self.model.generator.load_state_dict(ckpt_data["generator"], strict=True)
                if self.is_main_process:
                    print("  ✓ Loaded generator state (before FSDP)")
                
                # If latent_patch_embedding not in checkpoint, copy from patch_embedding
                if self.model.i2v_cross_attention_latent and not has_latent_in_ckpt:
                    with torch.no_grad():
                        self.model.generator.model.latent_patch_embedding.weight.copy_(
                            self.model.generator.model.patch_embedding.weight
                        )
                        if self.model.generator.model.patch_embedding.bias is not None:
                            self.model.generator.model.latent_patch_embedding.bias.copy_(
                                self.model.generator.model.patch_embedding.bias
                            )
                    if self.is_main_process:
                        print("  ✓ Initialized latent_patch_embedding from patch_embedding")
        
        elif getattr(config, "generator_ckpt", False):
            # Load pretrained weights (fresh training)
            if self.is_main_process:
                print(f"No checkpoint in logdir. Loading pretrained generator from {config.generator_ckpt}")
            
            # Load checkpoint (support both .pt and .safetensors)
            if config.generator_ckpt.endswith('.safetensors'):
                from safetensors.torch import load_file
                state_dict = load_file(config.generator_ckpt)
            else:
                state_dict = torch.load(config.generator_ckpt, map_location="cpu")
            
            # Unwrap if nested
            if "generator" in state_dict:
                state_dict = state_dict["generator"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            
            # Filter out parameters with size mismatches
            model_state_dict = self.model.generator.state_dict()
            filtered_state_dict = {}
            skipped_keys = []
            
            for key, value in state_dict.items():
                if key in model_state_dict:
                    if value.shape == model_state_dict[key].shape:
                        filtered_state_dict[key] = value
                    else:
                        skipped_keys.append(f"{key} (ckpt: {value.shape} vs model: {model_state_dict[key].shape})")
                else:
                    filtered_state_dict[key] = value
            
            # Load with strict=False to allow missing keys (e.g., latent_patch_embedding)
            missing, unexpected = self.model.generator.load_state_dict(filtered_state_dict, strict=False)
            
            # Check if latent_patch_embedding was in the missing keys
            latent_was_missing = any('latent_patch_embedding' in key for key in missing)
            
            if self.is_main_process:
                print("  ✓ Loaded pretrained weights (before FSDP)")
                if skipped_keys:
                    print(f"  → Skipped keys (shape mismatch): {skipped_keys[:3]}..." if len(skipped_keys) > 3 else f"  → Skipped keys: {skipped_keys}")
                if missing:
                    print(f"  → Missing keys (randomly initialized): {list(missing)[:5]}..." if len(missing) > 5 else f"  → Missing keys: {missing}")
                if unexpected:
                    print(f"  → Unexpected keys (ignored): {unexpected}")
            
            # Initialize latent_patch_embedding from patch_embedding if it was missing
            if self.model.i2v_cross_attention_latent and latent_was_missing:
                with torch.no_grad():
                    self.model.generator.model.latent_patch_embedding.weight.copy_(
                        self.model.generator.model.patch_embedding.weight
                    )
                    if self.model.generator.model.patch_embedding.bias is not None:
                        self.model.generator.model.latent_patch_embedding.bias.copy_(
                            self.model.generator.model.patch_embedding.bias
                        )
                if self.is_main_process:
                    print("  ✓ Initialized latent_patch_embedding from patch_embedding")
            
            # Initialize separate position modulation from video modulation weights
            if getattr(config, "separate_position_modulation", False):
                if hasattr(self.model.generator, 'model') and hasattr(self.model.generator.model, 'init_position_modulation_from_video'):
                    self.model.generator.model.init_position_modulation_from_video()
                    if self.is_main_process:
                        print("  ✓ Initialized position modulation from video modulation")
            
            self.step = 0  # Fresh training
        else:
            # No checkpoint at all
            if self.model.i2v_cross_attention_latent:
                self.model.initialize_latent_patch_embedding()
                if self.is_main_process:
                    print("  ✓ Initialized latent_patch_embedding (no checkpoint)")
            self.step = 0

    def _find_latest_checkpoint(self, logdir):
        """Find the latest checkpoint in the logdir. Returns (path, step_number)."""
        import glob
        import re
        
        if not logdir or not os.path.exists(logdir):
            return None, None
        
        # Find all checkpoint directories
        ckpt_pattern = os.path.join(logdir, "checkpoint_model_*/model.pt")
        ckpt_files = glob.glob(ckpt_pattern)
        
        if not ckpt_files:
            return None, None
        
        # Extract step numbers and find the latest
        def get_step(path):
            match = re.search(r'checkpoint_model_(\d+)', path)
            return int(match.group(1)) if match else -1
        
        latest_ckpt = max(ckpt_files, key=get_step)
        latest_step = get_step(latest_ckpt)
        return latest_ckpt, latest_step

    def save(self):
        print("Start gathering distributed model states...")
        # Generator uses FSDP, needs special state_dict gathering
        generator_state_dict = fsdp_state_dict(self.model.generator)
        
        # FSDP-compatible optimizer state dict
        # Use FSDP's optimizer state gathering to preserve parameter IDs
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
        
        with FSDP.state_dict_type(
            self.model.generator,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        ):
            optimizer_state_dict = FSDP.optim_state_dict(self.model.generator, self.generator_optimizer)

        state_dict = {
            "generator": generator_state_dict,
            "optimizer": optimizer_state_dict,  # FSDP-compatible optimizer state
            "step": self.step,
        }
        
        # EMA (only after warmup)
        if self.generator_ema is not None and self.config.ema_start_step < self.step:
            state_dict["generator_ema"] = self.generator_ema.state_dict()
        
        # Multi-subject conditioner (not FSDP, regular state dict)
        if hasattr(self, 'multi_subject_conditioner') and self.multi_subject_conditioner is not None:
            state_dict["multi_subject_conditioner"] = self.multi_subject_conditioner.state_dict()
        if self.multi_subject_optimizer is not None:
            state_dict["multi_subject_optimizer"] = self.multi_subject_optimizer.state_dict()
        
        # Single-subject conditioner (not FSDP, regular state dict)
        if hasattr(self, 'single_subject_conditioner') and self.single_subject_conditioner is not None:
            state_dict["single_subject_conditioner"] = self.single_subject_conditioner.state_dict()
        if self.single_subject_optimizer is not None:
            state_dict["single_subject_optimizer"] = self.single_subject_optimizer.state_dict()

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path,
                  f"checkpoint_model_{self.step:06d}", "model.pt"))

    def train_one_step(self, batch):
        self.log_iters = 1

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        # Use fixed_prompt if specified in config, otherwise use batch prompts
        if getattr(self.config, 'fixed_prompt', None):
            batch_size_for_prompt = len(batch["prompts"]) if "prompts" in batch else batch["ode_latent"].shape[0]
            text_prompts = [self.config.fixed_prompt] * batch_size_for_prompt
        else:
            text_prompts = batch["prompts"]
        if not self.config.load_raw_video:  # precomputed latent
            # Dataset shape after batching: [B, 1, F, C, H, W] for simple preencoded
            # or [B, num_denoising_steps, F, C, H, W] for ODE trajectory
            # [:, -1] removes singleton dim or selects last denoising step -> [B, F, C, H, W]
            clean_latent = batch["ode_latent"][:, -1].to(
                device=self.device, dtype=self.dtype)
        else:  # encode raw video to latent
            assert self.model.vae is not None, "load_vae must be True when load_raw_video is True"
            frames = batch["frames"].to(
                device=self.device, dtype=self.dtype)
            with torch.no_grad():
                clean_latent = self.model.vae.encode_to_latent(
                    frames).to(device=self.device, dtype=self.dtype)
        # Extract first frame for i2v conditioning (if enabled)
        if getattr(self.config, 'independent_first_frame', False):
            image_latent = clean_latent[:, 0:1, ]  # First frame for i2v
        else:
            image_latent = None  # Unconditional: denoise all frames
        
        # Extract action labels if using action conditioning
        action_labels = None
        subject_prototypes = None
        subject_ids = None
        initial_positions = None
        position_latents = None  # For VideoJAM mode
        frame_coords = None  # For coord_token_mode
        subject_mask = None  # For multi-game variable player counts
        
        # Pixel position mode: load raw pixel position maps
        position_pixels = None
        if getattr(self.config, 'pixel_position_mode', False):
            if "position_pixels" in batch:
                position_pixels = batch["position_pixels"].to(device=self.device, dtype=self.dtype)
                # position_pixels: [B, S, T, H, W]
            if "actions" in batch:
                action_labels = batch["actions"].to(device=self.device, dtype=torch.long)
            if "subject_ids" in batch:
                subject_ids = batch["subject_ids"].to(device=self.device, dtype=torch.long)
            if "initial_positions" in batch:
                initial_positions = batch["initial_positions"].to(device=self.device, dtype=self.dtype)
        
        # Position token mode: load per-subject VAE-encoded position latents
        elif getattr(self.config, 'position_token_mode', False):
            if "position_latents" in batch:
                position_latents = batch["position_latents"].to(device=self.device, dtype=self.dtype)
                # position_latents can be:
                # - [B, S, T, C, H, W] for multi-subject (S > 1)
                # - [B, T, C, H, W] for single subject (S = 1, after squeezing)
                # The model expects [B, S, T, C, H, W] or [B, T, C, H, W]
            if "actions" in batch:
                action_labels = batch["actions"].to(device=self.device, dtype=torch.long)
            if "subject_ids" in batch:
                subject_ids = batch["subject_ids"].to(device=self.device, dtype=torch.long)
            if "initial_positions" in batch:
                initial_positions = batch["initial_positions"].to(device=self.device, dtype=self.dtype)
        
        # Coordinate token mode: load per-frame coordinates from dataset
        elif getattr(self.config, 'coord_token_mode', False):
            if "frame_coords" in batch:
                frame_coords = batch["frame_coords"].to(device=self.device, dtype=self.dtype)
                # frame_coords: [B, T, S, 2] - normalized coordinates
            if "actions" in batch:
                action_labels = batch["actions"].to(device=self.device, dtype=torch.long)
            if "subject_ids" in batch:
                subject_ids = batch["subject_ids"].to(device=self.device, dtype=torch.long)
            if "initial_positions" in batch:
                initial_positions = batch["initial_positions"].to(device=self.device, dtype=self.dtype)
            if "subject_mask" in batch and getattr(self.config, 'use_subject_mask', False):
                subject_mask = batch["subject_mask"].to(device=self.device)
                # subject_mask: [B, S] boolean, True for real subjects, False for padding
        
        # VideoJAM mode: load VAE-encoded position latents from dataset
        elif getattr(self.config, 'videojam_mode', False):
            if "position_latents" in batch:
                position_latents = batch["position_latents"].to(device=self.device, dtype=self.dtype)
                # Squeeze if needed: [B, 1, T, C, H, W] -> [B, T, C, H, W]
                if position_latents.dim() == 6:
                    position_latents = position_latents.squeeze(1)
            if "actions" in batch:
                action_labels = batch["actions"].to(device=self.device, dtype=torch.long)
            if "subject_ids" in batch:
                subject_ids = batch["subject_ids"].to(device=self.device, dtype=torch.long)
            if "initial_positions" in batch:
                initial_positions = batch["initial_positions"].to(device=self.device, dtype=self.dtype)
        
        if getattr(self.config, 'use_spatial_rope_conditioning', False):
            # Spatial RoPE conditioning: actions, subject_ids, and initial_positions
            if "actions" in batch:
                action_labels = batch["actions"].to(device=self.device, dtype=torch.long)
            if "subject_ids" in batch:
                subject_ids = batch["subject_ids"].to(device=self.device, dtype=torch.long)
            if "initial_positions" in batch:
                initial_positions = batch["initial_positions"].to(device=self.device, dtype=self.dtype)
        elif getattr(self.config, 'use_position_aware_conditioning', False):
            # Position-aware multi-subject: actions, subject_ids, and initial_positions
            if "actions" in batch:
                action_labels = batch["actions"].to(device=self.device, dtype=torch.long)
            if "subject_ids" in batch:
                subject_ids = batch["subject_ids"].to(device=self.device, dtype=torch.long)
            if "initial_positions" in batch:
                initial_positions = batch["initial_positions"].to(device=self.device, dtype=self.dtype)
        elif getattr(self.config, 'use_simple_multi_subject_conditioning', False) or getattr(self.config, 'use_modulated_multi_subject_conditioning', False):
            # Simple or modulated multi-subject: actions are [B, S, T] and we have subject_ids [B, S]
            if "actions" in batch:
                action_labels = batch["actions"].to(device=self.device, dtype=torch.long)
            if "subject_ids" in batch:
                subject_ids = batch["subject_ids"].to(device=self.device, dtype=torch.long)
        elif getattr(self.config, 'use_multi_subject_conditioning', False):
            # Multi-subject with image pooling: actions are [B, S, T] and we have subject_prototypes [B, S, C, H, W]
            if "actions" in batch:
                action_labels = batch["actions"].to(device=self.device, dtype=torch.long)
            if "subject_prototypes" in batch:
                subject_prototypes = batch["subject_prototypes"].to(device=self.device, dtype=self.dtype)
        elif getattr(self.config, 'load_actions', False):
            if "actions" in batch:
                action_labels = batch["actions"].to(device=self.device, dtype=torch.long)
            else:
                if self.is_main_process:
                    print(f"[WARNING] load_actions=True but 'actions' not in batch! Keys: {list(batch.keys())}")
        
        # Text action conditioning: load actions & coords for dynamic text prompt construction
        pre_sampled_num_clean = None
        if getattr(self.config, 'text_action_conditioning', False):
            _tac_actions = batch["actions"].to(device=self.device, dtype=torch.long)  # [B, S, num_steps]
            _tac_frame_coords = batch["frame_coords"].to(device=self.device, dtype=self.dtype)  # [B, T, S, 2]
            _tac_game_names = batch["game_name"]  # list of strings
            _tac_num_real = batch["num_real_subjects"].to(device=self.device)  # [B]

            num_frame = self.config.image_or_video_shape[1]
            _tac_batch_size = _tac_actions.shape[0]
            pre_sampled_num_clean = torch.randint(1, num_frame, (_tac_batch_size,), device=self.device)

            from game_action_text import build_action_text, build_multi_frame_action_text
            _use_history = getattr(self.config, 'text_action_history', False)
            _compact = getattr(self.config, 'compact_action_text', False)
            text_prompts = list(text_prompts)  # ensure mutable list
            for b in range(_tac_batch_size):
                nc = pre_sampled_num_clean[b].item()
                if _use_history:
                    action_text = build_multi_frame_action_text(
                        _tac_game_names[b], _tac_actions[b],
                        _tac_frame_coords[b], _tac_num_real[b].item(),
                        num_frames=nc,
                        compact=_compact,
                    )
                else:
                    subj_actions = _tac_actions[b, :, nc - 1]
                    subj_positions = _tac_frame_coords[b, nc - 1]
                    action_text = build_action_text(
                        _tac_game_names[b], subj_actions, subj_positions, _tac_num_real[b].item(),
                        compact=_compact,
                    )
                text_prompts[b] = text_prompts[b] + " " + action_text

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        # Look up pre-encoded game text embeddings for video token cross-attention
        video_text_embeds = None
        if self.game_text_cache is not None:
            text_list = []
            for prompt in text_prompts:
                if prompt in self.game_text_cache:
                    text_list.append(self.game_text_cache[prompt])
                else:
                    # Fallback: use first cached embedding (shouldn't happen with proper descriptions)
                    if not hasattr(self, '_warned_missing_text'):
                        print(f"  WARNING: prompt not in cache: '{prompt[:50]}...', using null embedding fallback")
                        self._warned_missing_text = True
                    text_list.append(next(iter(self.game_text_cache.values())))
            video_text_embeds = torch.cat(text_list, dim=0).to(device=self.device, dtype=self.dtype)  # [B, 512, 4096]
        
        with torch.no_grad():
            if self.model.text_encoder is not None:
                conditional_dict = self.model.text_encoder(
                    text_prompts=text_prompts)

                if not getattr(self, "unconditional_dict", None):
                    unconditional_dict = self.model.text_encoder(
                        text_prompts=[self.config.negative_prompt] * batch_size)
                    unconditional_dict = {k: v.detach()
                                          for k, v in unconditional_dict.items()}
                    self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
                else:
                    unconditional_dict = self.unconditional_dict
            else:
                # Unconditional: create dummy conditioning (zeros)
                # Check if cached dict exists and has correct batch size
                if (not hasattr(self, "dummy_conditional_dict") or 
                    self.dummy_conditional_dict is None or
                    self.dummy_conditional_dict["prompt_embeds"].shape[0] != batch_size):
                    # Create dummy text embeddings (zeros) with correct key name
                    # Use the model's expected text dimension (NOT the hidden dim)
                    text_dim = self.model.generator.model.text_dim
                    text_len = self.model.generator.model.text_len
                    dummy_prompt_embeds = torch.zeros(
                        [batch_size, text_len, text_dim],  # [batch, text_len, text_dim]
                        device=self.device, 
                        dtype=self.dtype
                    )
                    self.dummy_conditional_dict = {
                        "prompt_embeds": dummy_prompt_embeds
                    }
                conditional_dict = self.dummy_conditional_dict
                unconditional_dict = self.dummy_conditional_dict

        # Step 3: Train the generator
        # Register attention hooks for periodic logging
        should_log_attention = (
            self.attention_logger is not None and 
            self.step % self.attention_log_interval == 0
        )
        if should_log_attention:
            self.attention_logger.register_hooks()
            self.attention_logger.clear()
        
        generator_loss, log_dict = self.model.generator_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent,  # None for unconditional, first frame for i2v
            action_labels=action_labels,  # Action class indices for action conditioning
            subject_prototypes=subject_prototypes,  # For multi-subject conditioning (image pooling)
            subject_ids=subject_ids,  # For simple multi-subject conditioning (learned embeddings)
            initial_positions=initial_positions,  # For position-aware conditioning
            position_latents=position_latents,  # For VideoJAM mode (VAE-encoded position maps)
            frame_coords=frame_coords,  # For coord_token_mode
            position_pixels=position_pixels,  # For pixel_position_mode (raw pixel position maps)
            subject_mask=subject_mask,  # For multi-game variable player counts
            video_text_embeds=video_text_embeds,  # Pre-encoded game text for video cross-attn [B, 512, 4096]
            pre_sampled_num_clean=pre_sampled_num_clean,  # For text action conditioning
        )
        
        # Collect attention norms and remove hooks
        attention_log_dict = {}
        if should_log_attention:
            attention_log_dict = self.attention_logger.get_log_dict()
            # Also get detailed per-block logs every 500 steps
            if self.step % 500 == 0:
                attention_log_dict.update(self.attention_logger.get_detailed_log_dict())
            self.attention_logger.remove_hooks()
        self.generator_optimizer.zero_grad()
        if self.multi_subject_optimizer is not None:
            self.multi_subject_optimizer.zero_grad()
        generator_loss.backward()
        
        # Log action embedding gradient norm (before clipping)
        action_embedding_grad_norm = None
        if hasattr(self.model.generator.model, 'action_embedding'):
            action_embedding_params = list(self.model.generator.model.action_embedding.parameters())
            if len(action_embedding_params) > 0 and action_embedding_params[0].grad is not None:
                grad_norms = []
                for p in action_embedding_params:
                    if p.grad is not None:
                        grad_norms.append(p.grad.detach().norm().item())
                if grad_norms:
                    action_embedding_grad_norm = sum(grad_norms)
        
        # Apply LR warmup if configured
        # Linear warmup from min_lr to full_lr over warmup_steps
        if self.lr_warmup_steps > 0 and self.step < self.lr_warmup_steps:
            warmup_progress = (self.step + 1) / self.lr_warmup_steps  # 0 to 1
            warmup_factor = self.lr_warmup_min_ratio + (1.0 - self.lr_warmup_min_ratio) * warmup_progress
            for param_group in self.generator_optimizer.param_groups:
                base_lr = param_group.get('_base_lr', self.base_lr)
                param_group['lr'] = base_lr * warmup_factor
            if self.multi_subject_optimizer is not None:
                for param_group in self.multi_subject_optimizer.param_groups:
                    base_lr = param_group.get('_base_lr', self.base_lr)
                    param_group['lr'] = base_lr * warmup_factor
            if self.single_subject_optimizer is not None:
                for param_group in self.single_subject_optimizer.param_groups:
                    base_lr = param_group.get('_base_lr', self.base_lr)
                    param_group['lr'] = base_lr * warmup_factor
            # Log warmup progress occasionally
            if self.step % 20 == 0 and self.is_main_process:
                current_lr = self.generator_optimizer.param_groups[0]['lr']
                print(f"[Warmup step {self.step}/{self.lr_warmup_steps}] LR: {current_lr:.2e}")
        elif self.lr_warmup_steps > 0 and self.step == self.lr_warmup_steps:
            # Warmup complete - log and continue (cosine decay will handle LR from here)
            if self.is_main_process:
                if self.lr_cosine_decay:
                    print(f"LR warmup complete at step {self.step}, starting cosine decay from {self.base_lr:.2e}")
                else:
                    print(f"LR warmup complete at step {self.step}, using full LR: {self.base_lr:.2e}")
        
        # Apply cosine decay after warmup (or from start if no warmup)
        if self.lr_cosine_decay and self.step >= self.lr_warmup_steps:
            import math
            # Progress through decay phase (0 to 1)
            decay_steps = self.num_iter - self.lr_warmup_steps
            decay_progress = (self.step - self.lr_warmup_steps) / max(decay_steps, 1)
            decay_progress = min(decay_progress, 1.0)  # Clamp to 1
            # Cosine decay from 1.0 to lr_min_ratio
            cosine_factor = self.lr_min_ratio + 0.5 * (1.0 - self.lr_min_ratio) * (1 + math.cos(math.pi * decay_progress))
            
            for param_group in self.generator_optimizer.param_groups:
                base_lr = param_group.get('_base_lr', self.base_lr)
                param_group['lr'] = base_lr * cosine_factor
            if self.multi_subject_optimizer is not None:
                for param_group in self.multi_subject_optimizer.param_groups:
                    base_lr = param_group.get('_base_lr', self.base_lr)
                    param_group['lr'] = base_lr * cosine_factor
            if self.single_subject_optimizer is not None:
                for param_group in self.single_subject_optimizer.param_groups:
                    base_lr = param_group.get('_base_lr', self.base_lr)
                    param_group['lr'] = base_lr * cosine_factor
            
            # Log decay progress occasionally
            if self.step % 50 == 0 and self.is_main_process:
                current_lr = self.generator_optimizer.param_groups[0]['lr']
                print(f"[Cosine decay step {self.step}] LR: {current_lr:.2e}")
        
        generator_grad_norm = self.model.generator.clip_grad_norm_(
            self.max_grad_norm)
        self.generator_optimizer.step()
        
        # Step multi-subject optimizer (with gradient clipping)
        if self.multi_subject_optimizer is not None:
            torch.nn.utils.clip_grad_norm_(
                self.multi_subject_conditioner.parameters(), self.max_grad_norm)
            self.multi_subject_optimizer.step()
        
        # Step single-subject optimizer (with gradient clipping)
        if self.single_subject_optimizer is not None:
            torch.nn.utils.clip_grad_norm_(
                self.single_subject_conditioner.parameters(), self.max_grad_norm)
            self.single_subject_optimizer.step()

        # Increment the step since we finished gradient update
        self.step += 1

        wandb_loss_dict = {
            "generator_loss": generator_loss.item(),
            "generator_grad_norm": generator_grad_norm.item(),
        }
        
        # Add individual loss components from log_dict (e.g., loss_video, loss_position, loss_coord)
        if "loss_video" in log_dict:
            wandb_loss_dict["loss_video"] = log_dict["loss_video"]
        if "loss_position" in log_dict:
            wandb_loss_dict["loss_position"] = log_dict["loss_position"]
        if "loss_coord" in log_dict:
            wandb_loss_dict["loss_coord"] = log_dict["loss_coord"]
        if "loss_position_token" in log_dict:
            wandb_loss_dict["loss_position_token"] = log_dict["loss_position_token"]
        if "timestep" in log_dict:
            wandb_loss_dict["timestep"] = log_dict["timestep"]
        
        # Add action embedding grad norm to logs if available
        if action_embedding_grad_norm is not None:
            wandb_loss_dict["action_embedding_grad_norm"] = action_embedding_grad_norm

        # Add multi-subject conditioner weight norms to logs
        if hasattr(self, 'multi_subject_conditioner'):
            conditioner = self.multi_subject_conditioner
            # Pool query norm
            if hasattr(conditioner, 'pooler') and hasattr(conditioner.pooler, 'pool_query'):
                wandb_loss_dict["conditioner/pool_query_norm"] = conditioner.pooler.pool_query.norm().item()
            # Subject projection weight norm
            if hasattr(conditioner, 'action_subject_embed') and hasattr(conditioner.action_subject_embed, 'subject_proj'):
                wandb_loss_dict["conditioner/subject_proj_norm"] = conditioner.action_subject_embed.subject_proj.weight.norm().item()
            # Action embedding weight norm
            if hasattr(conditioner, 'action_subject_embed') and hasattr(conditioner.action_subject_embed, 'action_embedding'):
                wandb_loss_dict["conditioner/action_embedding_norm"] = conditioner.action_subject_embed.action_embedding.weight.norm().item()
            # Pooler v_proj weight norm (most important for learning subject features)
            if hasattr(conditioner, 'pooler') and hasattr(conditioner.pooler, 'v_proj'):
                wandb_loss_dict["conditioner/v_proj_norm"] = conditioner.pooler.v_proj.weight.norm().item()
            # Grad norm for multi-subject conditioner
            total_grad_norm = 0.0
            for p in conditioner.parameters():
                if p.grad is not None:
                    total_grad_norm += p.grad.data.norm(2).item() ** 2
            wandb_loss_dict["conditioner/grad_norm"] = total_grad_norm ** 0.5

        # Log position embedding/head weight norms (for joint position diffusion)
        if hasattr(self.model.generator, 'model'):
            model = self.model.generator.model
            if hasattr(model, 'position_embedding') and model.position_embedding is not None:
                wandb_loss_dict["position/embedding_weight_norm"] = model.position_embedding.weight.norm().item()
                # Also log grad norm
                if model.position_embedding.weight.grad is not None:
                    wandb_loss_dict["position/embedding_grad_norm"] = model.position_embedding.weight.grad.norm().item()
            if hasattr(model, 'position_head') and model.position_head is not None:
                # Position head has a 'head' sub-module
                if hasattr(model.position_head, 'head'):
                    wandb_loss_dict["position/head_weight_norm"] = model.position_head.head.weight.norm().item()
                    if model.position_head.head.weight.grad is not None:
                        wandb_loss_dict["position/head_grad_norm"] = model.position_head.head.weight.grad.norm().item()
            # VideoJAM mode: log Win+/Wout+ projection weights
            if hasattr(model, 'win_proj') and model.win_proj is not None:
                wandb_loss_dict["videojam/win_proj_weight_norm"] = model.win_proj.weight.norm().item()
                if model.win_proj.weight.grad is not None:
                    wandb_loss_dict["videojam/win_proj_grad_norm"] = model.win_proj.weight.grad.norm().item()
            if hasattr(model, 'wout_proj') and model.wout_proj is not None:
                wandb_loss_dict["videojam/wout_proj_weight_norm"] = model.wout_proj.weight.norm().item()
                if model.wout_proj.weight.grad is not None:
                    wandb_loss_dict["videojam/wout_proj_grad_norm"] = model.wout_proj.weight.grad.norm().item()
            
            # Coordinate token mode: log encoder/decoder weight and grad norms
            if hasattr(model, 'coord_encoder') and model.coord_encoder is not None:
                # Handle both Linear encoder (has .weight) and Fourier encoder (has .proj.weight)
                if hasattr(model.coord_encoder, 'weight'):
                    # Linear encoder
                    wandb_loss_dict["coord/encoder_weight_norm"] = model.coord_encoder.weight.norm().item()
                    if model.coord_encoder.weight.grad is not None:
                        wandb_loss_dict["coord/encoder_grad_norm"] = model.coord_encoder.weight.grad.norm().item()
                elif hasattr(model.coord_encoder, 'proj') and model.coord_encoder.proj is not None:
                    # Fourier encoder with projection
                    wandb_loss_dict["coord/encoder_proj_weight_norm"] = model.coord_encoder.proj.weight.norm().item()
                    if model.coord_encoder.proj.weight.grad is not None:
                        wandb_loss_dict["coord/encoder_proj_grad_norm"] = model.coord_encoder.proj.weight.grad.norm().item()
            if hasattr(model, 'coord_decoder') and model.coord_decoder is not None:
                wandb_loss_dict["coord/decoder_weight_norm"] = model.coord_decoder.weight.norm().item()
                if model.coord_decoder.weight.grad is not None:
                    wandb_loss_dict["coord/decoder_grad_norm"] = model.coord_decoder.weight.grad.norm().item()
            # Also log coord encoder LayerNorm if present
            if hasattr(model, 'coord_encoder_norm') and model.coord_encoder_norm is not None:
                wandb_loss_dict["coord/encoder_layernorm_weight_norm"] = model.coord_encoder_norm.weight.norm().item()

        # Step 4: Logging
        if self.is_main_process:
            # Console logging
            log_msg = f"[Step {self.step}] Loss: {generator_loss.item():.6f}, Grad Norm: {generator_grad_norm.item():.4f}"
            # Add individual loss components for joint position diffusion or coord token mode
            if "loss_video" in log_dict:
                if "loss_position" in log_dict:
                    log_msg += f", L_vid: {log_dict['loss_video']:.6f}, L_pos: {log_dict['loss_position']:.6f}"
                elif "loss_coord" in log_dict:
                    log_msg += f", L_vid: {log_dict['loss_video']:.6f}, L_coord: {log_dict['loss_coord']:.6f}"
                elif "loss_position_token" in log_dict:
                    log_msg += f", L_vid: {log_dict['loss_video']:.6f}, L_pos_tok: {log_dict['loss_position_token']:.6f}"
            elif "loss_coord" in log_dict:
                # Coord-only mode: only coord loss
                log_msg += f", L_coord: {log_dict['loss_coord']:.6f}"
            if "timestep" in log_dict:
                log_msg += f", t={log_dict['timestep']:.0f}"
            if action_embedding_grad_norm is not None:
                log_msg += f", Action Emb Grad: {action_embedding_grad_norm:.4f}"
            # Add attention ratio to console log periodically
            if attention_log_dict and "attn/cross_self_ratio" in attention_log_dict:
                log_msg += f", Cross/Self: {attention_log_dict['attn/cross_self_ratio']:.4f}"
            print(log_msg)
            
            # WandB logging
            if not self.disable_wandb:
                # Merge attention logs into wandb dict
                wandb_loss_dict.update(attention_log_dict)
                wandb.log(wandb_loss_dict, step=self.step)

        if self.step % self.config.gc_interval == 0:
            if dist.get_rank() == 0:
                logging.info("DistGarbageCollector: Running GC.")
            gc.collect()

        # Step 5. Update EMA params
        # Initialize EMA if we've passed the start step and it doesn't exist yet
        if self.generator_ema is None and self.step >= self.config.ema_start_step and self.config.ema_weight > 0.0:
            if self.is_main_process:
                print(f"Step {self.step}: Initializing EMA model with weight {self.config.ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)
        
        if self.generator_ema is not None:
            self.generator_ema.update(self.model.generator)

    def generate_video(self, pipeline, prompts, image=None):
        batch_size = len(prompts)
        sampled_noise = torch.randn(
            [batch_size, 21, 16, 60, 104], device="cuda", dtype=self.dtype
        )
        video, _ = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True
        )
        current_video = video.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0
        return current_video

    def train(self):
        while True:
            batch = next(self.dataloader)
            self.train_one_step(batch)
            if (not self.config.no_save) and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            barrier()
            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time
