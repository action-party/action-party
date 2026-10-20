"""LMDB dataset loaders used for Melting Pot and legacy shapes experiments.

Key class for the headline exp-139 training run is
:class:`MultiGameLMDBDataset`, which iterates over a directory of per-game
LMDBs produced by ``scripts/dataset/create_all_games_dataset.py``. It returns
VAE-encoded video latents, per-player actions, per-frame coordinates,
orientations, subject masks (so padded subjects do not contribute to loss),
and per-game text prompts drawn from :mod:`utils.game_descriptions`.

The other loaders (:class:`ShardingLMDBDataset`, :class:`MultiSubjectLMDBDataset`,
:class:`PositionTokenLMDBDataset`, :class:`PixelPositionLMDBDataset`) are kept
for the ablation configs exp-131/134/141-145 which use smaller datasets.
"""
from utils.lmdb import get_array_shape_from_lmdb, retrieve_row_from_lmdb
from torch.utils.data import Dataset
import numpy as np
import torch
import lmdb
import json
from pathlib import Path
from PIL import Image
import os


class TextDataset(Dataset):
    def __init__(self, prompt_path, extended_prompt_path=None):
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]

        if extended_prompt_path is not None:
            with open(extended_prompt_path, encoding="utf-8") as f:
                self.extended_prompt_list = [line.rstrip() for line in f]
            assert len(self.extended_prompt_list) == len(self.prompt_list)
        else:
            self.extended_prompt_list = None

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        batch = {
            "prompts": self.prompt_list[idx],
            "idx": idx,
        }
        if self.extended_prompt_list is not None:
            batch["extended_prompts"] = self.extended_prompt_list[idx]
        return batch


class ODERegressionLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.env = lmdb.open(data_path, readonly=True,
                             lock=False, readahead=False, meminit=False)

        self.latents_shape = get_array_shape_from_lmdb(self.env, 'latents')
        self.max_pair = max_pair

    def __len__(self):
        return min(self.latents_shape[0], self.max_pair)

    def __getitem__(self, idx):
        """
        Outputs:
            - prompts: List of Strings
            - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        latents = retrieve_row_from_lmdb(
            self.env,
            "latents", np.float16, idx, shape=self.latents_shape[1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.env,
            "prompts", str, idx
        )
        return {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32)
        }


class ShardingLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8), load_actions: bool = False, load_subjects: bool = False):
        self.envs = []
        self.index = []
        self.load_actions = load_actions
        self.load_subjects = load_subjects

        for fname in sorted(os.listdir(data_path)):
            # Only open directories that look like LMDB shards
            # Matches: *.lmdb, shard_*, shard0, shard1, *_lmdb
            is_lmdb_shard = (
                fname.endswith('.lmdb') or 
                fname.startswith('shard_') or 
                (fname.startswith('shard') and len(fname) > 5 and fname[5:].isdigit()) or  # shard0, shard1, etc.
                fname.endswith('_lmdb')
            )
            if not is_lmdb_shard:
                continue
            path = os.path.join(data_path, fname)
            if not os.path.isdir(path):
                continue
            env = lmdb.open(path,
                            readonly=True,
                            lock=False,
                            readahead=False,
                            meminit=False)
            self.envs.append(env)

        self.latents_shape = [None] * len(self.envs)
        for shard_id, env in enumerate(self.envs):
            self.latents_shape[shard_id] = get_array_shape_from_lmdb(env, 'latents')
            for local_i in range(self.latents_shape[shard_id][0]):
                self.index.append((shard_id, local_i))

            # print("shard_id ", shard_id, " local_i ", local_i)

        self.max_pair = max_pair

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        """
            Outputs:
                - prompts: List of Strings
                - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
                - actions: (optional) Int action class index if load_actions=True
        """
        shard_id, local_idx = self.index[idx]

        latents = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "latents", np.float16, local_idx,
            shape=self.latents_shape[shard_id][1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "prompts", str, local_idx
        )

        result = {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32)
        }
        
        # Load action labels if requested (for action-conditional training)
        if self.load_actions:
            try:
                actions = retrieve_row_from_lmdb(
                    self.envs[shard_id],
                    "actions", np.int64, local_idx
                )
                # Squeeze to scalar: [1] -> [] (will be [B] after batching)
                result["actions"] = torch.tensor(actions, dtype=torch.long).squeeze()
            except:
                # If actions not found, skip (backward compatibility)
                pass
        
        # Load subject labels if requested (for subject-conditional training)
        if self.load_subjects:
            try:
                subjects = retrieve_row_from_lmdb(
                    self.envs[shard_id],
                    "subjects", np.int64, local_idx
                )
                # Squeeze to scalar: [1] -> [] (will be [B] after batching)
                result["subjects"] = torch.tensor(subjects, dtype=torch.long).squeeze()
            except:
                # If subjects not found, skip (backward compatibility)
                pass
        
        return result


class TextImagePairDataset(Dataset):
    def __init__(
        self,
        data_dir,
        transform=None,
        eval_first_n=-1,
        pad_to_multiple_of=None
    ):
        """
        Args:
            data_dir (str): Path to the directory containing:
                - target_crop_info_*.json (metadata file)
                - */ (subdirectory containing images with matching aspect ratio)
            transform (callable, optional): Optional transform to be applied on the image
        """
        self.transform = transform
        data_dir = Path(data_dir)

        # Find the metadata JSON file
        metadata_files = list(data_dir.glob('target_crop_info_*.json'))
        if not metadata_files:
            raise FileNotFoundError(f"No metadata file found in {data_dir}")
        if len(metadata_files) > 1:
            raise ValueError(f"Multiple metadata files found in {data_dir}")

        metadata_path = metadata_files[0]
        # Extract aspect ratio from metadata filename (e.g. target_crop_info_26-15.json -> 26-15)
        aspect_ratio = metadata_path.stem.split('_')[-1]

        # Use aspect ratio subfolder for images
        self.image_dir = data_dir / aspect_ratio
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        eval_first_n = eval_first_n if eval_first_n != -1 else len(self.metadata)
        self.metadata = self.metadata[:eval_first_n]

        # Verify all images exist
        for item in self.metadata:
            image_path = self.image_dir / item['file_name']
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

        self.dummy_prompt = "DUMMY PROMPT"
        self.pre_pad_len = len(self.metadata)
        if pad_to_multiple_of is not None and len(self.metadata) % pad_to_multiple_of != 0:
            # Duplicate the last entry
            self.metadata += [self.metadata[-1]] * (
                pad_to_multiple_of - len(self.metadata) % pad_to_multiple_of
            )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Returns:
            dict: A dictionary containing:
                - image: PIL Image
                - caption: str
                - target_bbox: list of int [x1, y1, x2, y2]
                - target_ratio: str
                - type: str
                - origin_size: tuple of int (width, height)
        """
        item = self.metadata[idx]

        # Load image
        image_path = self.image_dir / item['file_name']
        image = Image.open(image_path).convert('RGB')

        # Apply transform if specified
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'prompts': item['caption'],
            'target_bbox': item['target_crop']['target_bbox'],
            'target_ratio': item['target_crop']['target_ratio'],
            'type': item['type'],
            'origin_size': (item['origin_width'], item['origin_height']),
            'idx': idx
        }


class MultiSubjectLMDBDataset(Dataset):
    """
    Dataset for multi-subject action-conditioned videos.
    
    Each sample contains:
        - ode_latent: [F, C, H, W] video latents
        - prompts: text prompt
        - subject_ids: [S] which subjects are in this video
        - actions: [S, N-1] per-subject action sequences
        - subject_prototypes: [S, C, H', W'] prototype latents for subjects in this video
    """
    
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.envs = []
        self.index = []
        
        for fname in sorted(os.listdir(data_path)):
            # Only open directories that look like LMDB shards
            # Matches: *.lmdb, shard_*, shard0, shard1, *_lmdb
            is_lmdb_shard = (
                fname.endswith('.lmdb') or 
                fname.startswith('shard_') or 
                fname.startswith('shard') and fname[5:].isdigit() or  # shard0, shard1, etc.
                fname.endswith('_lmdb')
            )
            if not is_lmdb_shard:
                continue
            path = os.path.join(data_path, fname)
            if not os.path.isdir(path):
                continue
            env = lmdb.open(path,
                            readonly=True,
                            lock=False,
                            readahead=False,
                            meminit=False)
            self.envs.append(env)
        
        self.latents_shape = [None] * len(self.envs)
        for shard_id, env in enumerate(self.envs):
            self.latents_shape[shard_id] = get_array_shape_from_lmdb(env, 'latents')
            for local_i in range(self.latents_shape[shard_id][0]):
                self.index.append((shard_id, local_i))
        
        self.max_pair = max_pair
        
        # Load subject prototypes (shared across all samples, stored in first shard)
        self._load_subject_prototypes()
    
    def _load_subject_prototypes(self):
        """Load all subject prototype latents and metadata from the first shard."""
        env = self.envs[0]
        with env.begin(write=False) as txn:
            # Get prototype shape (optional - not needed for simple/modulated conditioners)
            shape_str = txn.get('subject_prototypes_shape'.encode())
            if shape_str is not None:
                shape = tuple(int(x) for x in shape_str.decode().split())
                
                # Load prototype data
                proto_bytes = txn.get('subject_prototypes_data'.encode())
                if proto_bytes is not None:
                    prototypes = np.frombuffer(proto_bytes, dtype=np.float16).reshape(shape)
                    self.subject_prototypes = torch.tensor(prototypes, dtype=torch.float32)
                    print(f"Loaded {self.subject_prototypes.shape[0]} subject prototypes, "
                          f"shape: {self.subject_prototypes.shape}")
                else:
                    self.subject_prototypes = None
                    print("No subject prototype data found (using learned embeddings instead)")
            else:
                self.subject_prototypes = None
                print("No subject prototypes in dataset (using learned embeddings instead)")
            
            # Get number of subjects per video
            num_subj_str = txn.get('num_subjects_per_video'.encode())
            self.num_subjects_per_video = int(num_subj_str.decode()) if num_subj_str else 2
            
            # Alternative: check 'num_subjects' key
            if num_subj_str is None:
                num_subj_str = txn.get('num_subjects'.encode())
                self.num_subjects_per_video = int(num_subj_str.decode()) if num_subj_str else 2
            
            # Get subject_ids shape: [num_samples, num_subjects_per_video]
            subj_ids_shape_str = txn.get('subject_ids_shape'.encode())
            if subj_ids_shape_str:
                subj_shape = tuple(int(x) for x in subj_ids_shape_str.decode().split())
                self.subject_ids_per_sample = subj_shape[1] if len(subj_shape) > 1 else self.num_subjects_per_video
            else:
                self.subject_ids_per_sample = self.num_subjects_per_video
            
            # Get actions shape: [num_samples, num_subjects, num_actions]
            actions_shape_str = txn.get('actions_shape'.encode())
            if actions_shape_str:
                actions_shape = tuple(int(x) for x in actions_shape_str.decode().split())
                # Shape is [num_samples, S, N-1], we want per-sample shape [S, N-1]
                self.actions_shape_per_sample = actions_shape[1:]
            else:
                self.actions_shape_per_sample = None
            
            # Check for initial positions: [num_samples, S, 2] for (x, y)
            positions_shape_str = txn.get('initial_positions_shape'.encode())
            if positions_shape_str:
                positions_shape = tuple(int(x) for x in positions_shape_str.decode().split())
                # Shape is [num_samples, S, 2], we want per-sample shape [S, 2]
                self.positions_shape_per_sample = positions_shape[1:]
                self.has_positions = True
                print(f"Dataset has initial positions: shape {self.positions_shape_per_sample}")
            else:
                self.positions_shape_per_sample = None
                self.has_positions = False
            
            # Get speed and image_size for position-based training
            speed_str = txn.get('speed'.encode())
            self.speed = int(speed_str.decode()) if speed_str else 2
            
            image_size_str = txn.get('image_size'.encode())
            self.image_size = int(image_size_str.decode()) if image_size_str else 256
            
            # Check for VideoJAM-style position latents (VAE-encoded position maps)
            videojam_str = txn.get('videojam_style'.encode())
            self.has_position_latents = videojam_str is not None and videojam_str.decode() == 'true'
            
            if self.has_position_latents:
                position_latents_shape_str = txn.get('position_latents_shape'.encode())
                if position_latents_shape_str:
                    shape = tuple(int(x) for x in position_latents_shape_str.decode().split())
                    self.position_latents_shape_per_sample = shape[1:]
                    print(f"VideoJAM dataset: position latents shape {self.position_latents_shape_per_sample}")
                else:
                    self.position_latents_shape_per_sample = None
                    self.has_position_latents = False
            else:
                self.position_latents_shape_per_sample = None
            
            print(f"Subjects per video: {self.num_subjects_per_video}, "
                  f"subject IDs per sample: {self.subject_ids_per_sample}")
    
    def __len__(self):
        return min(len(self.index), self.max_pair)
    
    def __getitem__(self, idx):
        shard_id, local_idx = self.index[idx]
        env = self.envs[shard_id]
        
        # Load video latents
        latents = retrieve_row_from_lmdb(
            env, "latents", np.float16, local_idx,
            shape=self.latents_shape[shard_id][1:]
        )
        if len(latents.shape) == 4:
            latents = latents[None, ...]
        
        # Load prompt
        prompts = retrieve_row_from_lmdb(env, "prompts", str, local_idx)
        
        # Load subject IDs for this video: [S]
        subject_ids = retrieve_row_from_lmdb(
            env, "subject_ids", np.int64, local_idx,
            shape=(self.subject_ids_per_sample,)
        )
        subject_ids = torch.tensor(subject_ids, dtype=torch.long)
        
        # Load actions for this video: [S, N-1]
        actions = retrieve_row_from_lmdb(
            env, "actions", np.int64, local_idx,
            shape=self.actions_shape_per_sample
        )
        actions = torch.tensor(actions, dtype=torch.long)
        
        result = {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32),
            "subject_ids": subject_ids,           # [S]
            "actions": actions,                   # [S, N-1]
        }
        
        # Get prototype latents if available (only for image-pooling conditioner)
        if self.subject_prototypes is not None:
            subject_protos = self.subject_prototypes[subject_ids]
            result["subject_prototypes"] = subject_protos  # [S, C, H', W']
        
        # Load initial positions if available (for position-aware conditioner)
        if self.has_positions and self.positions_shape_per_sample is not None:
            initial_positions = retrieve_row_from_lmdb(
                env, "initial_positions", np.float32, local_idx,
                shape=self.positions_shape_per_sample
            )
            result["initial_positions"] = torch.tensor(initial_positions, dtype=torch.float32)  # [S, 2]
        
        # Load VAE-encoded position latents if available (VideoJAM mode)
        if self.has_position_latents and self.position_latents_shape_per_sample is not None:
            position_latents = retrieve_row_from_lmdb(
                env, "position_latents", np.float16, local_idx,
                shape=self.position_latents_shape_per_sample
            )
            if len(position_latents.shape) == 4:
                position_latents = position_latents[None, ...]
            result["position_latents"] = torch.tensor(position_latents, dtype=torch.float32)  # [T, C, H, W]
        
        # Compute per-frame coordinates if we have initial positions and actions
        # This is for coord_token_mode: (x, y) per subject per frame, normalized to [0, 1]
        if self.has_positions and 'initial_positions' in result and 'actions' in result:
            num_frames = latents.shape[0] if len(latents.shape) == 4 else latents.shape[1]
            result["frame_coords"] = self._compute_frame_coords(
                result["initial_positions"], result["actions"], num_frames
            )
        
        return result
    
    def _compute_frame_coords(self, initial_positions, actions, num_frames):
        """
        Compute per-frame (x, y) coordinates for each subject.
        
        Args:
            initial_positions: [S, 2] initial (x, y) positions in pixels
            actions: [S, num_frames-1] action indices per subject
            num_frames: number of frames (latent frames, e.g., 9)
        
        Returns:
            frame_coords: [num_frames, S, 2] normalized coordinates in [0, 1]
        """
        # Action directions: action -> (dx, dy) in pixels
        ACTION_DELTAS = {
            0: (0, 0),      # still
            1: (0, -self.speed),     # up (y decreases)
            2: (0, self.speed),      # down (y increases)
            3: (-self.speed, 0),     # left (x decreases)
            4: (self.speed, 0)       # right (x increases)
        }
        
        S = initial_positions.shape[0]
        frame_coords = torch.zeros(num_frames, S, 2)
        
        # Frame 0: initial positions
        for s in range(S):
            frame_coords[0, s] = initial_positions[s]
        
        # Subsequent frames: apply actions
        for t in range(1, num_frames):
            for s in range(S):
                # Get action for this frame (actions[s, t-1] produces frame t)
                if t - 1 < actions.shape[1]:
                    action = actions[s, t - 1].item()
                else:
                    action = 0  # still if no more actions
                
                dx, dy = ACTION_DELTAS.get(action, (0, 0))
                prev_pos = frame_coords[t - 1, s]
                new_x = prev_pos[0] + dx
                new_y = prev_pos[1] + dy
                
                # Clamp to image bounds
                new_x = max(0, min(self.image_size - 1, new_x))
                new_y = max(0, min(self.image_size - 1, new_y))
                
                frame_coords[t, s, 0] = new_x
                frame_coords[t, s, 1] = new_y
        
        # Normalize to [0, 1]
        frame_coords = frame_coords / self.image_size
        
        return frame_coords  # [num_frames, S, 2]


class PositionTokenLMDBDataset(Dataset):
    """
    Dataset for position token diffusion mode.
    
    Loads pre-encoded video latents and per-subject position latents from LMDB.
    Dataset format uses per-field keys: video_latents_{i}_data, position_latents_{i}_data, etc.
    
    Each sample contains:
        - video_latents: [T, C, H, W] video latents from VAE
        - position_latents: [T, C, H, W] position map latents from VAE (grayscale RGB encoded)
        - actions: [num_actions] action sequence
        - initial_positions: [2] initial (x, y) position
    """
    
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        import pickle
        self.data_path = data_path
        self.max_pair = max_pair
        
        # Open LMDB shards
        self.envs = []
        self.index = []
        
        for fname in sorted(os.listdir(data_path)):
            if fname.endswith('.lmdb'):
                path = os.path.join(data_path, fname)
                env = lmdb.open(path,
                                readonly=True,
                                lock=False,
                                readahead=False,
                                meminit=False)
                self.envs.append(env)
        
        # Load metadata from first shard
        env = self.envs[0]
        with env.begin(write=False) as txn:
            # Get number of samples
            num_samples_bytes = txn.get(b'num_samples')
            if num_samples_bytes:
                self.num_samples_per_shard = [int(num_samples_bytes.decode())]
            else:
                # Fallback: count samples
                count = 0
                while txn.get(f'video_latents_{count}_data'.encode()) is not None:
                    count += 1
                self.num_samples_per_shard = [count]
            
            # Get shapes
            self.video_latents_shape = self._parse_shape(txn.get(b'video_latents_shape'))
            self.position_latents_shape = self._parse_shape(txn.get(b'position_latents_shape'))
            self.actions_shape = self._parse_shape(txn.get(b'actions_shape'))
            self.initial_positions_shape = self._parse_shape(txn.get(b'initial_positions_shape'))
            
            # Get num_subjects
            num_subj = txn.get(b'num_subjects')
            self.num_subjects = int(num_subj.decode()) if num_subj else 1
            
            # Get image_size and speed for coordinate computation
            image_size = txn.get(b'image_size')
            self.image_size = int(image_size.decode()) if image_size else 256
            speed = txn.get(b'speed')
            self.speed = int(speed.decode()) if speed else 16
        
        # Build index for all samples across shards
        for shard_id, env in enumerate(self.envs):
            with env.begin(write=False) as txn:
                num_samples_bytes = txn.get(b'num_samples')
                if num_samples_bytes:
                    num = int(num_samples_bytes.decode())
                else:
                    num = self.num_samples_per_shard[0] if shard_id == 0 else 0
                
                if shard_id < len(self.num_samples_per_shard):
                    self.num_samples_per_shard[shard_id] = num
                else:
                    self.num_samples_per_shard.append(num)
                
                for local_i in range(num):
                    self.index.append((shard_id, local_i))
        
        print(f"PositionTokenLMDBDataset: {len(self.index)} samples, "
              f"video_shape={self.video_latents_shape}, position_shape={self.position_latents_shape}, "
              f"num_subjects={self.num_subjects}")
    
    def _parse_shape(self, shape_bytes):
        """Parse shape string like '(9, 16, 32, 32)' to tuple."""
        if shape_bytes is None:
            return None
        shape_str = shape_bytes.decode()
        # Handle both "9 16 32 32" and "(9, 16, 32, 32)" formats
        if '(' in shape_str:
            shape_str = shape_str.strip('()')
            # Filter out empty strings (e.g., from "(8,)" -> "8,")
            parts = [int(x.strip()) for x in shape_str.split(',') if x.strip()]
        else:
            parts = [int(x) for x in shape_str.split() if x.strip()]
        return tuple(parts)
    
    def __len__(self):
        return min(len(self.index), self.max_pair)
    
    def __getitem__(self, idx):
        import pickle
        shard_id, local_idx = self.index[idx]
        env = self.envs[shard_id]
        
        with env.begin(write=False) as txn:
            # Load video latents
            video_bytes = txn.get(f'video_latents_{local_idx}_data'.encode())
            video_latents = pickle.loads(video_bytes)  # [T, C, H, W]
            
            # Load position latents
            position_bytes = txn.get(f'position_latents_{local_idx}_data'.encode())
            position_latents = pickle.loads(position_bytes)  # [T, C, H, W]
            
            # Load actions
            actions_bytes = txn.get(f'actions_{local_idx}_data'.encode())
            actions = pickle.loads(actions_bytes)  # [num_actions]
            
            # Load initial positions
            positions_bytes = txn.get(f'initial_positions_{local_idx}_data'.encode())
            initial_positions = pickle.loads(positions_bytes)  # [2] or [S, 2]
            
            # Load pre-computed frame coords if available (more reliable than recomputing)
            frame_coords_bytes = txn.get(f'frame_coords_{local_idx}_data'.encode())
            stored_frame_coords = pickle.loads(frame_coords_bytes) if frame_coords_bytes else None
        
        # Convert to tensors
        video_latents = torch.tensor(video_latents, dtype=torch.float32)
        position_latents = torch.tensor(position_latents, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        initial_positions = torch.tensor(initial_positions, dtype=torch.float32)
        
        # Ensure initial_positions is [S, 2]
        if initial_positions.dim() == 1:
            initial_positions = initial_positions.unsqueeze(0)  # [1, 2]
        
        # Ensure actions is [S, num_actions]
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)  # [1, num_actions]
        
        # Ensure position_latents is [S, T, C, H, W]
        # Can be [T, C, H, W] (1 subject) or [S, T, C, H, W] (multi-subject)
        if position_latents.dim() == 4:
            position_latents = position_latents.unsqueeze(0)  # [1, T, C, H, W]
        
        # Create subject_ids (just sequential for now)
        subject_ids = torch.arange(self.num_subjects, dtype=torch.long)
        
        # Use stored frame coordinates if available, otherwise compute from actions
        num_frames = video_latents.shape[0]
        if stored_frame_coords is not None:
            # stored_frame_coords: [T, S, 2] in pixel space
            frame_coords = torch.tensor(stored_frame_coords, dtype=torch.float32)
            # Normalize to [0, 1]
            frame_coords = frame_coords / self.image_size
        else:
            # Fallback: compute from initial positions and actions
            frame_coords = self._compute_frame_coords(initial_positions, actions, num_frames)
        
        result = {
            "prompts": "",  # No text prompts
            "ode_latent": video_latents.unsqueeze(0),  # [1, T, C, H, W] for compatibility
            "position_latents": position_latents,  # [S, T, C, H, W] per-subject position maps
            "actions": actions,  # [S, num_actions]
            "initial_positions": initial_positions,  # [S, 2]
            "subject_ids": subject_ids,  # [S]
            "frame_coords": frame_coords,  # [T, S, 2]
        }
        
        return result
    
    def _compute_frame_coords(self, initial_positions, actions, num_frames):
        """
        Compute per-frame (x, y) coordinates for each subject.
        
        Args:
            initial_positions: [S, 2] initial (x, y) positions in pixels
            actions: [S, num_actions] action indices per subject
            num_frames: number of frames (latent frames, e.g., 9)
        
        Returns:
            frame_coords: [num_frames, S, 2] normalized coordinates in [0, 1]
        """
        ACTION_DELTAS = {
            0: (0, 0),      # still
            1: (0, -self.speed),     # up (y decreases)
            2: (0, self.speed),      # down (y increases)
            3: (-self.speed, 0),     # left (x decreases)
            4: (self.speed, 0)       # right (x increases)
        }
        
        S = initial_positions.shape[0]
        frame_coords = torch.zeros(num_frames, S, 2)
        
        # Frame 0: initial positions
        for s in range(S):
            frame_coords[0, s] = initial_positions[s]
        
        # Subsequent frames: apply actions
        for t in range(1, num_frames):
            for s in range(S):
                if t - 1 < actions.shape[1]:
                    action = actions[s, t - 1].item()
                else:
                    action = 0
                
                dx, dy = ACTION_DELTAS.get(action, (0, 0))
                prev_pos = frame_coords[t - 1, s]
                new_x = max(0, min(self.image_size - 1, prev_pos[0] + dx))
                new_y = max(0, min(self.image_size - 1, prev_pos[1] + dy))
                
                frame_coords[t, s, 0] = new_x
                frame_coords[t, s, 1] = new_y
        
        # Normalize to [0, 1]
        frame_coords = frame_coords / self.image_size
        
        return frame_coords


class PixelPositionLMDBDataset(Dataset):
    """
    Dataset for pixel-space position diffusion mode.
    
    Loads pre-encoded video latents and raw pixel-space position maps from LMDB.
    Position maps are at latent spatial resolution (e.g., 32x32) with one channel per subject.
    
    Dataset format:
    - video_latents_{i}_data: [T, 16, H, W] VAE-encoded video
    - position_pixels_{i}_data: [T, S, H, W] raw pixel position maps (one channel per subject)
    - actions_{i}_data: [S, num_actions] action sequence per subject
    - initial_positions_{i}_data: [S, 2] initial (x, y) positions
    
    Each sample contains:
        - video_latents: [T, C, H, W] video latents from VAE
        - position_pixels: [S, T, H, W] pixel-space position maps per subject
        - actions: [S, num_actions] action sequence per subject
        - initial_positions: [S, 2] initial (x, y) position per subject
    """
    
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        import pickle
        self.data_path = data_path
        self.max_pair = max_pair
        
        # Open LMDB shards
        self.envs = []
        self.index = []
        
        for fname in sorted(os.listdir(data_path)):
            if fname.endswith('.lmdb'):
                path = os.path.join(data_path, fname)
                env = lmdb.open(path,
                                readonly=True,
                                lock=False,
                                readahead=False,
                                meminit=False)
                self.envs.append(env)
        
        # Load metadata from first shard
        env = self.envs[0]
        with env.begin(write=False) as txn:
            # Get number of samples
            num_samples_bytes = txn.get(b'num_samples')
            if num_samples_bytes:
                self.num_samples_per_shard = [int(num_samples_bytes.decode())]
            else:
                # Fallback: count samples
                count = 0
                while txn.get(f'video_latents_{count}_data'.encode()) is not None:
                    count += 1
                self.num_samples_per_shard = [count]
            
            # Get shapes
            self.video_latents_shape = self._parse_shape(txn.get(b'video_latents_shape'))
            self.position_pixels_shape = self._parse_shape(txn.get(b'position_pixels_shape'))
            self.actions_shape = self._parse_shape(txn.get(b'actions_shape'))
            self.initial_positions_shape = self._parse_shape(txn.get(b'initial_positions_shape'))
            
            # Get num_subjects
            num_subj = txn.get(b'num_subjects')
            self.num_subjects = int(num_subj.decode()) if num_subj else 2
            
            # Check for pixel_position_mode flag
            pixel_mode = txn.get(b'pixel_position_mode')
            if pixel_mode is None or pixel_mode.decode() != 'true':
                print("WARNING: Dataset may not be in pixel position mode")
            
            # Get image_size and speed for coordinate computation
            image_size = txn.get(b'image_size')
            self.image_size = int(image_size.decode()) if image_size else 256
            speed = txn.get(b'speed')
            self.speed = int(speed.decode()) if speed else 16
        
        # Build index for all samples across shards
        for shard_id, env in enumerate(self.envs):
            with env.begin(write=False) as txn:
                num_samples_bytes = txn.get(b'num_samples')
                if num_samples_bytes:
                    num = int(num_samples_bytes.decode())
                else:
                    num = self.num_samples_per_shard[0] if shard_id == 0 else 0
                
                if shard_id < len(self.num_samples_per_shard):
                    self.num_samples_per_shard[shard_id] = num
                else:
                    self.num_samples_per_shard.append(num)
                
                for local_i in range(num):
                    self.index.append((shard_id, local_i))
        
        print(f"PixelPositionLMDBDataset: {len(self.index)} samples, "
              f"video_shape={self.video_latents_shape}, position_pixels_shape={self.position_pixels_shape}, "
              f"num_subjects={self.num_subjects}")
    
    def _parse_shape(self, shape_bytes):
        """Parse shape string like '(9, 2, 32, 32)' to tuple."""
        if shape_bytes is None:
            return None
        shape_str = shape_bytes.decode()
        if '(' in shape_str:
            shape_str = shape_str.strip('()')
            parts = [int(x.strip()) for x in shape_str.split(',') if x.strip()]
        else:
            parts = [int(x) for x in shape_str.split() if x.strip()]
        return tuple(parts)
    
    def __len__(self):
        return min(len(self.index), self.max_pair)
    
    def __getitem__(self, idx):
        import pickle
        shard_id, local_idx = self.index[idx]
        env = self.envs[shard_id]
        
        with env.begin(write=False) as txn:
            # Load video latents
            video_bytes = txn.get(f'video_latents_{local_idx}_data'.encode())
            video_latents = pickle.loads(video_bytes)  # [T, C, H, W]
            
            # Load position pixels
            position_bytes = txn.get(f'position_pixels_{local_idx}_data'.encode())
            position_pixels = pickle.loads(position_bytes)  # [T, S, H, W]
            
            # Load actions
            actions_bytes = txn.get(f'actions_{local_idx}_data'.encode())
            actions = pickle.loads(actions_bytes)  # [S, num_actions]
            
            # Load initial positions
            positions_bytes = txn.get(f'initial_positions_{local_idx}_data'.encode())
            initial_positions = pickle.loads(positions_bytes)  # [S, 2]
        
        # Convert to tensors
        video_latents = torch.tensor(video_latents, dtype=torch.float32)
        position_pixels = torch.tensor(position_pixels, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        initial_positions = torch.tensor(initial_positions, dtype=torch.float32)
        
        # position_pixels is [T, S, H, W], transpose to [S, T, H, W] for per-subject handling
        # This matches the convention used in position_latents
        position_pixels = position_pixels.permute(1, 0, 2, 3)  # [S, T, H, W]
        
        # Ensure initial_positions is [S, 2]
        if initial_positions.dim() == 1:
            initial_positions = initial_positions.unsqueeze(0)
        
        # Ensure actions is [S, num_actions]
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        
        # Create subject_ids
        subject_ids = torch.arange(self.num_subjects, dtype=torch.long)
        
        # Compute frame coordinates
        num_frames = video_latents.shape[0]
        frame_coords = self._compute_frame_coords(initial_positions, actions, num_frames)
        
        result = {
            "prompts": "",
            "ode_latent": video_latents.unsqueeze(0),  # [1, T, C, H, W]
            "position_pixels": position_pixels,  # [S, T, H, W] pixel-space position maps
            "actions": actions,  # [S, num_actions]
            "initial_positions": initial_positions,  # [S, 2]
            "subject_ids": subject_ids,  # [S]
            "frame_coords": frame_coords,  # [T, S, 2]
        }
        
        return result
    
    def _compute_frame_coords(self, initial_positions, actions, num_frames):
        """Compute per-frame (x, y) coordinates for each subject."""
        ACTION_DELTAS = {
            0: (0, 0),
            1: (0, -self.speed),
            2: (0, self.speed),
            3: (-self.speed, 0),
            4: (self.speed, 0)
        }
        
        S = initial_positions.shape[0]
        frame_coords = torch.zeros(num_frames, S, 2)
        
        for s in range(S):
            frame_coords[0, s] = initial_positions[s]
        
        for t in range(1, num_frames):
            for s in range(S):
                if t - 1 < actions.shape[1]:
                    action = actions[s, t - 1].item()
                else:
                    action = 0
                
                dx, dy = ACTION_DELTAS.get(action, (0, 0))
                prev_pos = frame_coords[t - 1, s]
                new_x = max(0, min(self.image_size - 1, prev_pos[0] + dx))
                new_y = max(0, min(self.image_size - 1, prev_pos[1] + dy))
                
                frame_coords[t, s, 0] = new_x
                frame_coords[t, s, 1] = new_y
        
        frame_coords = frame_coords / self.image_size
        return frame_coords


class MultiGameLMDBDataset(Dataset):
    """
    Dataset for multi-game Melting Pot data with coord_token_mode.
    
    Supports loading from a directory with game subdirectories, each containing
    a shard_0.lmdb file. Handles variable num_subjects and num_actions across
    games by padding to the maximum.
    
    Each sample contains:
        - video_latents: [T, C, H, W] video latents from VAE
        - actions: [S, num_actions] per-subject action sequences
        - frame_coords: [T, S, 2] per-frame pixel coordinates
        - orientations: [T, S] per-frame orientations
        - initial_positions: [S, 2] initial positions
    
    No position_latents are needed (coord_token_mode uses frame_coords directly).
    """
    
    def __init__(self, data_path: str, max_pair: int = int(1e8),
                 max_subjects: int = None, max_action_classes: int = None,
                 use_game_descriptions: bool = True):
        import pickle
        self.data_path = data_path
        self.max_pair = max_pair
        
        # Discover LMDB shards (supports both flat and nested directory structures)
        self.envs = []
        self.shard_metadata = []  # per-shard metadata (num_subjects, num_actions, game_name)
        self.index = []  # (shard_id, local_idx)
        
        shard_paths = []
        # Check for nested game directories: data_path/game_name/shard_0.lmdb
        for entry in sorted(os.listdir(data_path)):
            entry_path = os.path.join(data_path, entry)
            if os.path.isdir(entry_path):
                # Check for .lmdb inside
                for fname in sorted(os.listdir(entry_path)):
                    if fname.endswith('.lmdb'):
                        shard_paths.append(os.path.join(entry_path, fname))
            elif entry.endswith('.lmdb'):
                shard_paths.append(entry_path)
        
        if not shard_paths:
            raise ValueError(f"No .lmdb shards found in {data_path}")
        
        # Open all shards and collect metadata
        global_max_subjects = 0
        global_max_actions = 0
        
        for shard_path in shard_paths:
            env = lmdb.open(shard_path,
                           readonly=True,
                           lock=False,
                           readahead=False,
                           meminit=False)
            
            with env.begin(write=False) as txn:
                num_samples = int(txn.get(b'num_samples').decode())
                num_subjects = int(txn.get(b'num_subjects').decode())
                num_actions_raw = txn.get(b'num_actions')
                num_actions = int(num_actions_raw.decode()) if num_actions_raw else 7
                game_name_raw = txn.get(b'game_name')
                game_name = game_name_raw.decode() if game_name_raw else os.path.basename(os.path.dirname(shard_path))
                image_size_raw = txn.get(b'image_size')
                image_size = int(image_size_raw.decode()) if image_size_raw else 512
                
                # Get shapes
                video_shape_raw = txn.get(b'video_latents_shape')
                video_shape = self._parse_shape(video_shape_raw) if video_shape_raw else None
            
            shard_id = len(self.envs)
            self.envs.append(env)
            self.shard_metadata.append({
                'num_subjects': num_subjects,
                'num_actions': num_actions,
                'game_name': game_name,
                'image_size': image_size,
                'video_shape': video_shape,
                'num_samples': num_samples,
            })
            
            global_max_subjects = max(global_max_subjects, num_subjects)
            global_max_actions = max(global_max_actions, num_actions)
            
            for local_i in range(num_samples):
                self.index.append((shard_id, local_i))
        
        # Set padding targets
        self.max_subjects = max_subjects or global_max_subjects
        self.max_action_classes = max_action_classes or global_max_actions
        self.num_subjects = self.max_subjects  # for compatibility with trainer
        
        # Use first shard's video shape as reference
        self.video_latents_shape = self.shard_metadata[0]['video_shape']
        self.position_latents_shape = None  # No position latents in this dataset
        self.image_size = self.shard_metadata[0]['image_size']
        
        # Load game descriptions for text conditioning
        self.use_game_descriptions = use_game_descriptions
        if use_game_descriptions:
            try:
                from utils.game_descriptions import GAME_DESCRIPTIONS, get_game_description
                self.game_descriptions = GAME_DESCRIPTIONS
                self.get_game_description = get_game_description
            except ImportError:
                print("WARNING: utils.game_descriptions not importable, falling back to game names as prompts")
                self.use_game_descriptions = False
                self.game_descriptions = {}
                self.get_game_description = lambda name: name
        
        # Summary
        game_names = [m['game_name'] for m in self.shard_metadata]
        subjects_range = f"{min(m['num_subjects'] for m in self.shard_metadata)}-{global_max_subjects}"
        actions_range = f"{min(m['num_actions'] for m in self.shard_metadata)}-{global_max_actions}"
        print(f"MultiGameLMDBDataset: {len(self.index)} samples from {len(self.envs)} shards "
              f"({len(set(game_names))} games)")
        print(f"  Subjects: {subjects_range} (padded to {self.max_subjects})")
        print(f"  Actions: {actions_range} (padded to {self.max_action_classes})")
        print(f"  Video shape: {self.video_latents_shape}")
        print(f"  Image size: {self.image_size}")
        if self.use_game_descriptions:
            described = sum(1 for g in set(game_names) if g in self.game_descriptions)
            print(f"  Game descriptions: {described}/{len(set(game_names))} games have descriptions")
    
    def _parse_shape(self, shape_bytes):
        if shape_bytes is None:
            return None
        shape_str = shape_bytes.decode()
        if '(' in shape_str:
            shape_str = shape_str.strip('()')
            parts = [int(x.strip()) for x in shape_str.split(',') if x.strip()]
        else:
            parts = [int(x) for x in shape_str.split() if x.strip()]
        return tuple(parts)
    
    def __len__(self):
        return min(len(self.index), self.max_pair)
    
    def __getitem__(self, idx):
        import pickle
        shard_id, local_idx = self.index[idx]
        env = self.envs[shard_id]
        meta = self.shard_metadata[shard_id]
        
        game_num_subjects = meta['num_subjects']
        game_num_actions = meta['num_actions']
        
        with env.begin(write=False) as txn:
            # Load video latents: [T, C, H, W]
            video_bytes = txn.get(f'video_latents_{local_idx}_data'.encode())
            video_latents = pickle.loads(video_bytes)
            
            # Load actions: [S, num_actions_per_step]
            actions_bytes = txn.get(f'actions_{local_idx}_data'.encode())
            actions = pickle.loads(actions_bytes)
            
            # Load frame coords: [T, S, 2]
            frame_coords_bytes = txn.get(f'frame_coords_{local_idx}_data'.encode())
            frame_coords = pickle.loads(frame_coords_bytes)
            
            # Load orientations: [T, S]
            orientations_bytes = txn.get(f'orientations_{local_idx}_data'.encode())
            orientations = pickle.loads(orientations_bytes) if orientations_bytes else None
            
            # Load initial positions: [S, 2]
            positions_bytes = txn.get(f'initial_positions_{local_idx}_data'.encode())
            initial_positions = pickle.loads(positions_bytes)
        
        # Convert to tensors
        video_latents = torch.tensor(video_latents, dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.long)
        frame_coords = torch.tensor(frame_coords, dtype=torch.float32)
        initial_positions = torch.tensor(initial_positions, dtype=torch.float32)
        
        # Ensure correct dimensions
        # actions: [S, num_actions_per_step] where S is game-specific
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        if actions.dim() == 2 and actions.shape[0] != game_num_subjects:
            # Might be [num_steps, S] instead of [S, num_steps] - transpose
            if actions.shape[1] == game_num_subjects:
                actions = actions.T  # [S, num_steps]
        
        # initial_positions: [S, 2]
        if initial_positions.dim() == 1:
            initial_positions = initial_positions.unsqueeze(0)
        
        # Wrap OOB coordinates using modulo to handle torus-topology games.
        # Some games (e.g., territory__rooms, allelopathic_harvest) have ghost sprites
        # (zapped/frozen players) whose positions can drift past the map edge.
        # Since these games have wrapping topology, modulo gives the correct wrapped
        # position and preserves the action-to-coordinate relationship (e.g., moving
        # right at the right edge wraps to the left side).
        canvas = float(self.image_size)
        frame_coords = frame_coords % canvas
        initial_positions = initial_positions % canvas
        
        # Normalize frame_coords to [0, 1]
        frame_coords = frame_coords / self.image_size
        
        T = video_latents.shape[0]  # temporal frames (latent)
        S = self.max_subjects
        
        # Convert orientations to tensor if available
        if orientations is not None:
            orientations = torch.tensor(np.array(orientations), dtype=torch.long)
        
        # Pad subjects dimension to max_subjects
        if game_num_subjects < S:
            pad_s = S - game_num_subjects
            
            # Pad frame_coords: [T, S_game, 2] -> [T, S, 2]
            frame_coords_padded = torch.zeros(T, S, 2)
            frame_coords_padded[:, :game_num_subjects, :] = frame_coords
            frame_coords = frame_coords_padded
            
            # Pad actions: [S_game, num_steps] -> [S, num_steps]
            num_steps = actions.shape[1] if actions.dim() == 2 else actions.shape[0]
            actions_padded = torch.zeros(S, num_steps, dtype=torch.long)
            actions_padded[:game_num_subjects] = actions
            actions = actions_padded
            
            # Pad initial_positions: [S_game, 2] -> [S, 2]
            positions_padded = torch.zeros(S, 2)
            positions_padded[:game_num_subjects] = initial_positions
            initial_positions = positions_padded
            
            # Pad orientations: [T, S_game] -> [T, S]
            if orientations is not None:
                orientations_padded = torch.zeros(T, S, dtype=torch.long)
                orientations_padded[:, :game_num_subjects] = orientations
                orientations = orientations_padded
        
        # Create subject mask: 1 for real subjects, 0 for padding
        subject_mask = torch.zeros(S, dtype=torch.bool)
        subject_mask[:game_num_subjects] = True
        
        # Create subject_ids
        subject_ids = torch.arange(S, dtype=torch.long)
        
        # Game text prompt (description or name)
        game_name = meta['game_name']
        if self.use_game_descriptions:
            prompt = self.get_game_description(game_name)
        else:
            prompt = game_name
        
        result = {
            "prompts": prompt,  # Game description or name as text prompt
            "game_name": game_name,  # Raw game name string for per-game lookups
            "ode_latent": video_latents.unsqueeze(0),  # [1, T, C, H, W]
            "actions": actions,  # [S, num_steps]
            "initial_positions": initial_positions,  # [S, 2]
            "subject_ids": subject_ids,  # [S]
            "frame_coords": frame_coords,  # [T, S, 2]
            "subject_mask": subject_mask,  # [S] boolean mask
            "num_real_subjects": torch.tensor(game_num_subjects, dtype=torch.long),
        }
        if orientations is not None:
            result["orientations"] = orientations  # [T, S]
        
        return result


def cycle(dl):
    while True:
        for data in dl:
            yield data
