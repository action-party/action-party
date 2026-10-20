"""
Attention logging utility for monitoring self-attention and cross-attention 
output norms during training.
"""

import torch
import torch.nn as nn
from collections import defaultdict
from typing import Dict, List, Optional


class AttentionLogger:
    """
    Hooks into WanAttentionBlock layers to log self-attention and cross-attention norms.
    
    Usage:
        logger = AttentionLogger(model.blocks)
        # ... run forward pass ...
        log_dict = logger.get_log_dict()
        logger.clear()  # Clear for next iteration
    """
    
    def __init__(self, blocks: nn.ModuleList):
        """
        Args:
            blocks: The nn.ModuleList containing WanAttentionBlock modules
        """
        self.hooks = []
        self.self_attn_norms = {}
        self.cross_attn_norms = {}
        self._registered = False
        self.blocks = blocks
        
    def register_hooks(self):
        """Register forward hooks on all attention layers."""
        if self._registered:
            return
            
        for i, block in enumerate(self.blocks):
            # Hook for self-attention
            if hasattr(block, 'self_attn') and block.self_attn is not None:
                hook = block.self_attn.register_forward_hook(
                    self._make_hook(f'block_{i}', 'self_attn')
                )
                self.hooks.append(hook)
            
            # Hook for cross-attention
            if hasattr(block, 'cross_attn') and block.cross_attn is not None:
                hook = block.cross_attn.register_forward_hook(
                    self._make_hook(f'block_{i}', 'cross_attn')
                )
                self.hooks.append(hook)
        
        self._registered = True
    
    def _make_hook(self, block_name: str, attn_type: str):
        """Create a forward hook that stores output norm."""
        def hook(module, input, output):
            # Handle tuple outputs (some attention layers return tuples)
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            
            # Compute L2 norm of output
            with torch.no_grad():
                norm = out.detach().float().norm().item()
            
            if attn_type == 'self_attn':
                self.self_attn_norms[block_name] = norm
            else:
                self.cross_attn_norms[block_name] = norm
        
        return hook
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self._registered = False
    
    def clear(self):
        """Clear stored norms for next iteration."""
        self.self_attn_norms = {}
        self.cross_attn_norms = {}
    
    def get_log_dict(self, prefix: str = "attn") -> Dict[str, float]:
        """
        Get a dictionary of logged norms suitable for wandb logging.
        
        Returns dict with keys like:
            - attn/self_attn_mean_norm
            - attn/cross_attn_mean_norm
            - attn/cross_self_ratio
            - attn/block_0_self_attn_norm (if detailed=True in future)
        """
        log_dict = {}
        
        if self.self_attn_norms:
            self_norms = list(self.self_attn_norms.values())
            log_dict[f"{prefix}/self_attn_mean_norm"] = sum(self_norms) / len(self_norms)
            log_dict[f"{prefix}/self_attn_max_norm"] = max(self_norms)
            log_dict[f"{prefix}/self_attn_min_norm"] = min(self_norms)
        
        if self.cross_attn_norms:
            cross_norms = list(self.cross_attn_norms.values())
            log_dict[f"{prefix}/cross_attn_mean_norm"] = sum(cross_norms) / len(cross_norms)
            log_dict[f"{prefix}/cross_attn_max_norm"] = max(cross_norms)
            log_dict[f"{prefix}/cross_attn_min_norm"] = min(cross_norms)
        
        # Ratio of cross to self attention (indicates relative strength of conditioning)
        if self.self_attn_norms and self.cross_attn_norms:
            self_mean = sum(self.self_attn_norms.values()) / len(self.self_attn_norms)
            cross_mean = sum(self.cross_attn_norms.values()) / len(self.cross_attn_norms)
            log_dict[f"{prefix}/cross_self_ratio"] = cross_mean / (self_mean + 1e-8)
        
        return log_dict
    
    def get_detailed_log_dict(self, prefix: str = "attn_blocks") -> Dict[str, float]:
        """
        Get per-block norms for detailed analysis.
        Only use periodically as this creates many log entries.
        """
        log_dict = {}
        
        for block_name, norm in self.self_attn_norms.items():
            log_dict[f"{prefix}/{block_name}_self_attn"] = norm
        
        for block_name, norm in self.cross_attn_norms.items():
            log_dict[f"{prefix}/{block_name}_cross_attn"] = norm
        
        return log_dict


class AttentionDifferenceLogger:
    """
    Compare attention outputs between different conditioning inputs.
    Useful for measuring how much the model is using the conditioning.
    """
    
    def __init__(self, blocks: nn.ModuleList):
        self.blocks = blocks
        self.baseline_outputs = {}
        self.comparison_outputs = {}
        self.hooks = []
        self._registered = False
        self._recording_baseline = True
    
    def register_hooks(self):
        if self._registered:
            return
            
        for i, block in enumerate(self.blocks):
            if hasattr(block, 'cross_attn') and block.cross_attn is not None:
                hook = block.cross_attn.register_forward_hook(
                    self._make_hook(f'block_{i}')
                )
                self.hooks.append(hook)
        
        self._registered = True
    
    def _make_hook(self, block_name: str):
        def hook(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            
            with torch.no_grad():
                if self._recording_baseline:
                    self.baseline_outputs[block_name] = out.detach().clone()
                else:
                    self.comparison_outputs[block_name] = out.detach().clone()
        
        return hook
    
    def set_recording_baseline(self, is_baseline: bool):
        """Switch between recording baseline or comparison outputs."""
        self._recording_baseline = is_baseline
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self._registered = False
    
    def clear(self):
        self.baseline_outputs = {}
        self.comparison_outputs = {}
        self._recording_baseline = True
    
    def get_difference_metrics(self) -> Dict[str, float]:
        """
        Compute difference metrics between baseline and comparison outputs.
        Returns relative difference per block and mean.
        """
        if not self.baseline_outputs or not self.comparison_outputs:
            return {}
        
        differences = []
        for block_name in self.baseline_outputs:
            if block_name in self.comparison_outputs:
                baseline = self.baseline_outputs[block_name].float()
                comparison = self.comparison_outputs[block_name].float()
                
                diff = (baseline - comparison).abs().mean().item()
                norm = baseline.abs().mean().item()
                rel_diff = diff / (norm + 1e-8) * 100  # Percentage
                differences.append(rel_diff)
        
        if differences:
            return {
                "cross_attn_diff/mean_pct": sum(differences) / len(differences),
                "cross_attn_diff/max_pct": max(differences),
                "cross_attn_diff/min_pct": min(differences),
            }
        return {}

