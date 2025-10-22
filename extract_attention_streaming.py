#!/usr/bin/env python3
"""
StreamingVLM Attention Extraction with Streaming Inference

Extracts and visualizes multimodal attention maps from StreamingVLM's first decoder layer
while using the full streaming inference pipeline.

Usage:
    python extract_attention_streaming.py \
        --video_path videos/short_clip.mp4 \
        --question "What is happening in this video?" \
        --output_dir outputs/attn_streaming
        
    # Optional: Capture decode steps
    python extract_attention_streaming.py \
        --video_path videos/short_clip.mp4 \
        --question "What is happening?" \
        --capture_decode \
        --max_decode_steps 4
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
import math
from typing import Dict, List, Tuple, Optional

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# StreamingVLM imports
from streaming_vlm.inference.qwen2_5.patch_model import convert_qwen2_5_to_streaming
from streaming_vlm.inference.streaming_args import StreamingArgs

# Video processing
try:
    from qwen_vl_utils import process_vision_info
    HAS_QWEN_VL_UTILS = True
except ImportError:
    HAS_QWEN_VL_UTILS = False
    print("Warning: qwen_vl_utils not found. Install with: pip install qwen-vl-utils")

# Optional visualization dependencies
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


# ============================================================================
# Attention Capture for Streaming
# ============================================================================

class StreamingAttentionCapture:
    """
    Captures attention from streaming inference, focusing on prefill phase.

    Attributes:
        prefill_attention: Attention from prefill phase [B, H, T, T]
        decode_attentions: List of attention tensors from first N decode steps
        capture_decode: Whether to capture decode-step attention
        max_decode_steps: How many decode steps to capture
    """

    def __init__(self, capture_decode: bool = False, max_decode_steps: int = 4):
        self.prefill_attention = None
        self.decode_attentions = []
        self.capture_decode = capture_decode
        self.max_decode_steps = max_decode_steps
        self.hook_handles = []
        self.is_prefill = True
        self.decode_count = 0

    def register_hook(self, model, layer_idx: int = 0):
        """Register forward hook on specified layer's self-attention."""

        def attention_hook(module, input, output):
            """Capture attention weights during forward pass."""
            # Output format: (attn_output, attn_weights, past_key_value)
            if isinstance(output, tuple) and len(output) >= 2:
                attn_output, attn_weights = output[0], output[1]

                if attn_weights is not None:
                    # Prefill phase: large q_len (many tokens processed at once)
                    # Decode phase: q_len = 1 (one new token)
                    bsz, num_heads, q_len, k_len = attn_weights.shape

                    if self.is_prefill and q_len > 1:
                        # This is the prefill phase - capture it (keep on GPU)
                        self.prefill_attention = attn_weights.detach()
                        self.is_prefill = False  # Transition to decode
                        print(f"[Prefill] Captured attention shape: {attn_weights.shape}")

                    elif self.capture_decode and self.decode_count < self.max_decode_steps:
                        # Capture first N decode steps (keep on GPU)
                        self.decode_attentions.append(attn_weights.detach())
                        self.decode_count += 1
                        print(f"[Decode {self.decode_count}] Captured attention shape: {attn_weights.shape}")

        # Hook into the language model's first layer
        target_module = model.model.language_model.layers[layer_idx].self_attn
        handle = target_module.register_forward_hook(attention_hook)
        self.hook_handles.append(handle)

        print(f"Registered hook on model.model.language_model.layers[{layer_idx}].self_attn")

    def clear_hooks(self):
        """Remove all registered hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []


# ============================================================================
# Token Index Mapper
# ============================================================================

class TokenIndexMapper:
    """Maps token indices to vision/text categories and per-frame boundaries."""

    def __init__(self, input_ids, video_token_id, video_grid_thw):
        """
        Args:
            input_ids: [batch, seq_len] tensor
            video_token_id: ID for video tokens (typically 151656)
            video_grid_thw: [num_videos, 3] tensor with (t, h, w) per video
        """
        self.input_ids = input_ids
        self.video_token_id = video_token_id
        self.video_grid_thw = video_grid_thw
        self.device = input_ids.device  # Store device for later use

        # Find video token positions (keep on same device as input_ids)
        video_mask = (input_ids[0] == video_token_id)
        self.vision_idx = torch.where(video_mask)[0].to(self.device)

        # Get video structure (pre-merger grid dimensions)
        if len(video_grid_thw.shape) == 2:
            t, h, w = video_grid_thw[0]
        else:
            t, h, w = video_grid_thw

        self.num_frames = int(t.item())
        self.pre_merger_grid = (int(h.item()), int(w.item()))
        pre_merger_patches = int(h.item()) * int(w.item())

        # Print diagnostics: pre-merger vs post-merger
        expected_tokens = self.num_frames * pre_merger_patches
        actual_tokens = len(self.vision_idx)
        print(f"   Pre-merger grid: {self.num_frames} frames × {self.pre_merger_grid} = {expected_tokens} patches")
        print(f"   Post-merger tokens: {actual_tokens} vision tokens in input_ids")

        if actual_tokens != expected_tokens:
            print(f"   ⚠️  Merger applied: {expected_tokens} → {actual_tokens} tokens")
            # Use actual post-merger token count
            self.patches_per_frame = actual_tokens // self.num_frames

            # Infer merger ratio and post-merger grid
            # Common mergers: 2x2 (4:1), 3x3 (9:1), etc.
            pre_h, pre_w = self.pre_merger_grid
            merger_ratio = expected_tokens / actual_tokens if actual_tokens > 0 else 1

            # Calculate post-merger grid dimensions
            # For 2x2 merger: h/2, w/2
            import math
            merger_side = int(round(math.sqrt(merger_ratio)))
            post_h = pre_h // merger_side
            post_w = pre_w // merger_side

            # Verify this produces correct token count
            if post_h * post_w == self.patches_per_frame:
                self.spatial_grid = (post_h, post_w)
                print(f"   Using post-merger: {self.patches_per_frame} patches/frame, "
                      f"spatial grid = {self.spatial_grid} (merger: {merger_side}x{merger_side})")
            else:
                # Fallback: find best factorization
                best_h, best_w = 1, self.patches_per_frame
                for h in range(1, int(math.sqrt(self.patches_per_frame)) + 1):
                    if self.patches_per_frame % h == 0:
                        w = self.patches_per_frame // h
                        if abs(h - w) < abs(best_h - best_w):
                            best_h, best_w = h, w
                self.spatial_grid = (best_h, best_w)
                print(f"   Using post-merger: {self.patches_per_frame} patches/frame, "
                      f"spatial grid ≈ {self.spatial_grid} (estimated)")
        else:
            # No merger applied
            self.patches_per_frame = pre_merger_patches
            self.spatial_grid = self.pre_merger_grid

    def get_frame_indices(self, frame_idx: int):
        """Get vision token positions for a specific frame (returns tensor on same device)."""
        if frame_idx >= self.num_frames:
            raise ValueError(f"Frame {frame_idx} out of range (max: {self.num_frames-1})")

        start_offset = frame_idx * self.patches_per_frame
        end_offset = (frame_idx + 1) * self.patches_per_frame

        # Return slice of vision_idx (already on correct device)
        return self.vision_idx[start_offset:end_offset]

    def get_question_indices(self, tokenizer):
        """Get user question token positions, excluding control tokens (returns tensor on same device as input_ids)."""
        ids = self.input_ids[0]
        dev = ids.device

        # Get special token IDs
        im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        user_id = tokenizer.convert_tokens_to_ids("user")
        assistant_id = tokenizer.convert_tokens_to_ids("assistant")
        nl_id = tokenizer("\n", add_special_tokens=False).input_ids[0]

        # Find the user segment: <|im_start|> user \n <question tokens> <|im_end|>
        starts = (ids == im_start_id).nonzero(as_tuple=True)[0]
        ends = (ids == im_end_id).nonzero(as_tuple=True)[0]

        text_start = None
        text_end = None
        for s, e in zip(starts.tolist(), ends.tolist()):
            # expect: <|im_start|> user \n  <QUESTION ...>  <|im_end|>
            if s + 1 < len(ids) and ids[s + 1].item() == user_id and e > s:
                t0 = s + 2
                # optional newline right after 'user'
                if t0 < len(ids) and ids[t0].item() == nl_id:
                    t0 += 1
                text_start, text_end = t0, e
                break

        if text_start is None:
            # Fallback: everything after last vision token to end
            text_start = (self.vision_idx[-1].item() + 1) if len(self.vision_idx) > 0 else 0
            text_end = self.input_ids.shape[1]

        # Drop any lingering control tokens inside the span
        keep = []
        for i in range(text_start, text_end):
            tok = ids[i].item()
            if tok in (im_start_id, im_end_id, user_id, assistant_id, nl_id):
                continue
            keep.append(i)

        return torch.tensor(keep, device=dev, dtype=torch.long)


# ============================================================================
# Attention Visualizer
# ============================================================================

class AttentionVisualizer:
    """
    Visualizes attention patterns as heatmaps and videos.

    Usage:
        viz = AttentionVisualizer(output_dir="outputs/attn")
        viz.plot_frame_to_text_attention(attn, frame_idx=0, ...)
        viz.plot_all_frames_summary(frame_attentions, ...)
    """

    def __init__(self, output_dir: str = "outputs/attn_maps"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_attention_matrix(
        self,
        attention: np.ndarray,
        title: str,
        x_label: str,
        y_label: str,
        y_tokens: List[str] = None,
        x_tokens: List[str] = None,
        spatial_grid: Tuple[int, int] = None,
        spatial_axis: int = None,
        save_path: str = None
    ):
        """
        Generic attention matrix plotter.

        Args:
            attention: [rows, cols] attention array
            title: Plot title
            x_label: X-axis label
            y_label: Y-axis label
            y_tokens: Optional token labels for y-axis
            x_tokens: Optional token labels for x-axis
            spatial_grid: Optional (h, w) for spatial visualization
            spatial_axis: Which axis to average (0=rows, 1=cols) for spatial plot
            save_path: Where to save
        """
        if spatial_grid is not None and spatial_axis is not None:
            # Two subplots: matrix + spatial
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        else:
            # Single plot: just the matrix
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

        # Main attention matrix
        if HAS_SEABORN:
            sns.heatmap(attention, cmap='hot', ax=ax1, cbar=True)
        else:
            im1 = ax1.imshow(attention, cmap='hot', aspect='auto')
            plt.colorbar(im1, ax=ax1)

        ax1.set_xlabel(x_label)
        ax1.set_ylabel(y_label)
        ax1.set_title(title)

        # Set token labels if provided and not too many
        if y_tokens and len(y_tokens) <= 20:
            ax1.set_yticks(range(len(y_tokens)))
            ax1.set_yticklabels(y_tokens, rotation=0)
        if x_tokens and len(x_tokens) <= 20:
            ax1.set_xticks(range(len(x_tokens)))
            ax1.set_xticklabels(x_tokens, rotation=45, ha='right')

        # Spatial subplot if requested
        if spatial_grid is not None and spatial_axis is not None:
            h, w = spatial_grid
            spatial_attn = attention.mean(axis=spatial_axis).reshape(h, w)

            if HAS_SEABORN:
                sns.heatmap(spatial_attn, cmap='hot', ax=ax2, cbar=True, square=True)
            else:
                im2 = ax2.imshow(spatial_attn, cmap='hot')
                plt.colorbar(im2, ax=ax2)

            ax2.set_xlabel('Patch X')
            ax2.set_ylabel('Patch Y')
            avg_label = 'text tokens' if spatial_axis == 0 else 'vision patches'
            ax2.set_title(f'Spatial Attention (Avg over {avg_label})')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / "attention_matrix.png", dpi=150, bbox_inches='tight')

        plt.close()

    def plot_frame_to_text_attention(
        self,
        attention: np.ndarray,
        frame_idx: int,
        spatial_grid: Tuple[int, int],
        text_tokens: List[str],
        save_path: str = None
    ):
        """
        Plot attention from text tokens to a single frame's vision patches.

        Args:
            attention: [num_text_tokens, num_patches] array (TEXT→VISION)
            frame_idx: Frame index for labeling
            spatial_grid: (height, width) of patch grid
            text_tokens: List of text token strings
            save_path: Optional custom save path
        """
        h, w = spatial_grid

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Full attention matrix (rows=text, cols=vision)
        if HAS_SEABORN:
            sns.heatmap(attention, cmap='hot', ax=ax1, cbar=True)
        else:
            im1 = ax1.imshow(attention, cmap='hot', aspect='auto')
            plt.colorbar(im1, ax=ax1)

        ax1.set_xlabel('Vision Patch Index')
        ax1.set_ylabel('Text Token Index')
        ax1.set_title(f'Frame {frame_idx}: Text→Vision Attention')

        # Set text token labels if not too many
        if len(text_tokens) <= 20:
            ax1.set_yticks(range(len(text_tokens)))
            ax1.set_yticklabels(text_tokens, rotation=0)

        # Right: Spatial attention (averaged over all text tokens)
        spatial_attn = attention.mean(axis=0).reshape(h, w)

        if HAS_SEABORN:
            sns.heatmap(spatial_attn, cmap='hot', ax=ax2, cbar=True, square=True)
        else:
            im2 = ax2.imshow(spatial_attn, cmap='hot')
            plt.colorbar(im2, ax=ax2)

        ax2.set_xlabel('Patch X')
        ax2.set_ylabel('Patch Y')
        ax2.set_title(f'Frame {frame_idx}: Spatial Attention (Avg over Text)')

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / f"frame_{frame_idx:03d}_attention.png"

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"   Saved: {save_path}")

    def plot_all_frames_summary(
        self,
        frame_attentions: List[np.ndarray],
        spatial_grid: Tuple[int, int],
        save_path: str = None
    ):
        """
        Create a summary grid showing spatial attention for all frames.

        Args:
            frame_attentions: List of [num_text, num_patches] arrays (TEXT→VISION)
            spatial_grid: (height, width) of patch grid
            save_path: Optional custom save path
        """
        h, w = spatial_grid
        num_frames = len(frame_attentions)

        # Determine grid layout
        cols = min(4, num_frames)
        rows = (num_frames + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))

        if num_frames == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i, attn in enumerate(frame_attentions):
            # Average attention over text tokens (axis=0 for TEXT→VISION)
            spatial_attn = attn.mean(axis=0).reshape(h, w)

            if HAS_SEABORN:
                sns.heatmap(spatial_attn, cmap='hot', ax=axes[i], cbar=True, square=True)
            else:
                im = axes[i].imshow(spatial_attn, cmap='hot')
                plt.colorbar(im, ax=axes[i])

            axes[i].set_title(f'Frame {i}')
            axes[i].set_xlabel('Patch X')
            axes[i].set_ylabel('Patch Y')

        # Hide unused subplots
        for i in range(num_frames, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / "all_frames_summary.png"

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"   ✓ Saved: all_frames_summary.png")

    def create_attention_video(
        self,
        frame_attentions: List[np.ndarray],
        spatial_grid: Tuple[int, int],
        fps: int = 2,
        save_path: str = None
    ):
        """
        Create a video showing attention evolution across frames.

        Args:
            frame_attentions: List of [num_text, num_patches] arrays (TEXT→VISION)
            spatial_grid: (height, width) of patch grid
            fps: Frames per second
            save_path: Optional custom save path
        """
        if not HAS_OPENCV:
            print("   ⊘ Skipping video creation (opencv not installed)")
            return

        h, w = spatial_grid

        if save_path is None:
            save_path = self.output_dir / "attention_evolution.mp4"

        # Create temporary frames
        temp_frames = []
        for i, attn in enumerate(frame_attentions):
            spatial_attn = attn.mean(axis=0).reshape(h, w)

            # Create image
            fig, ax = plt.subplots(figsize=(6, 6))
            im = ax.imshow(spatial_attn, cmap='hot')
            ax.set_title(f'Frame {i}')
            plt.colorbar(im, ax=ax)
            plt.tight_layout()

            # Save to temp file
            temp_path = self.output_dir / f"_temp_frame_{i:03d}.png"
            plt.savefig(temp_path, dpi=100)
            plt.close()

            temp_frames.append(temp_path)

        # Create video
        if len(temp_frames) > 0:
            first_frame = cv2.imread(str(temp_frames[0]))
            height, width, _ = first_frame.shape

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(str(save_path), fourcc, fps, (width, height))

            for frame_path in temp_frames:
                frame = cv2.imread(str(frame_path))
                video.write(frame)

            video.release()
            print(f"   ✓ Saved: attention_evolution.mp4")

            # Cleanup temp frames
            for frame_path in temp_frames:
                frame_path.unlink()


# ============================================================================
# Main Extraction Function
# ============================================================================

def extract_attention_streaming(
    video_path: str,
    question: str,
    model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    output_dir: str = "outputs/attn_streaming",
    max_new_tokens: int = 20,
    capture_decode: bool = False,
    max_decode_steps: int = 4,
    fps: float = 1.0,
    max_pixels: int = 200 * 200,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda:0"
) -> Optional[Dict]:
    """
    Extract attention using StreamingVLM's streaming inference pipeline.

    Args:
        video_path: Path to video file
        question: Question about the video
        model_path: HuggingFace model path
        output_dir: Where to save outputs
        max_new_tokens: Max tokens to generate
        capture_decode: Whether to capture first N decode steps
        max_decode_steps: How many decode steps to capture
        dtype: Model precision
        device: Device placement
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("STREAMING ATTENTION EXTRACTION")
    print("="*80)

    # ========================================================================
    # 1. Load Model with Eager Attention
    # ========================================================================

    print(f"\n[1/7] Loading model: {model_path}")

    # Handle device_map correctly: only use "auto" for multi-GPU, otherwise manual placement
    if device == "auto":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
            attn_implementation="eager",
            trust_remote_code=True
        )
        print("       Device: auto (multi-GPU/offload)")
    else:
        # Single GPU: load without device_map, then move manually
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            attn_implementation="eager",
            trust_remote_code=True
        )
        model.to(device)
        print(f"       Device: {device} (single GPU)")

    # Apply StreamingVLM patches
    print("[2/7] Applying StreamingVLM patches...")
    model = convert_qwen2_5_to_streaming(model)
    model.eval()

    # Load processor
    print("[3/7] Loading processor...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    # ========================================================================
    # 2. Register Attention Hook
    # ========================================================================

    print(f"[4/7] Registering attention hook (capture_decode={capture_decode})...")
    capture = StreamingAttentionCapture(
        capture_decode=capture_decode,
        max_decode_steps=max_decode_steps
    )
    capture.register_hook(model, layer_idx=0)

    # ========================================================================
    # 3. Prepare Input
    # ========================================================================

    print(f"[5/7] Processing video: {video_path}")
    print(f"       Question: {question}")

    messages = [
        {"role": "user", "content": [
            {"type": "video", "video": video_path},
            {"type": "text", "text": question}
        ]}
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Process video
    if HAS_QWEN_VL_UTILS:
        vision_info = process_vision_info(messages)

        # Handle different return formats
        if isinstance(vision_info, tuple) and len(vision_info) == 2:
            image_inputs, video_inputs = vision_info
        elif isinstance(vision_info, tuple) and len(vision_info) == 3:
            image_inputs, video_inputs, _ = vision_info
        else:
            image_inputs, video_inputs = vision_info, None

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            fps=fps,
            max_pixels=max_pixels
        )

        # Only move inputs to device for single-GPU (not "auto")
        if device != "auto":
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        print(f"       Video sampling: fps={fps}, max_pixels={max_pixels}")
    else:
        raise ImportError("qwen_vl_utils required. Install with: pip install qwen-vl-utils")

    # Safer video_grid_thw access with fallback
    grid = inputs.get("video_grid_thw", inputs.get("image_grid_thw", None))
    assert grid is not None, "Missing video/image grid tensor in processor outputs"

    print(f"       Input shape: {inputs['input_ids'].shape}")
    print(f"       Video grid THW: {grid}")

    # ========================================================================
    # 4. Run Streaming Generation
    # ========================================================================

    print("[6/7] Generating with StreamingVLM...")

    # Configure streaming args
    # Note: sink_size and window_size are controlled elsewhere in the streaming pipeline
    # StreamingArgs only controls position mode and text retention
    streaming_args = StreamingArgs(
        pos_mode="append",      # Contiguous RoPE (grows indefinitely)
        all_text=True           # Keep all text tokens in cache (including question)
    )

    with torch.inference_mode():  # Slightly faster than no_grad
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            streaming_args=streaming_args,
            output_attentions=True,  # Force eager mode
            use_cache=True,
            do_sample=False
        )

    # Decode output
    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    print(f"\n       Generated: {generated_text}")

    # ========================================================================
    # 5. Extract Attention
    # ========================================================================

    print("[7/7] Extracting and visualizing attention...")

    if capture.prefill_attention is None:
        print("\n❌ ERROR: No prefill attention captured!")
        capture.clear_hooks()
        return None

    attn_probs = capture.prefill_attention
    print(f"\n✓ Prefill attention shape: {attn_probs.shape}")

    # Map token indices (use the safer grid variable)
    mapper = TokenIndexMapper(
        inputs['input_ids'],
        model.config.video_token_id,
        grid
    )

    # Get pre-merger grid dimensions for upsampling
    pre_h, pre_w = mapper.pre_merger_grid
    pre_patches = pre_h * pre_w

    print(f"✓ Video structure: {mapper.num_frames} frames")
    print(f"   Grid (pre-merger): {mapper.pre_merger_grid} → {pre_patches} patches/frame")
    print(f"   Vision tokens (post-merger): {len(mapper.vision_idx)} (spatial grid ≈ {mapper.spatial_grid})")

    # Average over heads, keep on GPU
    attn_mean = attn_probs.mean(dim=1).squeeze(0).contiguous()  # [seq_len, seq_len]

    # Ensure attn_mean is on correct device for GPU-safe indexing
    if device != "auto":
        attn_mean = attn_mean.to(device)

    print(f"✓ Attention mean shape: {attn_mean.shape}, device: {attn_mean.device}")

    # Get question indices (on same device)
    text_idx = mapper.get_question_indices(processor.tokenizer)
    question_tokens = [
        processor.tokenizer.decode([inputs['input_ids'][0, i].item()])
        for i in text_idx.cpu()  # Only move to CPU for tokenizer decode
    ]
    print(f"✓ Question tokens kept: {len(text_idx)}")
    print(f"   Tokens: {question_tokens}")
    print(f"   Text position range: [{text_idx[0].item()}, {text_idx[-1].item()}]")

    # Extract per-frame attention in BOTH directions (all operations on GPU)
    frame_attentions_tv = []  # TEXT→VISION
    frame_attentions_vt = []  # VISION→TEXT (true causal-masked)
    frame_attentions_vt_transpose = []  # Transpose view for comparison

    print(f"\n📊 Extracting per-frame attention in both directions (upsampling to pre-merger grid):")

    for frame_idx in range(mapper.num_frames):
        # Get frame vision indices (already on GPU)
        frame_vision_idx = mapper.get_frame_indices(frame_idx)

        # ==================================================================
        # A) TEXT→VISION (rows=text, cols=vision)
        # ==================================================================
        frame_attn_tv = torch.index_select(attn_mean, 0, text_idx)
        frame_attn_tv = torch.index_select(frame_attn_tv, 1, frame_vision_idx)
        frame_attn_tv = torch.nn.functional.normalize(frame_attn_tv, p=1, dim=1)

        # ==================================================================
        # B) VISION→TEXT (rows=vision, cols=text) - TRUE MASKED ATTENTION
        # ==================================================================
        frame_attn_vt = torch.index_select(attn_mean, 0, frame_vision_idx)
        frame_attn_vt = torch.index_select(frame_attn_vt, 1, text_idx)
        frame_attn_vt = torch.nn.functional.normalize(frame_attn_vt, p=1, dim=1)

        # --- Upsample TEXT→VISION to pre-merger patch grid ---
        nv_merged = mapper.patches_per_frame
        if nv_merged != pre_patches:
            # Infer merger side (e.g., 2 for 2x2 merger)
            ratio = pre_patches // nv_merged
            merger_side = int(round(math.sqrt(ratio)))
            assert merger_side * merger_side * nv_merged == pre_patches, \
                f"Can't reconcile merger: nv={nv_merged}, pre={pre_patches}, ratio={ratio}"

            # Upsample TEXT→VISION: [Nt, Nv] -> [Nt, pre_patches]
            post_h, post_w = mapper.spatial_grid
            attn_hw = frame_attn_tv.view(frame_attn_tv.size(0), post_h, post_w)
            attn_hw = torch.repeat_interleave(attn_hw, repeats=merger_side, dim=1)  # H
            attn_hw = torch.repeat_interleave(attn_hw, repeats=merger_side, dim=2)  # W
            frame_attn_tv = attn_hw.view(frame_attn_tv.size(0), pre_patches)

            # Upsample VISION→TEXT: [Nv, Nt] -> [pre_patches, Nt]
            # Need to reshape [Nv, Nt] -> [post_h, post_w, Nt]
            attn_hw_vt = frame_attn_vt.view(post_h, post_w, frame_attn_vt.size(1))
            attn_hw_vt = torch.repeat_interleave(attn_hw_vt, repeats=merger_side, dim=0)  # H
            attn_hw_vt = torch.repeat_interleave(attn_hw_vt, repeats=merger_side, dim=1)  # W
            frame_attn_vt = attn_hw_vt.view(pre_patches, frame_attn_vt.size(1))

        # Convert to numpy for storage/visualization
        attn_tv_np = frame_attn_tv.detach().float().cpu().numpy()
        attn_vt_np = frame_attn_vt.detach().float().cpu().numpy()
        attn_vt_transpose_np = attn_tv_np.T  # [pre_patches, Nt] transpose view

        frame_attentions_tv.append(attn_tv_np)
        frame_attentions_vt.append(attn_vt_np)
        frame_attentions_vt_transpose.append(attn_vt_transpose_np)

        # Compute stats
        print(f"   Frame {frame_idx:2d}:")
        print(f"      TEXT→VISION: {attn_tv_np.shape} (upsampled to {pre_h}×{pre_w}), "
              f"mean={attn_tv_np.mean():.6f}, max={attn_tv_np.max():.6f}")
        print(f"      VISION→TEXT (masked): {attn_vt_np.shape}, "
              f"mean={attn_vt_np.mean():.6f}, max={attn_vt_np.max():.6f}")

    # ========================================================================
    # 6. Save Raw Tensors
    # ========================================================================

    print(f"\n💾 Saving raw tensors...")

    # Save prefill attention (full tensor)
    torch.save(attn_probs, output_dir / "prefill_attention.pt")
    print(f"   ✓ Saved: prefill_attention.pt")

    # Save per-frame attention matrices in all three formats
    for i in range(mapper.num_frames):
        np.save(output_dir / f"frame_{i:03d}_text_to_vision.npy", frame_attentions_tv[i])
        np.save(output_dir / f"frame_{i:03d}_vision_to_text_masked.npy", frame_attentions_vt[i])
        np.save(output_dir / f"frame_{i:03d}_vision_to_text_transpose.npy", frame_attentions_vt_transpose[i])

    print(f"   ✓ Saved: {mapper.num_frames} × 3 attention matrices per frame")
    print(f"      - *_text_to_vision.npy: [{len(text_idx)}, {pre_patches}] TEXT→VISION")
    print(f"      - *_vision_to_text_masked.npy: [{pre_patches}, {len(text_idx)}] VISION→TEXT (causal masked)")
    print(f"      - *_vision_to_text_transpose.npy: [{pre_patches}, {len(text_idx)}] Transpose view")

    # Save decode attentions if captured
    if capture_decode and len(capture.decode_attentions) > 0:
        for i, attn in enumerate(capture.decode_attentions):
            torch.save(attn, output_dir / f"decode_step_{i:02d}_attention.pt")
        print(f"   ✓ Saved: {len(capture.decode_attentions)} decode-step tensors")

    # ========================================================================
    # 7. Create Visualizations
    # ========================================================================

    print(f"\n🎨 Creating visualizations...")

    visualizer = AttentionVisualizer(output_dir)

    # Per-frame visualizations for all three attention types
    for frame_idx in range(mapper.num_frames):
        # A) TEXT→VISION (main result)
        visualizer.plot_frame_to_text_attention(
            frame_attentions_tv[frame_idx],
            frame_idx,
            (pre_h, pre_w),
            question_tokens,
            save_path=output_dir / f"frame_{frame_idx:03d}_text_to_vision.png"
        )

        # B) VISION→TEXT (true masked attention - will be mostly dark)
        visualizer.plot_attention_matrix(
            frame_attentions_vt[frame_idx],
            title=f"Frame {frame_idx}: Vision→Text Attention (Layer 0, Causal Masked)",
            x_label="Text Token Index",
            y_label="Vision Patch Index",
            x_tokens=question_tokens,
            spatial_grid=(pre_h, pre_w),
            spatial_axis=0,  # Average over vision patches
            save_path=output_dir / f"frame_{frame_idx:03d}_vision_to_text_masked.png"
        )

        # C) Transpose view (for comparison - same data as TEXT→VISION, swapped axes)
        visualizer.plot_attention_matrix(
            frame_attentions_vt_transpose[frame_idx],
            title=f"Frame {frame_idx}: Vision↔Text (Transpose of Text→Vision)",
            x_label="Text Token Index",
            y_label="Vision Patch Index",
            x_tokens=question_tokens,
            spatial_grid=(pre_h, pre_w),
            spatial_axis=0,  # Average over vision patches
            save_path=output_dir / f"frame_{frame_idx:03d}_vision_to_text_transpose.png"
        )

    print(f"   ✓ Saved: {mapper.num_frames} × 3 visualization types per frame")

    # Summary grids (use pre-merger grid)
    visualizer.plot_all_frames_summary(frame_attentions_tv, (pre_h, pre_w))
    print(f"   ✓ Saved: all_frames_summary.png (TEXT→VISION)")

    # Attention evolution video (use pre-merger grid)
    visualizer.create_attention_video(frame_attentions_tv, (pre_h, pre_w), fps=2)
    print(f"   ✓ Saved: attention_evolution.mp4 (TEXT→VISION)")

    # ========================================================================
    # 8. Save Metadata
    # ========================================================================

    metadata = {
        'video_path': video_path,
        'question': question,
        'generated_text': generated_text,
        'model_path': model_path,
        'video_structure': {
            'num_frames': mapper.num_frames,
            'pre_merger_grid': mapper.pre_merger_grid,
            'pre_merger_patches_per_frame': pre_patches,
            'post_merger_patches_per_frame': mapper.patches_per_frame,
            'post_merger_spatial_grid': mapper.spatial_grid,
        },
        'question_tokens': {
            'count': len(text_idx),
            'tokens': question_tokens
        },
        'attention_shape': {
            'prefill': list(attn_probs.shape),
            'text_to_vision': f"[{len(text_idx)}, {pre_patches}] (TEXT→VISION, upsampled to pre-merger grid)",
            'vision_to_text_masked': f"[{pre_patches}, {len(text_idx)}] (VISION→TEXT, causal masked)",
            'vision_to_text_transpose': f"[{pre_patches}, {len(text_idx)}] (Transpose of TEXT→VISION)"
        },
        'streaming_args': {
            'pos_mode': streaming_args.pos_mode,
            'all_text': streaming_args.all_text
        },
        'attention_stats_per_frame': [
            {
                'frame': i,
                'text_to_vision': {
                    'mean': float(frame_attentions_tv[i].mean()),
                    'std': float(frame_attentions_tv[i].std()),
                    'max': float(frame_attentions_tv[i].max()),
                    'min': float(frame_attentions_tv[i].min())
                },
                'vision_to_text_masked': {
                    'mean': float(frame_attentions_vt[i].mean()),
                    'std': float(frame_attentions_vt[i].std()),
                    'max': float(frame_attentions_vt[i].max()),
                    'min': float(frame_attentions_vt[i].min())
                }
            }
            for i in range(mapper.num_frames)
        ]
    }

    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   ✓ Saved: metadata.json")

    # Cleanup
    capture.clear_hooks()

    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETE")
    print(f"📁 Outputs saved to: {output_dir}")
    print("="*80)
    print("\n📊 Verification:")
    print(f"   • Prefill attention shape: {attn_probs.shape}")
    print(f"   • TEXT→VISION shape: [{len(text_idx)}, {pre_patches}] (meaningful patterns)")
    print(f"   • VISION→TEXT shape: [{pre_patches}, {len(text_idx)}] (causal masked, near-zero)")
    print(f"   • Spatial grid: {pre_h}×{pre_w} pre-merger patches")
    print(f"   • TEXT→VISION attention range: "
          f"[{min(a.mean() for a in frame_attentions_tv):.6f}, "
          f"{max(a.mean() for a in frame_attentions_tv):.6f}]")
    print(f"   • VISION→TEXT attention range: "
          f"[{min(a.mean() for a in frame_attentions_vt):.6f}, "
          f"{max(a.mean() for a in frame_attentions_vt):.6f}]")
    print(f"   • Visualizations: {mapper.num_frames} × 3 types (TEXT→VISION, VISION→TEXT masked, transpose)")

    return {
        'attention': attn_probs,
        'frame_attentions_text_to_vision': frame_attentions_tv,
        'frame_attentions_vision_to_text': frame_attentions_vt,
        'frame_attentions_vision_to_text_transpose': frame_attentions_vt_transpose,
        'mapper': mapper,
        'metadata': metadata,
        'generated_text': generated_text
    }


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract attention maps using StreamingVLM's streaming inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required
    parser.add_argument("--video_path", type=str, required=True,
                        help="Path to video file (short clip recommended)")
    parser.add_argument("--question", type=str, required=True,
                        help="Question about the video")

    # Optional
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="HuggingFace model path (Qwen2.5-VL with StreamingVLM patches applied)")
    parser.add_argument("--output_dir", type=str, default="outputs/attn_streaming",
                        help="Output directory")
    parser.add_argument("--max_new_tokens", type=int, default=20,
                        help="Max tokens to generate")
    parser.add_argument("--capture_decode", action="store_true",
                        help="Capture first N decode steps (off by default)")
    parser.add_argument("--max_decode_steps", type=int, default=4,
                        help="How many decode steps to capture if --capture_decode")
    parser.add_argument("--fps", type=float, default=1.0,
                        help="Frames per second for video sampling (lower = less memory)")
    parser.add_argument("--max_pixels", type=int, default=40000,
                        help="Max pixels for video resolution (lower = less memory, default=40000)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Model precision")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device placement (default: cuda:0, use 'auto' for multi-GPU)")

    args = parser.parse_args()

    # Map dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }

    # Run extraction
    results = extract_attention_streaming(
        video_path=args.video_path,
        question=args.question,
        model_path=args.model_path,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        capture_decode=args.capture_decode,
        max_decode_steps=args.max_decode_steps,
        fps=args.fps,
        max_pixels=args.max_pixels,
        dtype=dtype_map[args.dtype],
        device=args.device
    )

    if results:
        return 0
    else:
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
