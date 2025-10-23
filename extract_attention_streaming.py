#!/usr/bin/env python3
"""
StreamingVLM Attention Extraction

Extracts and visualizes multimodal attention maps from StreamingVLM's first decoder layer.

Usage:
    python extract_attention_streaming.py --video_path video.mp4 --question "What is happening?" --output_dir outputs/test
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
from streaming_vlm.inference.qwen2_5.patch_model import convert_qwen2_5_to_streaming
from streaming_vlm.inference.streaming_args import StreamingArgs

try:
    from qwen_vl_utils import process_vision_info
    HAS_QWEN_VL_UTILS = True
except ImportError:
    HAS_QWEN_VL_UTILS = False
    print("Warning: qwen_vl_utils not found")

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


class StreamingAttentionCapture:
    """Captures attention from streaming inference during prefill and decode phases."""

    def __init__(self, capture_decode: bool = False, max_decode_steps: int = 4):
        self.prefill_attention = None
        self.decode_attentions = []
        self.capture_decode = capture_decode
        self.max_decode_steps = max_decode_steps
        self.hook_handles = []
        self.is_prefill = True
        self.decode_count = 0

    def register_hook(self, model, layer_idx: int = 0):
        def attention_hook(module, input, output):
            if isinstance(output, tuple) and len(output) >= 2:
                attn_weights = output[1]
                if attn_weights is not None:
                    bsz, num_heads, q_len, k_len = attn_weights.shape
                    if self.is_prefill and q_len > 1:
                        self.prefill_attention = attn_weights.detach()
                        self.is_prefill = False
                        print(f"[Prefill] Captured attention shape: {attn_weights.shape}")
                    elif self.capture_decode and self.decode_count < self.max_decode_steps:
                        self.decode_attentions.append(attn_weights.detach())
                        self.decode_count += 1
                        print(f"[Decode {self.decode_count}] Captured attention shape: {attn_weights.shape}")

        target_module = model.model.language_model.layers[layer_idx].self_attn
        handle = target_module.register_forward_hook(attention_hook)
        self.hook_handles.append(handle)
        print(f"Registered hook on model.model.language_model.layers[{layer_idx}].self_attn")

    def clear_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []


class TokenIndexMapper:
    """Maps token indices to vision/text categories and per-frame boundaries."""

    def __init__(self, input_ids, video_token_id, video_grid_thw):
        self.input_ids = input_ids
        self.video_token_id = video_token_id
        self.video_grid_thw = video_grid_thw
        self.device = input_ids.device

        video_mask = (input_ids[0] == video_token_id)
        self.vision_idx = torch.where(video_mask)[0].to(self.device)

        if len(video_grid_thw.shape) == 2:
            t, h, w = video_grid_thw[0]
        else:
            t, h, w = video_grid_thw

        self.num_frames = int(t.item())
        self.pre_merger_grid = (int(h.item()), int(w.item()))
        pre_merger_patches = int(h.item()) * int(w.item())

        total_vision_tokens = len(self.vision_idx)
        assert total_vision_tokens > 0, "No vision tokens found"

        self.patches_per_frame = total_vision_tokens // self.num_frames
        self.spatial_grid = self._infer_spatial_grid(self.patches_per_frame)

    def _infer_spatial_grid(self, patches_per_frame: int) -> Tuple[int, int]:
        side = int(round(math.sqrt(patches_per_frame)))
        if side * side == patches_per_frame:
            return (side, side)
        pre_h, pre_w = self.pre_merger_grid
        for merger_side in range(1, 10):
            post_h = pre_h // merger_side
            post_w = pre_w // merger_side
            if post_h * post_w == patches_per_frame:
                return (post_h, post_w)
        return (side, patches_per_frame // side)

    def get_frame_indices(self, frame_idx: int) -> torch.Tensor:
        start = frame_idx * self.patches_per_frame
        end = start + self.patches_per_frame
        return self.vision_idx[start:end]

    def get_question_tokens(self, processor) -> torch.Tensor:
        ids = self.input_ids[0]
        dev = self.device

        im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        user_id = processor.tokenizer.convert_tokens_to_ids("user")
        assistant_id = processor.tokenizer.convert_tokens_to_ids("assistant")
        nl_id = processor.tokenizer.convert_tokens_to_ids("\n")

        text_start, text_end = None, None
        for i in range(len(ids)):
            if ids[i].item() == im_start_id:
                s = i
                e = i + 1
                while e < len(ids) and ids[e].item() != im_end_id:
                    e += 1
                if s + 1 < len(ids) and ids[s + 1].item() == user_id and e > s:
                    t0 = s + 2
                    if t0 < len(ids) and ids[t0].item() == nl_id:
                        t0 += 1
                    text_start, text_end = t0, e
                    break

        if text_start is None:
            text_start = (self.vision_idx[-1].item() + 1) if len(self.vision_idx) > 0 else 0
            text_end = self.input_ids.shape[1]

        keep = []
        for i in range(text_start, text_end):
            tok = ids[i].item()
            if tok in (im_start_id, im_end_id, user_id, assistant_id, nl_id):
                continue
            keep.append(i)

        return torch.tensor(keep, device=dev, dtype=torch.long)


class AttentionVisualizer:
    """Visualizes attention patterns as heatmaps and videos."""

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
        if spatial_grid is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        else:
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax2 = None

        if HAS_SEABORN:
            sns.heatmap(attention, cmap='hot', ax=ax1, cbar=True)
        else:
            im1 = ax1.imshow(attention, cmap='hot', aspect='auto')
            plt.colorbar(im1, ax=ax1)

        ax1.set_xlabel(x_label)
        ax1.set_ylabel(y_label)
        ax1.set_title(title)

        if y_tokens and len(y_tokens) <= 20:
            ax1.set_yticks(range(len(y_tokens)))
            ax1.set_yticklabels(y_tokens, rotation=0)
        if x_tokens and len(x_tokens) <= 20:
            ax1.set_xticks(range(len(x_tokens)))
            ax1.set_xticklabels(x_tokens, rotation=45, ha='right')

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
            avg_label = 'vision patches' if spatial_axis == 0 else 'text tokens'
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
        h, w = spatial_grid
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        if HAS_SEABORN:
            sns.heatmap(attention, cmap='hot', ax=ax1, cbar=True)
        else:
            im1 = ax1.imshow(attention, cmap='hot', aspect='auto')
            plt.colorbar(im1, ax=ax1)

        ax1.set_xlabel('Vision Patch Index')
        ax1.set_ylabel('Text Token Index')
        ax1.set_title(f'Frame {frame_idx}: Text→Vision Attention')

        if len(text_tokens) <= 20:
            ax1.set_yticks(range(len(text_tokens)))
            ax1.set_yticklabels(text_tokens, rotation=0)

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

    def plot_matrix_no_avg(
        self,
        mat: np.ndarray,
        title: str,
        x_label: str,
        y_label: str,
        save_path: str
    ):
        """Plot raw attention matrix without averaging."""
        fig, ax = plt.subplots(figsize=(12, 8))

        if HAS_SEABORN:
            sns.heatmap(mat, cmap='hot', ax=ax, cbar=True, vmin=0.0, vmax=1.0)
        else:
            im = ax.imshow(mat, aspect='auto', origin='lower', cmap='hot', vmin=0.0, vmax=1.0)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_all_frames_summary(
        self,
        frame_attentions: List[np.ndarray],
        spatial_grid: Tuple[int, int],
        save_path: str = None
    ):
        h, w = spatial_grid
        num_frames = len(frame_attentions)

        ncols = min(num_frames, 8)
        nrows = (num_frames + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
        if num_frames == 1:
            axes = np.array([axes])
        axes = axes.flatten() if num_frames > 1 else axes

        for idx, attn in enumerate(frame_attentions):
            spatial_attn = attn.mean(axis=0).reshape(h, w)
            ax = axes[idx] if num_frames > 1 else axes[0]

            if HAS_SEABORN:
                sns.heatmap(spatial_attn, cmap='hot', ax=ax, cbar=False, square=True)
            else:
                ax.imshow(spatial_attn, cmap='hot')

            ax.set_title(f'Frame {idx}', fontsize=10)
            ax.axis('off')

        for idx in range(num_frames, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        if save_path is None:
            save_path = self.output_dir / "all_frames_summary.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def create_attention_video(
        self,
        frame_attentions: List[np.ndarray],
        spatial_grid: Tuple[int, int],
        fps: int = 2,
        save_path: str = None
    ):
        if not HAS_OPENCV:
            print("   ⚠️  OpenCV not available, skipping video creation")
            return

        h, w = spatial_grid
        dpi = 100
        fig_w, fig_h = 6, 6
        frame_w, frame_h = int(fig_w * dpi), int(fig_h * dpi)

        if save_path is None:
            save_path = self.output_dir / "attention_evolution.mp4"

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(save_path), fourcc, fps, (frame_w, frame_h))

        for idx, attn in enumerate(frame_attentions):
            spatial_attn = attn.mean(axis=0).reshape(h, w)

            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            if HAS_SEABORN:
                sns.heatmap(spatial_attn, cmap='hot', ax=ax, cbar=True, square=True, vmin=0, vmax=1)
            else:
                im = ax.imshow(spatial_attn, cmap='hot', vmin=0, vmax=1)
                plt.colorbar(im, ax=ax)

            ax.set_title(f'Frame {idx}: Spatial Attention', fontsize=14)
            ax.axis('off')

            plt.tight_layout()
            fig.canvas.draw()

            img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)

            video_writer.write(img_bgr)
            plt.close(fig)

        video_writer.release()


def extract_attention_streaming(
    model_name: str,
    video_path: str,
    question: str,
    output_dir: str,
    device: str = "cuda:0",
    capture_decode: bool = False,
    max_decode_steps: int = 4,
    streaming_args: StreamingArgs = None
) -> Dict:
    """Extract attention maps from StreamingVLM inference."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STREAMING ATTENTION EXTRACTION")
    print("=" * 80)

    # Load model
    print(f"\n[1/7] Loading model: {model_name}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    print(f"       Device: {device} (single GPU)")

    # Apply streaming patches
    print(f"[2/7] Applying StreamingVLM patches...")
    if streaming_args is None:
        streaming_args = StreamingArgs()
    model = convert_qwen2_5_to_streaming(model)

    # Load processor
    print(f"[3/7] Loading processor...")
    processor = AutoProcessor.from_pretrained(model_name)

    # Register attention hook
    print(f"[4/7] Registering attention hook (capture_decode={capture_decode})...")
    capture = StreamingAttentionCapture(capture_decode, max_decode_steps)
    capture.register_hook(model, layer_idx=0)

    # Process video
    print(f"[5/7] Processing video: {video_path}")
    print(f"       Question: {question}")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path, "fps": 1.0, "max_pixels": 40000},
                {"type": "text", "text": question},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if HAS_QWEN_VL_UTILS:
        image_inputs, video_inputs = process_vision_info(messages)
    else:
        raise ImportError("qwen_vl_utils is required")

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    inputs = inputs.to(device)

    print(f"       Input shape: {inputs.input_ids.shape}")
    print(f"       Video grid THW: {inputs.video_grid_thw}")

    # Generate
    print(f"[6/7] Generating with StreamingVLM...")
    model.streaming_args = streaming_args

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.1,
            output_attentions=True
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    generated_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    print(f"\n       Generated: {generated_text[0]}")

    capture.clear_hooks()

    # Extract attention
    print(f"[7/7] Extracting and visualizing attention...\n")

    if capture.prefill_attention is None:
        raise RuntimeError("No prefill attention captured")

    attn_probs = capture.prefill_attention

    print(f"✓ Prefill attention shape: {attn_probs.shape}")

    # Get token mapping
    video_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    mapper = TokenIndexMapper(inputs.input_ids, video_token_id, inputs.video_grid_thw)

    pre_h, pre_w = mapper.pre_merger_grid
    pre_patches = pre_h * pre_w

    print(f"   Pre-merger grid: {mapper.num_frames} frames × ({pre_h}, {pre_w}) = {pre_patches} patches")
    print(f"   Post-merger tokens: {mapper.patches_per_frame * mapper.num_frames} vision tokens in input_ids")
    if mapper.patches_per_frame != pre_patches:
        print(f"   ⚠️  Merger applied: {pre_patches} → {mapper.patches_per_frame} tokens")
        print(f"   Using post-merger: {mapper.patches_per_frame} patches/frame, spatial grid = {mapper.spatial_grid} (merger: 2x2)")

    print(f"✓ Video structure: {mapper.num_frames} frames")
    print(f"   Grid (pre-merger): ({pre_h}, {pre_w}) → {pre_patches} patches/frame")
    print(f"   Vision tokens (post-merger): {mapper.patches_per_frame * mapper.num_frames} (spatial grid ≈ {mapper.spatial_grid})")

    # Average attention over heads
    attn_mean = attn_probs.mean(dim=1).squeeze(0)
    print(f"✓ Attention mean shape: {attn_mean.shape}, device: {attn_mean.device}")

    # Get question tokens
    text_idx = mapper.get_question_tokens(processor)
    question_tokens = [processor.tokenizer.decode([t]) for t in inputs.input_ids[0][text_idx]]

    print(f"✓ Question tokens kept: {len(text_idx)}")
    print(f"   Tokens: {question_tokens}")
    print(f"   Text position range: [{text_idx[0].item()}, {text_idx[-1].item()}]")

    # Extract per-frame attention
    frame_attentions_tv = []
    frame_attentions_vt = []

    print(f"\n📊 Extracting per-frame attention in both directions (upsampling to pre-merger grid):")

    for frame_idx in range(mapper.num_frames):
        frame_vision_idx = mapper.get_frame_indices(frame_idx)

        # TEXT→VISION
        frame_attn_tv = torch.index_select(attn_mean, 0, text_idx)
        frame_attn_tv = torch.index_select(frame_attn_tv, 1, frame_vision_idx)
        frame_attn_tv = torch.nn.functional.normalize(frame_attn_tv, p=1, dim=1)

        # VISION→TEXT
        frame_attn_vt = torch.index_select(attn_mean, 0, frame_vision_idx)
        frame_attn_vt = torch.index_select(frame_attn_vt, 1, text_idx)
        frame_attn_vt = torch.nn.functional.normalize(frame_attn_vt, p=1, dim=1)

        # Upsample if merger was applied
        nv_merged = mapper.patches_per_frame
        if nv_merged != pre_patches:
            ratio = pre_patches // nv_merged
            merger_side = int(round(math.sqrt(ratio)))
            assert merger_side * merger_side * nv_merged == pre_patches

            # Upsample TEXT→VISION
            post_h, post_w = mapper.spatial_grid
            attn_hw = frame_attn_tv.view(frame_attn_tv.size(0), post_h, post_w)
            attn_hw = torch.repeat_interleave(attn_hw, repeats=merger_side, dim=1)
            attn_hw = torch.repeat_interleave(attn_hw, repeats=merger_side, dim=2)
            frame_attn_tv = attn_hw.view(frame_attn_tv.size(0), pre_patches)

            # Upsample VISION→TEXT
            attn_hw_vt = frame_attn_vt.view(post_h, post_w, frame_attn_vt.size(1))
            attn_hw_vt = torch.repeat_interleave(attn_hw_vt, repeats=merger_side, dim=0)
            attn_hw_vt = torch.repeat_interleave(attn_hw_vt, repeats=merger_side, dim=1)
            frame_attn_vt = attn_hw_vt.view(pre_patches, frame_attn_vt.size(1))

        # Convert to numpy
        attn_tv_np = frame_attn_tv.detach().float().cpu().numpy()
        attn_vt_np = frame_attn_vt.detach().float().cpu().numpy()

        frame_attentions_tv.append(attn_tv_np)
        frame_attentions_vt.append(attn_vt_np)

        print(f"   Frame {frame_idx:2d}:")
        print(f"      TEXT→VISION: {attn_tv_np.shape} (upsampled to {pre_h}×{pre_w}), mean={attn_tv_np.mean():.6f}, max={attn_tv_np.max():.6f}")
        print(f"      VISION→TEXT (masked): {attn_vt_np.shape}, mean={attn_vt_np.mean():.6f}, max={attn_vt_np.max():.6f}")

    # Save tensors
    print(f"\n💾 Saving raw tensors...")
    torch.save(attn_probs, output_dir / "prefill_attention.pt")
    print(f"   ✓ Saved: prefill_attention.pt")

    for i in range(mapper.num_frames):
        np.save(output_dir / f"frame_{i:03d}_text_to_vision.npy", frame_attentions_tv[i])
        np.save(output_dir / f"frame_{i:03d}_vision_to_text.npy", frame_attentions_vt[i])

    print(f"   ✓ Saved: {mapper.num_frames} × 2 attention matrices per frame")
    print(f"      - *_text_to_vision.npy: [{len(text_idx)}, {pre_patches}] TEXT→VISION")
    print(f"      - *_vision_to_text.npy: [{pre_patches}, {len(text_idx)}] VISION→TEXT (causal masked)")

    # Save decode attentions
    if capture_decode and len(capture.decode_attentions) > 0:
        for i, attn in enumerate(capture.decode_attentions):
            torch.save(attn, output_dir / f"decode_step_{i:02d}_attention.pt")
        print(f"   ✓ Saved: {len(capture.decode_attentions)} decode-step tensors")

    # Create visualizations
    print(f"\n🎨 Creating visualizations...")

    visualizer = AttentionVisualizer(output_dir)

    for frame_idx in range(mapper.num_frames):
        # TEXT→VISION
        visualizer.plot_frame_to_text_attention(
            frame_attentions_tv[frame_idx],
            frame_idx,
            (pre_h, pre_w),
            question_tokens,
            save_path=output_dir / f"frame_{frame_idx:03d}_text_to_vision.png"
        )

        # VISION→TEXT (raw matrix, no averaging)
        visualizer.plot_matrix_no_avg(
            frame_attentions_vt[frame_idx],
            title=f"Frame {frame_idx}: Vision→Text Attention (Layer 0, prefill; no averaging)",
            x_label="Text Token Index",
            y_label="Vision Patch Index",
            save_path=output_dir / f"frame_{frame_idx:03d}_vision_to_text.png"
        )

    print(f"   ✓ Saved: {mapper.num_frames} × 2 visualization types per frame (TEXT→VISION and VISION→TEXT)")

    # Summary visualizations
    visualizer.plot_all_frames_summary(frame_attentions_tv, (pre_h, pre_w))
    print(f"   ✓ Saved: all_frames_summary.png (TEXT→VISION)")

    visualizer.create_attention_video(frame_attentions_tv, (pre_h, pre_w), fps=2)
    print(f"   ✓ Saved: attention_evolution.mp4")

    # Save metadata
    metadata = {
        "model": model_name,
        "video_path": video_path,
        "question": question,
        "num_frames": mapper.num_frames,
        "spatial_grid": mapper.spatial_grid,
        "pre_merger_grid": mapper.pre_merger_grid,
        "patches_per_frame": mapper.patches_per_frame,
        "num_text_tokens": len(text_idx),
        "attention_shape": list(attn_probs.shape),
        "generated_text": generated_text[0]
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Summary
    print(f"\n" + "=" * 80)
    print(f"EXTRACTION COMPLETE")
    print(f"=" * 80)
    print(f"   • Output dir: {output_dir}")
    print(f"   • Frames: {mapper.num_frames}")
    print(f"   • TEXT→VISION attention range: [{min(a.mean() for a in frame_attentions_tv):.6f}, {max(a.mean() for a in frame_attentions_tv):.6f}]")
    print(f"   • VISION→TEXT attention range: [{min(a.mean() for a in frame_attentions_vt):.6f}, {max(a.mean() for a in frame_attentions_vt):.6f}]")
    print(f"   • Visualizations: {mapper.num_frames} × 2 types (TEXT→VISION, VISION→TEXT)")

    return {
        'attention': attn_probs,
        'frame_attentions_text_to_vision': frame_attentions_tv,
        'frame_attentions_vision_to_text': frame_attentions_vt,
        'mapper': mapper,
        'metadata': metadata,
        'generated_text': generated_text
    }


def main():
    parser = argparse.ArgumentParser(description="Extract attention maps using StreamingVLM")
    parser.add_argument("--video_path", type=str, required=True, help="Path to video file")
    parser.add_argument("--question", type=str, required=True, help="Question about the video")
    parser.add_argument("--output_dir", type=str, default="outputs/attn", help="Output directory")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Model name")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--capture_decode", action="store_true", help="Capture decode-step attention")
    parser.add_argument("--max_decode_steps", type=int, default=4, help="Max decode steps to capture")

    args = parser.parse_args()

    extract_attention_streaming(
        model_name=args.model_name,
        video_path=args.video_path,
        question=args.question,
        output_dir=args.output_dir,
        device=args.device,
        capture_decode=args.capture_decode,
        max_decode_steps=args.max_decode_steps
    )


if __name__ == "__main__":
    import sys
    sys.exit(main())