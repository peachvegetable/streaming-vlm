#!/usr/bin/env python3
"""
StreamingVLM Attention Map Extraction (Version Compatible)

This version works WITHOUT relying on convert_qwen2_5_to_streaming,
which has model structure dependencies.

Usage:
    python extract_attention_maps_fixed.py --video_path VIDEO --question "QUESTION"
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
import tempfile
import urllib.request
from typing import Dict, List, Tuple, Optional
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# Import qwen_vl_utils for proper video processing
try:
    from qwen_vl_utils import process_vision_info
    HAS_QWEN_VL_UTILS = True
except ImportError:
    HAS_QWEN_VL_UTILS = False
    print("Warning: qwen_vl_utils not found. Video processing may fail.")
    print("Install with: pip install qwen-vl-utils")

# Import only what we need
from streaming_vlm.inference.streaming_args import StreamingArgs

# Import these from the original files
import sys
sys.path.insert(0, str(Path(__file__).parent))
from extract_attention_maps import (
    download_video_from_url,
    is_url,
    AttentionCapture,
    TokenIndexMapper,
    AttentionVisualizer
)


def extract_and_visualize_attention_fixed(
    model,
    processor,
    video_path: str,
    question: str,
    output_dir: str = "outputs/attn_maps",
    max_new_tokens: int = 20
):
    """
    Extract attention WITHOUT using streaming patches.
    This works with the standard transformers model structure.
    """

    print("\n" + "="*80)
    print("ATTENTION EXTRACTION (FIXED VERSION - NO PATCHING)")
    print("="*80)

    # Initialize components
    capture = AttentionCapture()
    visualizer = AttentionVisualizer(output_dir)

    # Download video if URL provided
    temp_video_path = None
    if is_url(video_path):
        print(f"\nDetected URL: {video_path}")
        temp_video_path = download_video_from_url(video_path, output_dir=output_dir)
        actual_video_path = temp_video_path
        original_video_path = video_path
    else:
        actual_video_path = video_path
        original_video_path = video_path

    # Register hook on Layer 0 (DIRECTLY, no patching needed)
    # The structure is: model.model.layers[0].self_attn
    print(f"\nRegistering hook on model.model.layers[0].self_attn")

    # Register hook directly on the self_attn module
    target_module = model.model.layers[0].self_attn

    def attention_hook(module, input, output):
        """Capture attention weights from self-attention forward."""
        # Output format varies by implementation
        # For Qwen2.5-VL eager attention: (attn_output, attn_weights, past_key_value)
        if isinstance(output, tuple) and len(output) >= 2:
            attn_output, attn_weights = output[0], output[1]
            if attn_weights is not None:
                capture.captured_attention['layer_0'] = {
                    'attn_probs': attn_weights.detach().cpu(),
                    'shape': attn_weights.shape
                }
                print(f"[Hook Layer 0] Captured attention shape: {attn_weights.shape}")
            else:
                print(f"[Hook Layer 0] WARNING: attn_weights is None")

    hook_handle = target_module.register_forward_hook(attention_hook)
    capture.hook_handles.append(hook_handle)

    # Prepare input
    print(f"\nProcessing video: {original_video_path}")
    print(f"Question: {question}")

    messages = [
        {"role": "user", "content": [
            {"type": "video", "video": actual_video_path},
            {"type": "text", "text": question}
        ]}
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Process vision info if available (recommended way)
    if HAS_QWEN_VL_UTILS:
        print("Using qwen_vl_utils for video processing...")
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(model.device)
    else:
        # Fallback: try direct path (may not work for all setups)
        print("Trying direct video path processing...")
        try:
            inputs = processor(text=[text], videos=[actual_video_path], return_tensors="pt").to(model.device)
        except ValueError as e:
            print(f"\nERROR: Could not process video: {e}")
            print("\nPlease install qwen_vl_utils:")
            print("  pip install qwen-vl-utils")
            capture.clear_hooks()
            if temp_video_path:
                import os
                try:
                    os.remove(temp_video_path)
                except:
                    pass
            return None

    print(f"\nInput shape: {inputs['input_ids'].shape}")
    print(f"Video grid THW: {inputs['video_grid_thw']}")

    # Run generation (NO streaming_args since we're not using patches)
    print("\nGenerating (with attention capture)...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            output_attentions=True,  # Force eager mode
            use_cache=True,
            do_sample=False
        )

    # Decode output
    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    print(f"\nGenerated: {generated_text}")

    # Get captured attention
    if 'layer_0' in capture.captured_attention:
        attn_probs = capture.captured_attention['layer_0']['attn_probs']
    else:
        attn_probs = None

    if attn_probs is None:
        print("\nERROR: No attention captured!")
        capture.clear_hooks()
        if temp_video_path:
            import os
            try:
                os.remove(temp_video_path)
            except:
                pass
        return

    print(f"\nCaptured attention shape: {attn_probs.shape}")
    bsz, num_heads, q_len, k_len = attn_probs.shape

    # Map token indices
    mapper = TokenIndexMapper(
        inputs['input_ids'],
        model.config.video_token_id,
        inputs['video_grid_thw']
    )

    # Get text and vision indices
    text_idx = mapper.get_question_indices()
    print(f"\nQuestion token indices: {text_idx}")
    print(f"Number of question tokens: {len(text_idx)}")

    # Decode question tokens
    question_tokens = [
        processor.tokenizer.decode([inputs['input_ids'][0, i].item()])
        for i in text_idx
    ]
    print(f"Question tokens: {question_tokens}")

    # Average attention over heads
    attn_mean = attn_probs.mean(dim=1)[0]  # [q_len, k_len]

    # Extract per-frame attention
    frame_attentions = []

    print(f"\nExtracting attention for {mapper.num_frames} frames...")
    for frame_idx in range(mapper.num_frames):
        frame_vision_idx = mapper.get_frame_indices(frame_idx)

        frame_attn = attn_mean[frame_vision_idx, :][:, text_idx].numpy()
        frame_attentions.append(frame_attn)

        print(f"Frame {frame_idx}: attention shape {frame_attn.shape}, "
              f"mean={frame_attn.mean():.6f}, max={frame_attn.max():.6f}")

        # Visualize this frame
        visualizer.plot_frame_to_text_attention(
            frame_attn,
            frame_idx,
            mapper.spatial_grid,
            question_tokens
        )

    # Create summary visualization
    visualizer.plot_all_frames_summary(frame_attentions, mapper.spatial_grid)

    # Create video
    visualizer.create_attention_video(frame_attentions, mapper.spatial_grid)

    # Save metadata
    metadata = {
        'video_path': original_video_path,
        'question': question,
        'generated_text': generated_text,
        'num_frames': mapper.num_frames,
        'patches_per_frame': mapper.patches_per_frame,
        'spatial_grid': mapper.spatial_grid,
        'num_question_tokens': len(text_idx),
        'question_tokens': question_tokens,
        'attention_shape': list(attn_probs.shape),
        'attention_stats_per_frame': [
            {
                'frame': i,
                'mean': float(attn.mean()),
                'std': float(attn.std()),
                'max': float(attn.max()),
                'min': float(attn.min())
            }
            for i, attn in enumerate(frame_attentions)
        ]
    }

    metadata_path = visualizer.output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved metadata: {metadata_path}")

    # Cleanup
    capture.clear_hooks()

    if temp_video_path:
        try:
            import os
            os.remove(temp_video_path)
            print(f"\nCleaned up temporary file: {temp_video_path}")
        except Exception as e:
            print(f"\nWarning: Could not remove temporary file {temp_video_path}: {e}")

    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print(f"Outputs saved to: {visualizer.output_dir}")
    print("="*80)

    return {
        'attention': attn_probs,
        'frame_attentions': frame_attentions,
        'mapper': mapper,
        'metadata': metadata
    }


def main():
    parser = argparse.ArgumentParser(description="Extract attention maps (fixed version)")
    parser.add_argument("--model_path", type=str, default="mit-han-lab/StreamingVLM",
                        help="Path to model")
    parser.add_argument("--video_path", type=str, required=True,
                        help="Path to video or URL (Note: YouTube needs special handling)")
    parser.add_argument("--question", type=str, default="What is happening in this video?",
                        help="Question about the video")
    parser.add_argument("--output_dir", type=str, default="outputs/attn_maps",
                        help="Output directory")
    parser.add_argument("--max_new_tokens", type=int, default=20,
                        help="Max tokens to generate")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Model dtype")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Device to use (auto/cpu/cuda)")

    args = parser.parse_args()

    # Map dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    dtype = dtype_map[args.dtype]

    # Load model (NO patching!)
    print(f"Loading model on device: {args.device}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=args.device,
        attn_implementation="eager"  # Critical!
    )
    model.eval()

    # Load processor
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(args.model_path)

    # Run extraction
    results = extract_and_visualize_attention_fixed(
        model=model,
        processor=processor,
        video_path=args.video_path,
        question=args.question,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens
    )

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
