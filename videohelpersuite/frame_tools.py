import torch
from torch import Tensor
import itertools
from typing import List, Dict, Any, Tuple, Optional, Union

# Frame extraction tools
class ExtractFrames:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
            }
        }
    
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/image"
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("first_frame", "last_frame")
    FUNCTION = "extract_frames"

    def extract_frames(self, images: Tensor):
        """
        Extract the first and last frames from an image sequence
        
        Args:
            images: Input image tensor with shape [batch_size, height, width, channels]
            
        Returns:
            first_frame: First frame image tensor with shape [1, height, width, channels]
            last_frame: Last frame image tensor with shape [1, height, width, channels]
        """
        if images.size(0) == 0:
            raise ValueError("Input image sequence is empty")
        
        # Extract first and last frames
        first_frame = images[0:1]  # Keep dimensions as [1, h, w, c]
        last_frame = images[-1:] if images.size(0) > 1 else first_frame
        
        return (first_frame, last_frame)


# Frame merging tools
class MergeFrames:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frames_1": ("IMAGE",),
            },
            "optional": {
                "frames_2": ("IMAGE",),
                "frames_3": ("IMAGE",),
                "frames_4": ("IMAGE",),
                "frames_5": ("IMAGE",),
            }
        }
    
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/image"
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("merged_frames", "frame_count")
    FUNCTION = "merge_frames"

    def merge_frames(self, frames_1: Tensor, 
                    frames_2: Optional[Tensor] = None,
                    frames_3: Optional[Tensor] = None, 
                    frames_4: Optional[Tensor] = None,
                    frames_5: Optional[Tensor] = None):
        """
        Merge multiple image sequences in order
        
        Args:
            frames_1: First group of image tensors with shape [batch_size_1, height, width, channels]
            frames_2 to frames_5: Optional additional groups of image tensors
            
        Returns:
            merged_frames: Merged image tensor with shape [total_batch_size, height, width, channels]
            frame_count: Total number of frames after merging
        """
        # Collect all non-empty input frames
        all_frames = [frames_1]
        if frames_2 is not None:
            all_frames.append(frames_2)
        if frames_3 is not None:
            all_frames.append(frames_3)
        if frames_4 is not None:
            all_frames.append(frames_4)
        if frames_5 is not None:
            all_frames.append(frames_5)
        
        # Check shape compatibility of all frames (height and width)
        if not all(frames.shape[1:] == frames_1.shape[1:] for frames in all_frames):
            # Get mismatched shape information for error message
            shapes = [frames.shape for frames in all_frames]
            raise ValueError(f"All input frames must have the same dimensions, but received shapes: {shapes}")
        
        # Merge all frames in order
        merged_frames = torch.cat(all_frames, dim=0)
        frame_count = merged_frames.size(0)
        
        return (merged_frames, frame_count)

# Node mappings to be imported in nodes.py
NODE_CLASS_MAPPINGS = {
    "VHS_ExtractFrames": ExtractFrames,
    "VHS_MergeFrames": MergeFrames,
}

# Node display name mappings to be imported in nodes.py
NODE_DISPLAY_NAME_MAPPINGS = {
    "VHS_ExtractFrames": "Extract Frames (First/Last) 🎥🅥🅗🅢",
    "VHS_MergeFrames": "Merge Frames Sequence 🎥🅥🅗🅢",
}
