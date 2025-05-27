import torch
from torch import Tensor
import itertools
from typing import List, Dict, Any, Tuple, Optional, Union

# 帧提取工具
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
        从图像序列中提取首帧和尾帧
        
        Args:
            images: 输入图像张量，形状为 [batch_size, height, width, channels]
            
        Returns:
            first_frame: 首帧图像张量，形状为 [1, height, width, channels]
            last_frame: 尾帧图像张量，形状为 [1, height, width, channels]
        """
        if images.size(0) == 0:
            raise ValueError("输入图像序列为空")
        
        # 提取首帧和尾帧
        first_frame = images[0:1]  # 保持维度为 [1, h, w, c]
        last_frame = images[-1:] if images.size(0) > 1 else first_frame
        
        return (first_frame, last_frame)


# 帧合并工具
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
        按顺序合并多组图像序列
        
        Args:
            frames_1: 第一组图像张量，形状为 [batch_size_1, height, width, channels]
            frames_2 到 frames_5: 可选的额外图像张量组
            
        Returns:
            merged_frames: 合并后的图像张量，形状为 [total_batch_size, height, width, channels]
            frame_count: 合并后的总帧数
        """
        # 收集所有非空输入帧
        all_frames = [frames_1]
        if frames_2 is not None:
            all_frames.append(frames_2)
        if frames_3 is not None:
            all_frames.append(frames_3)
        if frames_4 is not None:
            all_frames.append(frames_4)
        if frames_5 is not None:
            all_frames.append(frames_5)
        
        # 检查所有帧的形状兼容性 (高度和宽度)
        if not all(frames.shape[1:] == frames_1.shape[1:] for frames in all_frames):
            # 获取不匹配的形状信息，用于错误消息
            shapes = [frames.shape for frames in all_frames]
            raise ValueError(f"所有输入帧的尺寸必须相同，但收到了以下形状：{shapes}")
        
        # 按顺序合并所有帧
        merged_frames = torch.cat(all_frames, dim=0)
        frame_count = merged_frames.size(0)
        
        return (merged_frames, frame_count)

# 要在 nodes.py 中导入的节点映射
NODE_CLASS_MAPPINGS = {
    "VHS_ExtractFrames": ExtractFrames,
    "VHS_MergeFrames": MergeFrames,
}

# 要在 nodes.py 中导入的节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "VHS_ExtractFrames": "Extract Frames (First/Last) 🎥🅥🅗🅢",
    "VHS_MergeFrames": "Merge Frames Sequence 🎥🅥🅗🅢",
}
