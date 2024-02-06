from torch import Tensor
import torch

import comfy.utils

from .utils import BIGMIN, BIGMAX


class MergeStrategies:
    MATCH_A = "match A"
    MATCH_B = "match B"
    MATCH_SMALLER = "match smaller"
    MATCH_LARGER = "match larger"

    list_all = [MATCH_A, MATCH_B, MATCH_SMALLER, MATCH_LARGER]


class ScaleMethods:
    NEAREST_EXACT = "nearest-exact"
    BILINEAR = "bilinear"
    AREA = "area"
    BICUBIC = "bicubic"
    BISLERP = "bislerp"

    list_all = [NEAREST_EXACT, BILINEAR, AREA, BICUBIC, BISLERP]


class CropMethods:
    DISABLED = "disabled"
    CENTER = "center"

    list_all = [DISABLED, CENTER]


class SplitLatents:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "latents": ("LATENT",),
                    "split_index": ("INT", {"default": 0, "step": 1, "min": BIGMIN, "max": BIGMAX}),
                },
            }
    
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/latent"

    RETURN_TYPES = ("LATENT", "INT", "LATENT", "INT")
    RETURN_NAMES = ("LATENT_A", "A_count", "LATENT_B", "B_count")
    FUNCTION = "split_latents"

    def split_latents(self, latents: dict, split_index: int):
        latents = latents.copy()
        group_a = latents["samples"][:split_index]
        group_b = latents["samples"][split_index:]
        group_a_latent = {"samples": group_a}
        group_b_latent = {"samples": group_b}
        return (group_a_latent, group_a.size(0), group_b_latent, group_b.size(0))


class SplitImages:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "images": ("IMAGE",),
                    "split_index": ("INT", {"default": 0, "step": 1, "min": BIGMIN, "max": BIGMAX}),
                },
            }
    
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/image"

    RETURN_TYPES = ("IMAGE", "INT", "IMAGE", "INT")
    RETURN_NAMES = ("IMAGE_A", "A_count", "IMAGE_B", "B_count")
    FUNCTION = "split_images"

    def split_images(self, images: Tensor, split_index: int):
        group_a = images[:split_index]
        group_b = images[split_index:]
        return (group_a, group_a.size(0), group_b, group_b.size(0))


class SplitMasks:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "mask": ("MASK",),
                    "split_index": ("INT", {"default": 0, "step": 1, "min": BIGMIN, "max": BIGMAX}),
                },
            }
    
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/mask"

    RETURN_TYPES = ("MASK", "INT", "MASK", "INT")
    RETURN_NAMES = ("MASK_A", "A_count", "MASK_B", "B_count")
    FUNCTION = "split_masks"

    def split_masks(self, mask: Tensor, split_index: int):
        group_a = mask[:split_index]
        group_b = mask[split_index:]
        return (group_a, group_a.size(0), group_b, group_b.size(0))


class MergeLatents:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latents_A": ("LATENT",),
                "latents_B": ("LATENT",),
                "merge_strategy": (MergeStrategies.list_all,),
                "scale_method": (ScaleMethods.list_all,),
                "crop": (CropMethods.list_all,),
            }
        }
    
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/latent"

    RETURN_TYPES = ("LATENT", "INT",)
    RETURN_NAMES = ("LATENT", "count",)
    FUNCTION = "merge"

    def merge(self, latents_A: dict, latents_B: dict, merge_strategy: str, scale_method: str, crop: str):
        latents = []
        latents_A = latents_A.copy()["samples"]
        latents_B = latents_B.copy()["samples"]

        # if not same dimensions, do scaling
        if latents_A.shape[3] != latents_B.shape[3] or latents_A.shape[2] != latents_B.shape[2]:
            A_size = latents_A.shape[3] * latents_A.shape[2]
            B_size = latents_B.shape[3] * latents_B.shape[2]
            # determine which to use
            use_A_as_template = True
            if merge_strategy == MergeStrategies.MATCH_A:
                pass
            elif merge_strategy == MergeStrategies.MATCH_B:
                use_A_as_template = False
            elif merge_strategy in (MergeStrategies.MATCH_SMALLER, MergeStrategies.MATCH_LARGER):
                if A_size <= B_size:
                    use_A_as_template = True if merge_strategy == MergeStrategies.MATCH_SMALLER else False
            # apply scaling
            if use_A_as_template:
                latents_B = comfy.utils.common_upscale(latents_B, latents_A.shape[3], latents_A.shape[2], scale_method, crop)
            else:
                latents_A = comfy.utils.common_upscale(latents_A, latents_B.shape[3], latents_B.shape[2], scale_method, crop)

        latents.append(latents_A)
        latents.append(latents_B)

        merged = {"samples": torch.cat(latents, dim=0)}
        return (merged, len(merged["samples"]),)


class MergeImages:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images_A": ("IMAGE",),
                "images_B": ("IMAGE",),
                "merge_strategy": (MergeStrategies.list_all,),
                "scale_method": (ScaleMethods.list_all,),
                "crop": (CropMethods.list_all,),
            }
        }
    
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/image"

    RETURN_TYPES = ("IMAGE", "INT",)
    RETURN_NAMES = ("IMAGE", "count",)
    FUNCTION = "merge"

    def merge(self, images_A: Tensor, images_B: Tensor, merge_strategy: str, scale_method: str, crop: str):
        images = []
        # if not same dimensions, do scaling
        if images_A.shape[3] != images_B.shape[3] or images_A.shape[2] != images_B.shape[2]:
            images_A = images_A.movedim(-1,1)
            images_B = images_B.movedim(-1,1)

            A_size = images_A.shape[3] * images_A.shape[2]
            B_size = images_B.shape[3] * images_B.shape[2]
            # determine which to use
            use_A_as_template = True
            if merge_strategy == MergeStrategies.MATCH_A:
                pass
            elif merge_strategy == MergeStrategies.MATCH_B:
                use_A_as_template = False
            elif merge_strategy in (MergeStrategies.MATCH_SMALLER, MergeStrategies.MATCH_LARGER):
                if A_size <= B_size:
                    use_A_as_template = True if merge_strategy == MergeStrategies.MATCH_SMALLER else False
            # apply scaling
            if use_A_as_template:
                images_B = comfy.utils.common_upscale(images_B, images_A.shape[3], images_A.shape[2], scale_method, crop)
            else:
                images_A = comfy.utils.common_upscale(images_A, images_B.shape[3], images_B.shape[2], scale_method, crop)
            images_A = images_A.movedim(1,-1)
            images_B = images_B.movedim(1,-1)

        images.append(images_A)
        images.append(images_B)
        all_images = torch.cat(images, dim=0)
        return (all_images, all_images.size(0),)


class MergeMasks:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask_A": ("MASK",),
                "mask_B": ("MASK",),
                "merge_strategy": (MergeStrategies.list_all,),
                "scale_method": (ScaleMethods.list_all,),
                "crop": (CropMethods.list_all,),
            }
        }
    
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/mask"

    RETURN_TYPES = ("MASK", "INT",)
    RETURN_NAMES = ("MASK", "count",)
    FUNCTION = "merge"

    def merge(self, mask_A: Tensor, mask_B: Tensor, merge_strategy: str, scale_method: str, crop: str):
        masks = []
        # if not same dimensions, do scaling
        if mask_A.shape[2] != mask_B.shape[2] or mask_A.shape[1] != mask_B.shape[1]:
            A_size = mask_A.shape[2] * mask_A.shape[1]
            B_size = mask_B.shape[2] * mask_B.shape[1]
            # determine which to use
            use_A_as_template = True
            if merge_strategy == MergeStrategies.MATCH_A:
                pass
            elif merge_strategy == MergeStrategies.MATCH_B:
                use_A_as_template = False
            elif merge_strategy in (MergeStrategies.MATCH_SMALLER, MergeStrategies.MATCH_LARGER):
                if A_size <= B_size:
                    use_A_as_template = True if merge_strategy == MergeStrategies.MATCH_SMALLER else False
            # add dimension where image channels would be expected to work with common_upscale
            mask_A = torch.unsqueeze(mask_A, 1)
            mask_B = torch.unsqueeze(mask_B, 1)
            # apply scaling
            if use_A_as_template:
                mask_B = comfy.utils.common_upscale(mask_B, mask_A.shape[3], mask_A.shape[2], scale_method, crop)
            else:
                mask_A = comfy.utils.common_upscale(mask_A, mask_B.shape[3], mask_B.shape[2], scale_method, crop)
            # undo dimension increase
            mask_A = torch.squeeze(mask_A, 1)
            mask_B = torch.squeeze(mask_B, 1)

        masks.append(mask_A)
        masks.append(mask_B)
        all_masks = torch.cat(masks, dim=0)
        return (all_masks, all_masks.size(0),)


class SelectEveryNthLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "latents": ("LATENT",),
                    "select_every_nth": ("INT", {"default": 1, "min": 1, "max": BIGMAX, "step": 1}),
                },
            }
    
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/latent"

    RETURN_TYPES = ("LATENT", "INT",)
    RETURN_NAMES = ("LATENT", "count",)
    FUNCTION = "select_latents"

    def select_latents(self, latents: dict, select_every_nth: int):
        sub_latents = latents.copy()["samples"][0::select_every_nth]
        return ({"samples": sub_latents}, sub_latents.size(0))
    

class SelectEveryNthImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "images": ("IMAGE",),
                    "select_every_nth": ("INT", {"default": 1, "min": 1, "max": BIGMAX, "step": 1}),
                },
            }
    
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/image"

    RETURN_TYPES = ("IMAGE", "INT",)
    RETURN_NAMES = ("IMAGE", "count",)
    FUNCTION = "select_images"

    def select_images(self, images: Tensor, select_every_nth: int):
        sub_images = images[0::select_every_nth]
        return (sub_images, sub_images.size(0))
    

class SelectEveryNthMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "mask": ("MASK",),
                    "select_every_nth": ("INT", {"default": 1, "min": 1, "max": BIGMAX, "step": 1}),
                },
            }
    
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/mask"

    RETURN_TYPES = ("MASK", "INT",)
    RETURN_NAMES = ("MASK", "count",)
    FUNCTION = "select_masks"

    def select_masks(self, mask: Tensor, select_every_nth: int):
        sub_mask = mask[0::select_every_nth]
        return (sub_mask, sub_mask.size(0))


class GetLatentCount:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latents": ("LATENT",),
            }
        }
    
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/latent"

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("count",)
    FUNCTION = "count_input"

    def count_input(self, latents: dict):
        return (latents["samples"].size(0),)


class GetImageCount:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
            }
        }
    
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/image"

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("count",)
    FUNCTION = "count_input"

    def count_input(self, images: Tensor):
        return (images.size(0),)
    

class GetMaskCount:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
            }
        }
    
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/mask"

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("count",)
    FUNCTION = "count_input"

    def count_input(self, mask: Tensor):
        return (mask.size(0),)


class DuplicateLatents:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latents": ("LATENT",),
                "multiply_by": ("INT", {"default": 1, "min": 1, "max": BIGMAX, "step": 1})
            }
        }
    
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/latent"

    RETURN_TYPES = ("LATENT", "INT",)
    RETURN_NAMES = ("LATENT", "count",)
    FUNCTION = "duplicate_input"

    def duplicate_input(self, latents: dict[str, Tensor], multiply_by: int):
        new_latents = latents.copy()
        full_latents = []
        for n in range(0, multiply_by):
            full_latents.append(new_latents["samples"])
        new_latents["samples"] = torch.cat(full_latents, dim=0)
        return (new_latents, new_latents["samples"].size(0),)


class DuplicateImages:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "multiply_by": ("INT", {"default": 1, "min": 1, "max": BIGMAX, "step": 1})
            }
        }
    
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/image"

    RETURN_TYPES = ("IMAGE", "INT",)
    RETURN_NAMES = ("IMAGE", "count",)
    FUNCTION = "duplicate_input"

    def duplicate_input(self, images: Tensor, multiply_by: int):
        full_images = []
        for n in range(0, multiply_by):
            full_images.append(images)
        new_images = torch.cat(full_images, dim=0)
        return (new_images, new_images.size(0),)


class DuplicateMasks:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "multiply_by": ("INT", {"default": 1, "min": 1, "max": BIGMAX, "step": 1})
            }
        }
    
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/mask"

    RETURN_TYPES = ("MASK", "INT",)
    RETURN_NAMES = ("MASK", "count",)
    FUNCTION = "duplicate_input"

    def duplicate_input(self, mask: Tensor, multiply_by: int):
        full_masks = []
        for n in range(0, multiply_by):
            full_masks.append(mask)
        new_mask = torch.cat(full_masks, dim=0)
        return (new_mask, new_mask.size(0),)


# class SelectLatents:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {
#                 "required": {
#                     "images": ("IMAGE",),
#                     "select_indeces": ("STRING", {"default": ""}),
#                 },
#             }
