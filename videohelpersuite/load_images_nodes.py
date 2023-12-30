import os
import hashlib
import numpy as np
import torch
from PIL import Image, ImageOps

import folder_paths
from comfy.k_diffusion.utils import FolderOfImages
from .logger import logger
from .utils import calculate_file_hash, get_sorted_dir_files_from_directory, validate_path


def is_changed_load_images(directory: str, image_load_cap: int = 0, skip_first_images: int = 0, select_every_nth: int = 1):
    if not os.path.isdir(directory):
            return False
        
    dir_files = get_sorted_dir_files_from_directory(directory, skip_first_images, select_every_nth, FolderOfImages.IMG_EXTENSIONS)
    if image_load_cap != 0:
        dir_files = dir_files[:image_load_cap]

    m = hashlib.sha256()
    for filepath in dir_files:
        m.update(calculate_file_hash(filepath).encode()) # strings must be encoded before hashing
    return m.digest().hex()


def validate_load_images(directory: str):
    if not os.path.isdir(directory):
            return f"Directory '{directory}' cannot be found."
    dir_files = os.listdir(directory)
    if len(dir_files) == 0:
        return f"No files in directory '{directory}'."

    return True


def load_images(directory: str, image_load_cap: int = 0, skip_first_images: int = 0, select_every_nth: int = 1):
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory '{directory} cannot be found.")

    dir_files = get_sorted_dir_files_from_directory(directory, skip_first_images, select_every_nth, FolderOfImages.IMG_EXTENSIONS)

    if len(dir_files) == 0:
        raise FileNotFoundError(f"No files in directory '{directory}'.")

    images = []
    masks = []

    limit_images = False
    if image_load_cap > 0:
        limit_images = True
    image_count = 0
    loaded_alpha = False
    zero_mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")

    for image_path in dir_files:
        if limit_images and image_count >= image_load_cap:
            break
        i = Image.open(image_path)
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
            if not loaded_alpha:
                loaded_alpha = True
                zero_mask = torch.zeros((len(image[0]),len(image[0][0])), dtype=torch.float32, device="cpu")
                masks = [zero_mask] * image_count
        else:
            mask = zero_mask
        images.append(image)
        masks.append(mask)
        image_count += 1
    
    if len(images) == 0:
        raise FileNotFoundError(f"No images could be loaded from directory '{directory}'.")

    return (torch.cat(images, dim=0), torch.stack(masks, dim=0), image_count)


class LoadImagesFromDirectoryUpload:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        directories = []
        for item in os.listdir(input_dir):
            if not os.path.isfile(os.path.join(input_dir, item)) and item != "clipspace":
                directories.append(item)
        return {
            "required": {
                "directory": (directories,),
            },
            "optional": {
                "image_load_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                "skip_first_images": ("INT", {"default": 0, "min": 0, "step": 1}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "INT")
    FUNCTION = "load_images"

    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"

    def load_images(self, directory: str, **kwargs):
        directory = folder_paths.get_annotated_filepath(directory.strip())
        return load_images(directory, **kwargs)
    
    @classmethod
    def IS_CHANGED(s, directory: str, **kwargs):
        directory = folder_paths.get_annotated_filepath(directory.strip())
        return is_changed_load_images(directory, **kwargs)

    @classmethod
    def VALIDATE_INPUTS(s, directory: str, **kwargs):
        directory = folder_paths.get_annotated_filepath(directory.strip())
        return validate_load_images(directory)


class LoadImagesFromDirectoryPath:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": "X://path/to/images", "vhs_path_extensions": []}),
            },
            "optional": {
                "image_load_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                "skip_first_images": ("INT", {"default": 0, "min": 0, "step": 1}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "INT")
    FUNCTION = "load_images"

    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"

    def load_images(self, directory: str, **kwargs):
        if directory is None or validate_load_images(directory) != True:
            raise Exception("directory is not valid: " + directory)

        return load_images(directory, **kwargs)
    
    @classmethod
    def IS_CHANGED(s, directory: str, **kwargs):
        if directory is None:
            return "input"
        return is_changed_load_images(directory, **kwargs)

    @classmethod
    def VALIDATE_INPUTS(s, directory: str, **kwargs):
        if directory is None:
            return True
        return validate_load_images(directory)
