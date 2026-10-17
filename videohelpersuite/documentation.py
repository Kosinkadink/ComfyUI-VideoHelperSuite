from .logger import logger

descriptions = {
  "VHS_VideoCombine": "Combine an image sequence into a video",
  "VHS_LoadVideo": "Loads a video from the input folder",
  "VHS_LoadVideoFFmpeg": "Loads a video from the input folder using ffmpeg instead of opencv",
  "VHS_LoadVideoPath": "Loads a video from an arbitrary path",
  "VHS_LoadVideoFFmpegPath": "Loads a video from an arbitrary path using ffmpeg instead of opencv",
  "VHS_LoadImages": "Loads a sequence of images from a subdirectory of the input folder",
  "VHS_LoadImagesPath": "Loads a sequence of images from an arbitrary path",
  "VHS_LoadImagePath": "Load a single image from a given path",
  "VHS_LoadAudio": "Loads an audio file from an arbitrary path",
  "VHS_LoadAudioUpload": "Loads an audio file from the input directory",
  "VHS_AudioToVHSAudio": "utility function for compatibility with external nodes",
  "VHS_VHSAudioToAudio": "utility function for compatibility with external nodes",
  "VHS_PruneOutputs": "Automates deletion of undesired outputs from a Video Combine node.",
  "VHS_BatchManager": "Split the processing of a very long video into sets of smaller Meta Batches",
  "VHS_VideoInfo": "Splits information on a video into a numerous outputs",
  "VHS_VideoInfoSource": "Splits information on a video into a numerous outputs describing the file itself without accounting for load options",
  "VHS_VideoInfoLoaded": "Splits information on a video into a numerous outputs describing the file itself after accounting for load options",
  "VHS_SelectFilename": "Select a single filename from the VHS_FILENAMES output by a Video Combine and return it as a string",
  "VHS_VAEEncodeBatched": "Encode images as latents with a manually specified batch size.",
  "VHS_VAEDecodeBatched": "Decode latents to images with a manually specified batch size",
  "VHS_SplitLatents": "Split a set of latents into two groups",
  "VHS_SplitImages": "Split a set of images into two groups",
  "VHS_SplitMasks": "Split a set of masks into two groups",
  "VHS_MergeLatents": "Combine two groups of latents into a single group of latents",
  "VHS_MergeImages": "Combine two groups of images into a single group of images",
  "VHS_MergeMasks": "Combine two groups of masks into a single group of masks",
  "VHS_GetLatentCount": "Return the number of latents in an input as an INT",
  "VHS_GetImageCount": "Return the number of images in an input as an INT",
  "VHS_GetMaskCount": "Return the number of masks in an input as an INT",
  "VHS_DuplicateLatents": "Append copies of a latent to itself so it repeats",
  "VHS_DuplicateImages": "Append copies of a image to itself so it repeats",
  "VHS_DuplicateMasks": "Append copies of a mask to itself so it repeats",
  "VHS_SelectEveryNthLatent": "Keep only 1 latent for every interval",
  "VHS_SelectEveryNthImage": "Keep only 1 image for every interval",
  "VHS_SelectEveryNthMask": "Keep only 1 mask for every interval",
  "VHS_Unbatch": "Unbatch a list of items into a single concatenated item",
  "VHS_SelectLatest": "Experimental virtual node to select the most recently modified file from a given folder",
}

def format_descriptions(nodes):
    for k in descriptions:
        nodes[k].DESCRIPTION = descriptions[k]
    undocumented_nodes = []
    for k in nodes:
        if not hasattr(nodes[k], "DESCRIPTION"):
            undocumented_nodes.append(k)
    if len(undocumented_nodes) > 0:
        logger.warning('Some nodes have not been documented %s', undocumented_nodes)
