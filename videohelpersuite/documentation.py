def image(src):
    return f'<img src={src} style="width: 0px; min-width: 100%">'
def video(src):
    return f'<video src={src} autoplay muted loop controls controlslist="nodownload noremoteplayback noplaybackrate" style="width: 0px; min-width: 100%" class="VHS_loopedvideo">'
def short_desc(desc):
    return  f'<div id=VHS_shortdesc style="font-size: .8em">Combine an image sequence into a video</div>'
    return  f'<div id=VHS_shortdesc style="font-size: .8em">{desc}</div>'

common_descriptions = {
  'merge_strategy': [
      'Determines what the output resolution will be if input resolutions don\'t match',
      {'match A': 'Always use the resolution for A',
      'match B': 'Always use the resolution  for B',
      'match smaller': 'Pick the smaller resolution by area',
      'match larger': 'Pick the larger resolution by area',
      }],
  'scale_method': [
    'Determines what method to use if scaling is required',
    {'nearest-exact': 'Each pixel of the ouput is set to the closest matching pixel of the unscaled input. No blending is performed',
    'bilinear': ''
      }],
  'crop_method': ''
}

descriptions = {
  'VHS_VideoCombine': ['Video Combine', short_desc('Combine an image sequence into a video'), {
    'Inputs': {
        'images': 'The images to be turned into a video',
        'audio':'(optional) audio to add to the video',
        'meta_batch': '(optional) Connect to a Meta Batch manager to divide extremely long image sequences into sub batches. See the documentation for Meta Batch Manager',
        'vae':['(optional) If provided, the node will take latents as input instead of images. This drastically reduces the required RAM (not VRAM) when working with long (100+ frames) sequences'
               "Unlike on Load Video, this isn't always a strict upgrade over using a standalone VAE Decode.",
               "If you have multiple Video Combine outputs, then the VAE decode will be performed for each output node increasing execution time",
               "If you make any change to output settings on the Video Combine (such as changing the output format), the VAE decode will be performed again as the decoded result is (by design) not cached"
               ]
        },
    'Widgets':{
        'frame_rate': 'The frame rate  which will be used for the output video. Consider converting this to an input and connecting this to a load Video with Video Info(Loaded)->fps. When including audio, failure to properly set this will result in audio desync',
        'loop_count': 'The number of additional times the video should repeat.',
        'filename_prefix': 'A prefix to add to the name of the output filename. This can include subfolders or format strings.',
        'format': 'The output format to use. Formats starting with, \'image\' are saved with PIL, but formats starting with \'video\' utilize the video_formats system. \'video\' options require ffmpeg and selecting one frequently adds additional options to the node.',
        'pingpong': 'Play the video normally, then repeat the video in reverse so that it \'pingpongs\' back and forth. This is frequently used to minimize the appearance of skips on very short animations.',
        'save_output': 'Specifies if output files should be saved to the output folder, or the temporary output folder'
        },
    'Common Format Widgets': {
        'crf': 'Determines how much to prioritize quality over filesize. Numbers vary between formats, but on each format that includes it, the default value provides visually loss less output',
        'pix_fmt': ['The pixel format to use for output. Alternative options will often have higher quality at the cost of increased file size and  reduced compatibility with external software.', {
            'yuv420p': 'The most common and default format',
            'yuv420p10le': 'Use 10 bit color depth. This can improve color quality when combined with 16bit input color depth',
            'yuva420p': 'Include transparency in the output video'
            }],
        'input_color_depth': 'VHS supports outputting 16bit images. While this produces higher quality output, the difference usually isn\'t visible without postprocessing and it significantly increases file size and processing time.',
        'save_metadata': 'Determines if metadata for the workflow should be included in the output video file'
        }
    }],
  'VHS_LoadVideo': ['Load Video', short_desc('Loads a video from the input folder'),
    {'Inputs': {
        'meta_batch': '(optional) Connect to a Meta Batch manager to divide extremely long sequences into sub batches. See the documentation for Meta Batch Manager',
        'vae': ['(optional) If provided the node will output latents instead of images. This drastically reduces the required RAM (not VRAM) when working with long (100+ frames) sequences',
                'Using this is strongly encouraged unless connecting to a node that requires a blue image connection such as Apply Controllnet',
                ],
        },
     'Outputs': {
         'IMAGE': 'The loaded images',
         'frame_count': 'The length of images just returned',
         'audio': 'The audio from the loaded video',
         'video_info': 'Exposes additional info about the video such as the source frame rate, or the total length',
         'LATENT': 'The loaded images converted pre-converted to latents. Only available when a vae is connected',
         },
     'Widgets': {
         'video': 'The video file to be loaded. Lists all files with a video extension in the ComfyUI/Input folder',
         'force_rate': 'Drops or duplicates frames so that the produced output has the target frame rate. Many motion models are trained on videos  of a specific frame rate and will give better results if input matches that frame rate. If set to 0, all frames are returned. May give unusual results with inputs that have a variable frame rate like animated gifs. Reducing this value can also greatly reduce the execution time and memory requirements.',
         'force_size': ['Allows for conveniently scaling the input without requiring an additional node. Provides options to maintain aspect ratio or conveniently target common training formats for Animate Diff', {'custom_width': 'Allows for an arbitrary width to be entered, cropping to maintain aspect ratio if possible',
               'custom_height': 'Allows for an arbitrary height to be entered, cropping to maintain aspect ratio if possible'}],
         'frame_load_cap': 'The maximum number of frames to load. If 0, all frames are loaded.',
         'skip_first_frames': 'A number of frames which are discarded before producing output.',
         'select_every_nth': 'Similar to frame rate. Keeps only the first of every n frames and discards the rest. Has better compatibility with variable frame rate inputs such as gifs. When combined with force_rate, select_every_nth_applies after force_rate so the resulting output has a frame rate equivalent to force_rate/select_every_nth. select_every_nth does not apply to skip_first_frames',
         'choose video to upload': 'An upload button is provided to upload local files to the input folder',
         'videopreview': 'Displays a preview for the selected video input. If advanced previews is enabled, this preview will reflect the frame_load_cap, force_rate, skip_first_frames, and select_every_nth values chosen. If the video has audio, it will also be previewed when moused over. Additional preview options can be accessed with right click.',
         }
        }],
  'VHS_LoadVideoPath': ['Load Video', short_desc('Loads a video from an arbitrary path'),
    {'Inputs': {
        'meta_batch': '(optional) Connect to a Meta Batch manager to divide extremely long sequences into sub batches. See the documentation for Meta Batch Manager',
        'vae': ['(optional) If provided the node will output latents instead of images. This drastically reduces the required RAM (not VRAM) when working with long (100+ frames) sequences',
                'Using this is strongly encouraged unless connecting to a node that requires a blue image connection such as Apply Controllnet',
                ],
        },
     'Outputs': {
         'IMAGE': 'The loaded images',
         'frame_count': 'The length of images just returned',
         'audio': 'The audio from the loaded video',
         'video_info': 'Exposes additional info about the video such as the source frame rate, or the total length',
         'LATENT': 'The loaded images converted pre-converted to latents. Only available when a vae is connected',
         },
     'Widgets': {
         'video': 'The video file to be loaded. When edited, provides a list of suggested files for completion',
         'force_rate': 'Drops or duplicates frames so that the produced output has the target frame rate. Many motion models are trained on videos  of a specific frame rate and will give better results if input matches that frame rate. If set to 0, all frames are returned. May give unusual results with inputs that have a variable frame rate like animated gifs. Reducing this value can also greatly reduce the execution time and memory requirements.',
         'force_size': ['Allows for conveniently scaling the input without requiring an additional node. Provides options to maintain aspect ratio or conveniently target common training formats for Animate Diff', {'custom_width': 'Allows for an arbitrary width to be entered, cropping to maintain aspect ratio if possible',
               'custom_height': 'Allows for an arbitrary height to be entered, cropping to maintain aspect ratio if possible'}],
         'frame_load_cap': 'The maximum number of frames to load. If 0, all frames are loaded.',
         'skip_first_frames': 'A number of frames which are discarded before producing output.',
         'select_every_nth': 'Similar to frame rate. Keeps only the first of every n frames and discards the rest. Has better compatibility with variable frame rate inputs such as gifs. When combined with force_rate, select_every_nth_applies after force_rate so the resulting output has a frame rate equivalent to force_rate/select_every_nth. select_every_nth does not apply to skip_first_frames',
         'videopreview': 'Displays a preview for the selected video input. Will only be shown if Advanced Previews is enabled. This preview will reflect the frame_load_cap, force_rate, skip_first_frames, and select_every_nth values chosen. If the video has audio, it will also be previewed when moused over. Additional preview options can be accessed with right click.',
         }
        }],
  'VHS_LoadImages': ['Load Images', short_desc('Loads a sequence of images from a subdirectory of the input folder'),
    {'Inputs': {
        'meta_batch': '(optional) Connect to a Meta Batch manager to divide extremely long sequences into sub batches. See the documentation for Meta Batch Manager',
        },
     'Outputs': {
         'IMAGE': 'The loaded images',
         'MASK': 'The alpha channel of the loaded images.',
         'frame_count': 'The length of images just returned',
         },
     'Widgets': {
         'directory': 'The directory images will be loaded from. Filtered to process jpg, png, ppm, bmp, tif, and webp files',
         'image_load_cap': 'The maximum number of images to load. If 0, all images are loaded.',
         'skip_first_images': 'A number of images which are discarded before producing output.',
         'select_every_nth': 'Keeps only the first of every n frames and discards the rest.',
         'choose folder to upload': 'An upload button is provided to upload a local folder containing images to the input folder',
         'videopreview': 'Displays a preview for the selected video input. Will only be shown if Advanced Previews is enabled. This preview will reflect the image_load_cap, skip_first_images, and select_every_nth values chosen. Additional preview options can be accessed with right click.',
         }
        }],
  'VHS_LoadImagesPath': ['Load Images', short_desc('Loads a sequence of images from a subdirectory of the input folder'),
    {'Inputs': {
        'meta_batch': '(optional) Connect to a Meta Batch manager to divide extremely long sequences into sub batches. See the documentation for Meta Batch Manager',
        },
     'Outputs': {
         'IMAGE': 'The loaded images',
         'MASK': 'The alpha channel of the loaded images.',
         'frame_count': 'The length of images just returned',
         },
     'Widgets': {
         'directory': 'The directory images will be loaded from. When edited, provides a list of suggested folders for completion. Filtered to process jpg, png, ppm, bmp, tif, and webp files',
         'image_load_cap': 'The maximum number of images to load. If 0, all images are loaded.',
         'skip_first_images': 'A number of images which are discarded before producing output.',
         'select_every_nth': 'Keeps only the first of every n frames and discards the rest.',
         'videopreview': 'Displays a preview for the selected video input. Will only be shown if Advanced Previews is enabled. This preview will reflect the image_load_cap, skip_first_images, and select_every_nth values chosen. Additional preview options can be accessed with right click.',
         }
        }],
  "VHS_LoadAudio": ['Load Audio (Path)', short_desc('Loads an audio file from an arbitrary path'),
    {'Outputs': {
         'audio': 'The loaded audio',
         },
     'Widgets': {
         'audio_file': 'The audio file to be loaded. When edited, provides a list of suggested files for completion',
         'seek_seconds': 'An offset from the start of the sound file that the audio should start from',
         }
        }],
  "VHS_LoadAudioUpload": ['Load Audio (Path)', short_desc('Loads an audio file from an arbitrary path'),
    "Very similar in functionality to the built-in LoadAudio. It was originally added before VHS swapped to use Comfy's internal AUDIO format, but provides the additional options for start time and duration",
    {'Outputs': {
         'audio': 'The loaded audio',
         },
     'Widgets': {
         'audio': 'The audio file to be loaded.',
         'start_time': 'An offset from the start of the sound file that the audio should start from',
         'duration': 'A maximum limit for the audio. Disabled if 0',
         'choose audio to upload': 'An upload button is provided to upload an audio file to the input folder',
         }
        }],
  "VHS_AudioToVHSAudio": ['Audio to legacy VHS_AUDIO', short_desc('utility function for compatibility with external nodes'),
    "VHS used to use an internal VHS_AUDIO format for routing audio between inputs and outputs. This format was intended to only be used internally and was designed with a focus on performance over ease of use. Since ComfyUI now has an internal AUDIO format, VHS now uses this format. However, some custom node packs were made that are external to both ComfyUI and VHS that use VHS_AUDIO. This node was added so that those external nodes can still function",
    {'Inputs': {
        'audio': 'An input in the standardized AUDIO format',
        },
     'Outputs': {
         'vhs_audio': 'An output in the legacy VHS_AUDIO format for use with external nodes',
         },
        }],
  "VHS_VHSAudioToAudio": ['Legacy VHS_AUDIO to Audio', short_desc('utility function for compatibility with external nodes'),
    "VHS used to use an internal VHS_AUDIO format for routing audio between inputs and outputs. This format was intended to only be used internally and was designed with a focus on performance over ease of use. Since ComfyUI now has an internal AUDIO format, VHS now uses this format. However, some custom node packs were made that are external to both ComfyUI and VHS that use VHS_AUDIO. This node was added so that those external nodes can still function",
    {'Inputs': {
        'vhs_audio': 'An input in the legacy VHS_AUDIO format produced by an external node',
        },
     'Outputs': {
         'vhs_audio': 'An output in the standardized AUDIO format',
         },
        }],
  "VHS_PruneOutputs": ['Prune Outputs', short_desc('Automates deletion of undesired outputs from a Video Combine node.'),
    'Video Combine produces a number of file outputs in addition to the final output. Some of these, such as a video file without audio included, are implementation limitations and are not feasible to solve. As an alternative, the Prune Outputs node is added to automate the deletion of these undesired file outputs if they are not desired',
    {'Inputs': {
        'filenames': 'A connection from a Video Combine node to indicate which outputs should be pruned',
        },
     'Widgets': {
         'options': ['Which files should be deleted',
             {'Intermediate': 'Delete any files that were required for intermediate processing but are not that final output, like the no-audio output file when audio is included',
              'Intermediate and Utility': 'Delete all produced files that aren\' the final output, including the first frame png'
         }]}
     }],
  "VHS_BatchManager": ['Meta Batch Manager', short_desc('Split the processing of a very long video into sets of smaller Meta Batches'),
    "The Meta Batch Manager allows for extremely long input videos to be processed when all other methods for fitting the content in RAM fail. It makes no difference on VRAM usage.",
    "It must be connected to at least one Input (a Load Video or Load Images) AND at least one Video Combine",
    "It functions by holding both the inputs and ouputs open between executions, and automatically requeue's the workflow until one of the inputs is unable to provide additional images.",
    "Because each sub execution only contains a subset of the total frames, each sub execution creates a hard window which temporal smoothing can not be applied across, which results in jumps in the output.",
    {'Outputs': {
         'meta_batch': 'Add all connected nodes to this Meta Batch',
         },
     'Widgets': {
         'frames_per_batch': 'How many frames to process for each sub execution. If loading as image, each frame will use about 50MB of RAM (not VRAM), and this can safely be set in the 100-1000 range, depending on available memory. When loading and combine from latent space (no blue image noodles exist), this value can be much higher, around the 2,000 to 20,000 range',
         }
        }],
  "VHS_VideoInfo": ['Video Info', short_desc('Splits information on a video into a numerous outputs'),
    {'Inputs': {
        'video_info': 'A connection to a Load Video node',
        },
     'Outputs': {
         'source_fps🟨': 'The frame rate of the video',
         'source_frame_count🟨': 'How many total frames the video contains before accounting for frame rate or select_every_nth',
         'source_duration🟨': 'The length of images just returned in seconds',
         'source_width🟨': 'The width',
         'source_height🟨': 'The height',
         'loaded_fps🟦': 'The frame rate after accounting for force_rate and select_every_nth. This output is of particular use as it can be connected to the converted frame_rate input of a Video Combine node to ensure audio remains synchronized.',
         'loaded_frame_count🟦': 'The number of frames returned by the current execution. Identical to the frame_count returned by the node itself',
         'loaded_duration🟦': 'The duration in seconds of returned images after accounting for frame_load_cap',
         'loaded_width🟦': 'The width of the video after scaling. These coordinates are in image space even if loading to latent space',
         'loaded_height🟦': 'The height of the video after scaling. These coordinates are in image space even if loading to latent space',
         },
        }],
  "VHS_VideoInfoSource": ['Video Info Source', short_desc('Splits information on a video into a numerous outputs describing the file itself without accounting for load options'),
    {'Inputs': {
        'video_info': 'A connection to a Load Video node',
        },
     'Outputs': {
         'source_fps🟨': 'The frame rate of the video',
         'source_frame_count🟨': 'How many total frames the video contains before accounting for frame rate or select_every_nth',
         'source_duration🟨': 'The length of images just returned in seconds',
         'source_width🟨': 'The original width',
         'source_height🟨': 'The original height',
         }
     }],
  "VHS_VideoInfoLoaded": ['Video Info Loaded', short_desc('Splits information on a video into a numerous outputs describing the file itself after accounting for load options'),
    {'Inputs': {
        'video_info': 'A connection to a Load Video node',
        },
     'Outputs': {
         'loaded_fps🟦': 'The frame rate after accounting for force_rate and select_every_nth. This output is of particular use as it can be connected to the converted frame_rate input of a Video Combine node to ensure audio remains synchronized.',
         'loaded_frame_count🟦': 'The number of frames returned by the current execution. Identical to the frame_count returned by the node itself',
         'loaded_duration🟦': 'The duration in seconds of returned images after accounting for frame_load_cap',
         'loaded_width🟦': 'The width of the video after scaling. This is the dimension of the corresponding image even if loading as a latent directly',
         'loaded_height🟦': 'The height of the video after scaling. This is the dimension of the corresponding image even if loading as a latent directly',
         }
     }],
    # Batched Nodes
  "VHS_VAEEncodeBatched": ['VAE Encode Batched', short_desc('Encode images as latents with a manually specified batch size.'),
    "Some people have ran into VRAM issues when encoding or decoding large batches of images. As a workaround, this node lets you manually set a batch size when encoding images.",
    "Unless these issues have been encountered, it is simpler to use the native VAE Encode or to encode directly from a Load Video",
    {'Inputs': {
        'pixels': 'The images to be encoded.',
        'vae': 'The VAE to use when encoding.',
        },
     'Outputs': {
         'LATENT': 'The encoded latents.',
        },
     'Widgets': {
         'per_batch': 'The maximum number of images to encode in each batch.',
        },
     }],
  "VHS_VAEDecodeBatched": ['VAE Decode Batched', short_desc('Decode latents to images with a manually specified batch size'),
    "Some people have ran into VRAM issues when encoding or decoding large batches of images. As a workaround, this node lets you manually set a batch size when decoding latents.",
    "Unless these issues have been encountered, it is simpler to use the native VAE Decode or to decode from a Video Combine directly",
    {'Inputs': {
        'samples': 'The latents to be decoded.',
        'vae': 'The VAE to use when decoding.',
        },
     'Outputs': {
         'IMAGE': 'The decoded images.',
        },
     'Widgets': {
         'per_batch': 'The maximum number of images to decode in each batch.',
        },
     }],
    # Latent and Image nodes
  "VHS_SplitLatents": ['Split Latents', short_desc('Split a set of latents into two groups'),
    {'Inputs': {
        'latents': 'The latents to be split.',
        },
     'Outputs': {
         'LATENT_A': 'The first group of latents',
         'A_count': 'The number of latents in group A. This will be equal to split_index unless the latents input has length less than split_index',
         'LATENT_B': 'The second group of latents',
         'B_count': 'The number of latents in group B'
        },
     'Widgets': {
        'split_index': 'The index of the first latent that will be in the second output groups.',
        },

    }],
    "VHS_SplitImages": ['Split Images', short_desc('Split a set of images into two groups'),
    {'Inputs': {
        'images': 'The images to be split.',
        },
     'Outputs': {
         'IMAGE_A': 'The first group of images',
         'A_count': 'The number of images in group A. This will be equal to split_index unless the images input has length less than split_index',
         'IMAGE_B': 'The second group of images',
         'B_count': 'The number of images in group B'
        },
     'Widgets': {
        'split_index': 'The index of the first latent that will be in the second output groups.',
        },

    }],
    "VHS_SplitMasks": None,
    "VHS_SplitImages": ['Split Masks', short_desc('Split a set of masks into two groups'),
    {'Inputs': {
        'mask': 'The masks to be split.',
        },
     'Outputs': {
         'MASK_A': 'The first group of masks',
         'A_count': 'The number of masks in group A. This will be equal to split_index unless the mask input has length less than split_index',
         'MASK_B': 'The second group of masks',
         'B_count': 'The number of masks in group B'
        },
     'Widgets': {
        'split_index': 'The index of the first latent that will be in the second output groups.',
        },

    }],
    "VHS_MergeLatents": ['Merge Latents', short_desc('Combine two groups of latents into a single group of latents'),
    {'Inputs': {
        'latents_A': 'The first group of latents',
        'latents_B': 'The first group of latents',
        },
     'Outputs': {
         'LATENT': 'The combined group of latents',
         'count': 'The legnth of the combined group',
        },
     'Widgets': {
        'merge_strategy': common_descriptions['merge_strategy'],
        'scale_method': common_descriptions['scale_method'],
        'crop': common_descriptions['crop_method'],
        },

    }],
    "VHS_MergeImages": None,
    "VHS_MergeMasks": None,
    "VHS_GetLatentCount": None,
    "VHS_GetImageCount": None,
    "VHS_GetMaskCount": None,
    "VHS_DuplicateLatents": None,
    "VHS_DuplicateImages": None,
    "VHS_DuplicateMasks": None,
    "VHS_SelectEveryNthLatent": None,
    "VHS_SelectEveryNthImage": None,
    "VHS_SelectEveryNthMask": None,
    "VHS_SelectLatents": None,
    "VHS_SelectImages": None,
    "VHS_SelectMasks": None,
}

sizes = ['1.4','1.2','1']
def as_html(entry, depth=0):
    if isinstance(entry, dict):
        size = 0.8 if depth < 2 else 1
        html = ''
        for k in entry:
            if k == "collapsed":
                continue
            collapse_single = k.endswith("_collapsed")
            if collapse_single:
                name = k[:-len("_collapsed")]
            else:
                name = k
            collapse_flag = ' VHS_precollapse' if entry.get("collapsed", False) or collapse_single else ''
            html += f'<div vhs_title=\"{name}\" style=\"display: flex; font-size: {size}em\" class=\"VHS_collapse{collapse_flag}\"><div style=\"color: #AAA; height: 1.5em; width: 1.5em; min-width: 1.5em; max-width: 1.5em\">[-]</div><div style=\"width: 100%\">{name}: {as_html(entry[k], depth=depth+1)}</div></div>'
        return html
    if isinstance(entry, list):
        html = ''
        for i in entry:
            html += f'<div>{as_html(i, depth=depth)}</div>'
        return html
    return str(entry)

def format_descriptions(nodes):
    for k in descriptions:
        if k.endswith("_collapsed"):
            k = k[:-len("_collapsed")]
        nodes[k].DESCRIPTION = as_html(descriptions[k])
    return
    for k in nodes:
        assert hasattr(nodes[k], "DESCRIPTION"), f'{k} does not have a description'
