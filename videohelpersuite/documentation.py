descriptions = {
        'VHS_VideoCombine': {'Inputs': {'images': 'The images to be turned into a video','audio':'(optional) audio to add to the video','meta_batch': '(optional) Connect to a Meta Batch manager to divide extremely long image sequences into sub batches. See the documentation for Meta Batch Manager','vae':'(optional) If provided, the node will take latents as input instead of images. This drastically reduces the required RAM (not VRAM) required when working with very long sequences'}, 'Widgets':{'frame_rate': 'The frame rate  which will be used for the output video. Consider converting this to an input and connecting this  to a load Video with Video Info(Loaded)->fps. When including audio, failure to properly set this will result in audio desync', 'loop_count': 'The number of additional times the video should repeat.', 'filename_prefix': 'A prefix to add to the name of the output filename. This can include subfolders or format strings.', 'format': 'The output format to use. Formats starting with, \'image\' are saved with PIL, but formats starting with \'video\' utilize the video_formats system. \'video\' options require ffmpeg and selecting one frequently adds additional options to the node.', 'pingpong': 'Play the video normally, then repeat the video in reverse so that it \'pingpongs\' back and forth. This is frequently used to minimize the appearance of skips on very short animations.', 'save_output': 'Specifies if output files should be saved to the output folder, or the temporary output folder'}, 'Common Format Widgets': {'crf': 'Determines how much to prioritize quality over filesize. Numbers vary between formats, but on each format that includes it, the default value provides visually loss less output', 'pix_fmt': 'The pixel format to use for output. Alternative options will often have higher bit depth, or support alpha output, but may have reduced compatibility with external software.', 'input_color_depth': 'VHS supports outputting 16bit images. While this produces higher quality output, the difference usually isn\'t visible without postprocessing and it significantly increases file size and processing time.', 'save_metadata': 'Determines if metadata for the workflow should be included in the output video file'}}
}

sizes = ['1.4','1.2','1']
def as_html(entry, depth=0):
    if isinstance(entry, dict):
        size = 0.8
        html = ''
        for k in entry:
            #html += f'<{tag} title={k}>{k}: {as_html(entry[k], depth=depth+1)}</{tag}>'
            html += f'<div title={k} style=\"display: flex; font-size: {size}em\" class=\"VHS_collapse\"><div style=\"color: #AAA; height: 1.5em\">[-]</div><div>{k}: {as_html(entry[k], depth=depth+1)}</div></div>'
        return html
    if isinstance(entry, list):
        html = ''
        for i in entry:
            html += f'<div style=\"font-size: {size}em\">{as_html(i, depth=depth+1)}</div>'
        return html
    return str(entry)

def format_descriptions(nodes):
    for k in descriptions:
        nodes[k].DESCRIPTION = as_html(descriptions[k])

