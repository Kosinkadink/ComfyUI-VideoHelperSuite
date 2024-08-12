def image(src):
    return f'<img src={src} style="width: 0px; min-width: 100%">'
descriptions = {
  'VHS_VideoCombine': ['Video Combine', '<div id=VHS_shortdesc style="font-size: .8em">Combine an image sequence into a video</div>', {
    'Inputs': {'images': 'The images to be turned into a video','audio':'(optional) audio to add to the video','meta_batch': '(optional) Connect to a Meta Batch manager to divide extremely long image sequences into sub batches. See the documentation for Meta Batch Manager','vae':'(optional) If provided, the node will take latents as input instead of images. This drastically reduces the required RAM (not VRAM) required when working with very long sequences'},
    'Widgets':{'frame_rate_collapsed': 'The frame rate  which will be used for the output video. Consider converting this to an input and connecting this  to a load Video with Video Info(Loaded)->fps. When including audio, failure to properly set this will result in audio desync', 'loop_count': 'The number of additional times the video should repeat.', 'filename_prefix': 'A prefix to add to the name of the output filename. This can include subfolders or format strings.', 'format': 'The output format to use. Formats starting with, \'image\' are saved with PIL, but formats starting with \'video\' utilize the video_formats system. \'video\' options require ffmpeg and selecting one frequently adds additional options to the node.', 'pingpong': 'Play the video normally, then repeat the video in reverse so that it \'pingpongs\' back and forth. This is frequently used to minimize the appearance of skips on very short animations.', 'save_output': 'Specifies if output files should be saved to the output folder, or the temporary output folder'},
    'Common Format Widgets': {'crf': 'Determines how much to prioritize quality over filesize. Numbers vary between formats, but on each format that includes it, the default value provides visually loss less output', 'pix_fmt': ['The pixel format to use for output. Alternative options will often have higher quality at the cost of increased file size and  reduced compatibility with external software.', {'yuv420p': 'The most common and default format', 'yuv420p10le': 'Use 10 bit color depth. This can improve color quality when combined with 16bit input color depth', 'yuva420p': 'Include transparency in the output video', 'collapsed': True}], 'input_color_depth': 'VHS supports outputting 16bit images. While this produces higher quality output, the difference usually isn\'t visible without postprocessing and it significantly increases file size and processing time.', 'save_metadata': 'Determines if metadata for the workflow should be included in the output video file'}}]
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
            html += f'<div vhs_title={name} style=\"display: flex; font-size: {size}em\" class=\"VHS_collapse{collapse_flag}\"><div style=\"color: #AAA; height: 1.5em; width: 1.5em\">[-]</div><div style=\"width: 100%\">{name}: {as_html(entry[k], depth=depth+1)}</div></div>'
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

