import hashlib
import os
from typing import Iterable
import shutil
import subprocess
import re

import server
from .logger import logger

BIGMIN = -(2**53-1)
BIGMAX = (2**53-1)

DIMMAX = 8192

def ffmpeg_suitability(path):
    try:
        version = subprocess.run([path, "-version"], check=True,
                                 capture_output=True).stdout.decode("utf-8")
    except:
        return 0
    score = 0
    #rough layout of the importance of various features
    simple_criterion = [("libvpx", 20),("264",10), ("265",3),
                        ("svtav1",5),("libopus", 1)]
    for criterion in simple_criterion:
        if version.find(criterion[0]) >= 0:
            score += criterion[1]
    #obtain rough compile year from copyright information
    copyright_index = version.find('2000-2')
    if copyright_index >= 0:
        copyright_year = version[copyright_index+6:copyright_index+9]
        if copyright_year.isnumeric():
            score += int(copyright_year)
    return score


if "VHS_FORCE_FFMPEG_PATH" in os.environ:
    ffmpeg_path = os.environ.get("VHS_FORCE_FFMPEG_PATH")
else:
    ffmpeg_paths = []
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        imageio_ffmpeg_path = get_ffmpeg_exe()
        ffmpeg_paths.append(imageio_ffmpeg_path)
    except:
        if "VHS_USE_IMAGEIO_FFMPEG" in os.environ:
            raise
        logger.warn("Failed to import imageio_ffmpeg")
    if "VHS_USE_IMAGEIO_FFMPEG" in os.environ:
        ffmpeg_path = imageio_ffmpeg_path
    else:
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg is not None:
            ffmpeg_paths.append(system_ffmpeg)
        if os.path.isfile("ffmpeg"):
            ffmpeg_paths.append(os.path.abspath("ffmpeg"))
        if os.path.isfile("ffmpeg.exe"):
            ffmpeg_paths.append(os.path.abspath("ffmpeg.exe"))
        if len(ffmpeg_paths) == 0:
            logger.error("No valid ffmpeg found.")
            ffmpeg_path = None
        elif len(ffmpeg_paths) == 1:
            #Evaluation of suitability isn't required, can take sole option
            #to reduce startup time
            ffmpeg_path = ffmpeg_paths[0]
        else:
            ffmpeg_path = max(ffmpeg_paths, key=ffmpeg_suitability)
gifski_path = os.environ.get("VHS_GIFSKI", None)
if gifski_path is None:
    gifski_path = os.environ.get("JOV_GIFSKI", None)
    if gifski_path is None:
        gifski_path = shutil.which("gifski")

def get_sorted_dir_files_from_directory(directory: str, skip_first_images: int=0, select_every_nth: int=1, extensions: Iterable=None):
    directory = directory.strip()
    dir_files = os.listdir(directory)
    dir_files = sorted(dir_files)
    dir_files = [os.path.join(directory, x) for x in dir_files]
    dir_files = list(filter(lambda filepath: os.path.isfile(filepath), dir_files))
    # filter by extension, if needed
    if extensions is not None:
        extensions = list(extensions)
        new_dir_files = []
        for filepath in dir_files:
            ext = "." + filepath.split(".")[-1]
            if ext.lower() in extensions:
                new_dir_files.append(filepath)
        dir_files = new_dir_files
    # start at skip_first_images
    dir_files = dir_files[skip_first_images:]
    dir_files = dir_files[0::select_every_nth]
    return dir_files


# modified from https://stackoverflow.com/questions/22058048/hashing-a-file-in-python
def calculate_file_hash(filename: str, hash_every_n: int = 1):
    #Larger video files were taking >.5 seconds to hash even when cached,
    #so instead the modified time from the filesystem is used as a hash
    h = hashlib.sha256()
    h.update(filename.encode())
    h.update(str(os.path.getmtime(filename)).encode())
    return h.hexdigest()

prompt_queue = server.PromptServer.instance.prompt_queue
def requeue_workflow_unchecked():
    """Requeues the current workflow without checking for multiple requeues"""
    currently_running = prompt_queue.currently_running
    (_, _, prompt, extra_data, outputs_to_execute) = next(iter(currently_running.values()))

    #Ensure batch_managers are marked stale
    prompt = prompt.copy()
    for uid in prompt:
        if prompt[uid]['class_type'] == 'VHS_BatchManager':
            prompt[uid]['inputs']['requeue'] = prompt[uid]['inputs'].get('requeue',0)+1

    #execution.py has guards for concurrency, but server doesn't.
    #TODO: Check that this won't be an issue
    number = -server.PromptServer.instance.number
    server.PromptServer.instance.number += 1
    prompt_id = str(server.uuid.uuid4())
    prompt_queue.put((number, prompt_id, prompt, extra_data, outputs_to_execute))

requeue_guard = [None, 0, 0, {}]
def requeue_workflow(requeue_required=(-1,True)):
    assert(len(prompt_queue.currently_running) == 1)
    global requeue_guard
    (run_number, _, prompt, _, _) = next(iter(prompt_queue.currently_running.values()))
    if requeue_guard[0] != run_number:
        #Calculate a count of how many outputs are managed by a batch manager
        managed_outputs=0
        for bm_uid in prompt:
            if prompt[bm_uid]['class_type'] == 'VHS_BatchManager':
                for output_uid in prompt:
                    if prompt[output_uid]['class_type'] in ["VHS_VideoCombine"]:
                        for inp in prompt[output_uid]['inputs'].values():
                            if inp == [bm_uid, 0]:
                                managed_outputs+=1
        requeue_guard = [run_number, 0, managed_outputs, {}]
    requeue_guard[1] = requeue_guard[1]+1
    requeue_guard[3][requeue_required[0]] = requeue_required[1]
    if requeue_guard[1] == requeue_guard[2] and max(requeue_guard[3].values()):
        requeue_workflow_unchecked()

def get_audio(file, start_time=0, duration=0):
    args = [ffmpeg_path, "-v", "error", "-i", file]
    if start_time > 0:
        args += ["-ss", str(start_time)]
    if duration > 0:
        args += ["-t", str(duration)]
    try:
        res =  subprocess.run(args + ["-f", "wav", "-"],
                              stdout=subprocess.PIPE, check=True).stdout
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to extract audio from: {file}")
        return False
    return res


def lazy_eval(func):
    class Cache:
        def __init__(self, func):
            self.res = None
            self.func = func
        def get(self):
            if self.res is None:
                self.res = self.func()
            return self.res
    cache = Cache(func)
    return lambda : cache.get()


def is_url(url):
    return url.split("://")[0] in ["http", "https"]

def validate_sequence(path):
    #Check if path is a valid ffmpeg sequence that points to at least one file
    (path, file) = os.path.split(path)
    if not os.path.isdir(path):
        return False
    match = re.search('%0?\d+d', file)
    if not match:
        return False
    seq = match.group()
    if seq == '%d':
        seq = '\\\\d+'
    else:
        seq = '\\\\d{%s}' % seq[1:-1]
    file_matcher = re.compile(re.sub('%0?\d+d', seq, file))
    for file in os.listdir(path):
        if file_matcher.fullmatch(file):
            return True
    return False

def hash_path(path):
    if path is None:
        return "input"
    if is_url(path):
        return "url"
    return calculate_file_hash(path.strip("\""))


def validate_path(path, allow_none=False, allow_url=True):
    if path is None:
        return allow_none
    if is_url(path):
        #Probably not feasible to check if url resolves here
        return True if allow_url else "URLs are unsupported for this path"
    if not os.path.isfile(path.strip("\"")):
        return "Invalid file path: {}".format(path)
    return True
