import hashlib
import os
from typing import Iterable
import shutil
import subprocess

from .logger import logger

ffmpeg_path = shutil.which("ffmpeg")
if "VHS_USE_IMAGEIO_FFMPEG" in os.environ or ffmpeg_path is None:
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        ffmpeg_path = get_ffmpeg_exe()
    except:
        logger.warn("Failed to import imageio_ffmpeg")
if ffmpeg_path is None:
    logger.warning("ffmpeg could not be found. Outputs that require it have been disabled")

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
    h = hashlib.sha256()
    b = bytearray(10*1024*1024) # read 10 megabytes at a time
    mv = memoryview(b)
    with open(filename, 'rb', buffering=0) as f:
        i = 0
        # don't hash entire file, only portions of it if requested
        while n := f.readinto(mv):
            if i%hash_every_n == 0:
                h.update(mv[:n])
            i += 1
    return h.hexdigest()

def get_audio(file, start_time=0, duration=0):
    args = [ffmpeg_path, "-v", "error", "-i", file]
    if start_time > 0:
        args += ["-ss", str(start_time)]
    if duration > 0:
        args += ["-t", str(duration)]
    return subprocess.run(args + ["-f", "wav", "-"],
                          stdout=subprocess.PIPE, check=True).stdout
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
