import youtube_dl
import time
import json
import os
import numpy as np
import subprocess
from youtube_dl import main
def getdata():
    video_dl = youtube_dl.YoutubeDL({'outtmpl': 'v_%(id)s.%(ext)s'})
    video_dir = '/home/username/datasets/finegym'
    os.makedirs(video_dir, exist_ok=True)
    os.chdir(video_dir)
    excit_video=set()
    for root, dirs, files in os.walk(video_dir): 
        for file in files:
            if file.split(".")[-1] != "part" and file.split(".")[-1] in ["mp4", "mkv", "webm"]:
                excit_video.add(file.split(".")[0])
    print(sorted(excit_video), len(excit_video))

    with open(os.path.join(video_dir, 'finegym_annotation_info_v1.0.json'), 'r') as f:
        lines=json.load(f)
    video_ids=[]
    for line in lines:
        if line not in excit_video and line[0].lower()>="e":
            video_ids.append(line)
    print(sorted(video_ids), len(video_ids))
    time.sleep(10)

    ydl_opts = {}
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        for i,video_id in enumerate(video_ids):
            cmd_base = "youtube-dl -f best -f mp4 "
            cmd_base += '"https://www.youtube.com/watch?v=%s" '
            cmd_base += '-o "%s/FineGym_Raw_database/%s.mp4"'
            os.system(cmd_base % (video_id, video_dir, video_id))
    os.chdir('..')

if __name__ == "__main__":
    getdata()
