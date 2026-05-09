import numpy as np
from decord import VideoReader, cpu
import os
import cv2


# video_path = r"HwnB8aCn8yE.mp4"
# N1cdUjctpG8

video_path = "0w4OTD4L0GQ.mp4"
for_get_frames_num = 32
force_sample = False

vr = VideoReader(video_path, ctx=cpu(0))
total_frame_num = len(vr)
fps = round(vr.get_avg_fps())
frame_idx = [i for i in range(0, len(vr), fps)]
# sample_fps = args.for_get_frames_num if total_frame_num > args.for_get_frames_num else total_frame_num
if len(frame_idx) > for_get_frames_num or force_sample:
    sample_fps = for_get_frames_num
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
spare_frames = vr.get_batch(frame_idx).asnumpy()
print(f'frame length: {len(spare_frames)}')

output_dir = os.path.basename(video_path).split('.')[0]
os.makedirs(f"{output_dir}", exist_ok=True)
for i, frame in enumerate(spare_frames):
    cv2.imwrite(f'{output_dir}/frame_{i}.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
