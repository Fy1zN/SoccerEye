import cv2
import os


def read_video(video_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frames = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frames.append(frame)

    cap.release()

    return frames


def save_video(output_video_frames, output_video_path, fps=24):
    if len(output_video_frames) == 0:
        raise ValueError("No frames provided")

    height, width = output_video_frames[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out = cv2.VideoWriter(
        output_video_path,
        fourcc,
        fps,
        (width, height)
    )

    for frame in output_video_frames:
        out.write(frame)

    out.release()