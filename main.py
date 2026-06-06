from utils import read_video, save_video
from trackers.tracker import Tracker


def main():

    print("Starting SoccerEye...")

    # Load video
    frames = read_video(
        "Input_data/08fd33_4.mp4"
    )

    print(f"Loaded {len(frames)} frames")
    print(f"Frame shape: {frames[0].shape}")

    # Load tracker
    tracker = Tracker(
        "models/best.pt"
    )

    print("Tracker initialized")

    # Detect + Track
    tracks = tracker.get_object_tracks(
        frames
    )

    print("Tracking complete")

    # Draw annotations
    output_frames = tracker.draw_annotations(
        frames,
        tracks
    )

    print("Annotations complete")

    # Save output video
    save_video(
        output_frames,
        "output/tracked_video.avi"
    )

    print("Video saved to output/tracked_video.avi")
    print("Done")


if __name__ == "__main__":
    main()