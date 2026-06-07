from utils import read_video, save_video
from trackers.tracker import Tracker
from team_assigner import TeamAssigner

import cv2
import os


def main():

    print("Starting SoccerEye...")

    # Load video
    frames = read_video(
        "Input_data/08fd33_4.mp4"
    )

    print(f"Loaded {len(frames)} frames")
    print(f"Frame shape: {frames[0].shape}")

    # Create stubs folder
    os.makedirs("stubs", exist_ok=True)

    # Initialize tracker
    tracker = Tracker(
        "models/best.pt"
    )

    print("Tracker initialized")

    # Detect + Track
    tracks = tracker.get_object_tracks(
        frames,
        read_from_stub=True,      # False first run, True afterwards
        stub_path="stubs/track_stubs.pkl"
    )

    print("Tracking complete")

    # --------------------------------------------------
    # Save sample player crop
    # --------------------------------------------------

    for track_id, player in tracks["players"][0].items():

        bbox = player["bbox"]

        frame = frames[0]

        cropped_image = frame[
            int(bbox[1]):int(bbox[3]),
            int(bbox[0]):int(bbox[2])
        ]

        cv2.imwrite(
            "output/player_crop.jpg",
            cropped_image
        )

        print(
            f"Saved player crop: output/player_crop.jpg "
            f"(Player ID: {track_id})"
        )

        break

    # --------------------------------------------------
    # Team Assignment
    # --------------------------------------------------

    team_assigner = TeamAssigner()

    team_assigner.assign_team_color(
        frames[0],
        tracks["players"][0]
    )

    for frame_num, player_track in enumerate(
        tracks["players"]
    ):

        for player_id, track in player_track.items():

            team = team_assigner.get_player_team(
                frames[frame_num],
                track["bbox"],
                player_id
            )

            tracks["players"][frame_num][
                player_id
            ]["team"] = team

            tracks["players"][frame_num][
                player_id
            ]["team_color"] = (
                team_assigner.team_colors[team]
            )

    print("Teams assigned")

    # --------------------------------------------------
    # Draw annotations
    # --------------------------------------------------

    output_frames = tracker.draw_annotations(
        frames,
        tracks
    )

    print("Annotations complete")

    # --------------------------------------------------
    # Save output video
    # --------------------------------------------------

    save_video(
        output_frames,
        "output/tracked_video.avi"
    )

    print("Video saved to output/tracked_video.avi")
    print("Done")


if __name__ == "__main__":
    main()