from utils import read_video, save_video
from trackers.tracker import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import (
    CameraMovementEstimator
)
from view_transformer import (
    ViewTransformer
)
from speed_and_distance_estimator import (
    SpeedAndDistance_Estimator
)

import cv2
import os
import numpy as np


def main():

    print("Starting SoccerEye...")

    # Load video
    frames = read_video(
        "Input_data/08fd33_4.mp4"
    )

    print(f"Loaded {len(frames)} frames")
    print(f"Frame shape: {frames[0].shape}")

    os.makedirs(
        "stubs",
        exist_ok=True
    )

    # --------------------------------------------------
    # INITIALIZE TRACKER
    # --------------------------------------------------

    tracker = Tracker(
        "models/best.pt"
    )

    print("Tracker initialized")

    # --------------------------------------------------
    # DETECT + TRACK
    # --------------------------------------------------

    tracks = tracker.get_object_tracks(
        frames,
        read_from_stub=True,
        stub_path="stubs/track_stubs.pkl"
    )

    print("Tracking complete")

    # --------------------------------------------------
    # INTERPOLATE BALL POSITIONS
    # --------------------------------------------------

    tracks["ball"] = (
        tracker.interpolate_ball_positions(
            tracks["ball"]
        )
    )

    print(
        "Ball interpolation complete"
    )

    # --------------------------------------------------
    # ADD POSITIONS TO TRACKS
    # --------------------------------------------------

    tracker.add_position_to_tracks(
        tracks
    )

    print(
        "Object positions added"
    )

    # --------------------------------------------------
    # CAMERA MOVEMENT ESTIMATION
    # --------------------------------------------------

    camera_movement_estimator = (
        CameraMovementEstimator(
            frames[0]
        )
    )

    camera_movement_per_frame = (
        camera_movement_estimator
        .get_camera_movement(
            frames,
            read_from_stub=False,
            stub_path=
            "stubs/camera_movement_stub.pkl"
        )
    )

    camera_movement_estimator\
        .add_adjust_positions_to_tracks(
            tracks,
            camera_movement_per_frame
        )

    print(
        "Camera movement estimated"
    )

    # --------------------------------------------------
    # VIEW TRANSFORMER
    # --------------------------------------------------

    view_transformer = (
        ViewTransformer()
    )

    view_transformer\
        .add_transformed_position_to_tracks(
            tracks
        )

    print(
        "Perspective transformation complete"
    )

    # --- Change 3: Add Debug Counter ---
    count_total = 0
    count_transformed = 0

    for frame_tracks in tracks["players"]:
        for _, player in frame_tracks.items():
            count_total += 1

            if player.get("position_transformed") is not None:
                count_transformed += 1

    print(
        f"Transformed positions: "
        f"{count_transformed}/{count_total}"
    )

    # --------------------------------------------------
    # SPEED & DISTANCE ESTIMATION
    # --------------------------------------------------

    speed_and_distance_estimator = (
        SpeedAndDistance_Estimator()
    )

    speed_and_distance_estimator\
        .add_speed_and_distance_to_tracks(
            tracks
        )

    print(
        "Speed and distance calculated"
    )

    # --------------------------------------------------
    # SAVE FIRST PLAYER CROP
    # --------------------------------------------------

    for track_id, player in (
        tracks["players"][0].items()
    ):

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
            f"Saved player crop: "
            f"output/player_crop.jpg "
            f"(Player ID: {track_id})"
        )

        break

    # --------------------------------------------------
    # TEAM ASSIGNMENT
    # --------------------------------------------------

    team_assigner = TeamAssigner()

    team_assigner.assign_team_color(
        frames[0],
        tracks["players"][0]
    )

    print("Team Colors:")
    print(
        team_assigner.team_colors
    )

    for frame_num, player_track in enumerate(
        tracks["players"]
    ):

        for player_id, track in (
            player_track.items()
        ):

            team = (
                team_assigner
                .get_player_team(
                    frames[frame_num],
                    track["bbox"],
                    player_id
                )
            )

            tracks["players"][
                frame_num
            ][player_id]["team"] = (
                team
            )

            tracks["players"][
                frame_num
            ][player_id]["team_color"] = (
                team_assigner
                .team_colors[team]
            )

    print("Teams assigned")

    # --------------------------------------------------
    # BALL POSSESSION
    # --------------------------------------------------

    player_ball_assigner = (
        PlayerBallAssigner()
    )

    team_ball_control = []

    possession_count = 0

    for frame_num, player_track in enumerate(
        tracks["players"]
    ):

        ball_bbox = (
            tracks["ball"][frame_num][1]["bbox"]
        )

        assigned_player = (
            player_ball_assigner
            .assign_ball_to_player(
                player_track,
                ball_bbox
            )
        )

        if assigned_player != -1:

            tracks["players"][
                frame_num
            ][assigned_player][
                "has_ball"
            ] = True

            team_ball_control.append(
                tracks["players"][
                    frame_num
                ][assigned_player][
                    "team"
                ]
            )

            possession_count += 1

            if frame_num % 100 == 0:

                print(
                    f"Frame {frame_num}: "
                    f"Player {assigned_player} "
                    f"has possession"
                )

        else:

            if len(team_ball_control) > 0:

                team_ball_control.append(
                    team_ball_control[-1]
                )

            else:

                team_ball_control.append(
                    1
                )

    team_ball_control = np.array(
        team_ball_control
    )

    print(
        f"Possession assigned on "
        f"{possession_count} frames"
    )

    # --------------------------------------------------
    # DRAW OUTPUT
    # --------------------------------------------------

    output_frames = (
        tracker.draw_annotations(
            frames,
            tracks,
            team_ball_control
        )
    )

    print(
        "Annotations complete"
    )

    # --------------------------------------------------
    # DRAW CAMERA MOVEMENT
    # --------------------------------------------------

    output_frames = (
        camera_movement_estimator
        .draw_camera_movement(
            output_frames,
            camera_movement_per_frame
        )
    )

    print(
        "Camera movement overlay added"
    )

    # --------------------------------------------------
    # DRAW SPEED & DISTANCE
    # --------------------------------------------------

    output_frames = (
        speed_and_distance_estimator
        .draw_speed_and_distance(
            output_frames,
            tracks
        )
    )

    print(
        "Speed and distance overlay added"
    )

    # --------------------------------------------------
    # SAVE VIDEO
    # --------------------------------------------------

    save_video(
        output_frames,
        "output/tracked_video.avi"
    )

    print(
        "Video saved to "
        "output/tracked_video.avi"
    )

    print("Done")


if __name__ == "__main__":
    main()