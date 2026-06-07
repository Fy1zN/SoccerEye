from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
import numpy as np

from utils import (
    get_center_of_bbox,
    get_bbox_width,
    get_foot_position
)


class Tracker:

    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):

        batch_size = 20
        detections = []

        for i in range(0, len(frames), batch_size):

            detections_batch = self.model.predict(
                frames[i:i + batch_size],
                conf=0.25,
                verbose=False
            )

            detections.extend(detections_batch)

        return detections

    def get_object_tracks(
        self,
        frames,
        read_from_stub=False,
        stub_path=None
    ):

        if (
            read_from_stub
            and stub_path is not None
            and os.path.exists(stub_path)
        ):
            with open(stub_path, "rb") as f:
                tracks = pickle.load(f)

            return tracks

        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):

            cls_names = detection.names
            cls_names_inv = {
                v: k for k, v in cls_names.items()
            }

            detection_sv = sv.Detections.from_ultralytics(
                detection
            )

            # convert goalkeeper -> player
            for i, class_id in enumerate(
                detection_sv.class_id
            ):
                if cls_names[class_id] == "goalkeeper":
                    detection_sv.class_id[i] = (
                        cls_names_inv["player"]
                    )

            tracked = (
                self.tracker
                .update_with_detections(
                    detection_sv
                )
            )

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            # tracked objects
            for item in tracked:

                bbox = item[0].tolist()
                cls_id = item[3]
                track_id = item[4]

                if cls_id == cls_names_inv["player"]:

                    tracks["players"][frame_num][
                        track_id
                    ] = {
                        "bbox": bbox
                    }

                elif cls_id == cls_names_inv["referee"]:

                    tracks["referees"][frame_num][
                        track_id
                    ] = {
                        "bbox": bbox
                    }

            # ball detections
            for item in detection_sv:

                bbox = item[0].tolist()
                cls_id = item[3]

                if cls_id == cls_names_inv["ball"]:

                    tracks["ball"][frame_num][1] = {
                        "bbox": bbox
                    }

        if stub_path is not None:

            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(
        self,
        frame,
        bbox,
        color,
        track_id=None
    ):

        y2 = int(bbox[3])

        x_center, _ = (
            get_center_of_bbox(bbox)
        )

        width = (
            get_bbox_width(bbox)
        )

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(
                int(width),
                int(0.35 * width)
            ),
            angle=0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2
        )

        if track_id is not None:

            cv2.putText(
                frame,
                str(track_id),
                (x_center - 10, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        return frame

    def draw_triangle(
        self,
        frame,
        bbox,
        color
    ):

        y = int(bbox[1])

        x, _ = (
            get_center_of_bbox(bbox)
        )

        points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20]
        ])

        cv2.drawContours(
            frame,
            [points],
            0,
            color,
            cv2.FILLED
        )

        cv2.drawContours(
            frame,
            [points],
            0,
            (0, 0, 0),
            2
        )

        return frame

    def draw_annotations(
        self,
        video_frames,
        tracks
    ):

        output_frames = []

        for frame_num, frame in enumerate(
            video_frames
        ):

            frame = frame.copy()

            player_dict = (
                tracks["players"][frame_num]
            )

            referee_dict = (
                tracks["referees"][frame_num]
            )

            ball_dict = (
                tracks["ball"][frame_num]
            )

            # players
            for track_id, player in (
                player_dict.items()
            ):

                color = player.get(
                    "team_color",
                    (255, 0, 0)
                )

                color = tuple(
                    int(c) for c in color
                )

                frame = self.draw_ellipse(
                    frame,
                    player["bbox"],
                    color,
                    track_id
                )

            # referees
            for _, referee in (
                referee_dict.items()
            ):

                frame = self.draw_ellipse(
                    frame,
                    referee["bbox"],
                    (0, 255, 255)
                )

            # ball
            for _, ball in (
                ball_dict.items()
            ):

                frame = self.draw_triangle(
                    frame,
                    ball["bbox"],
                    (0, 255, 0)
                )

            output_frames.append(frame)

        return output_framess