# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Simple tool to compare a live pose against a dataset of correct techniques by
analyzing keypoints and joint angles.

The script uses a YOLO pose model to detect human keypoints on video frames or a
file. A small set of joint triplets (e.g. shoulder-elbow-wrist) is used to
calculate the angle at the middle joint. A "correct" dataset consisting of
precomputed angle vectors is loaded from CSV and the current frame's angles are
compared to the dataset mean (or nearest exemplar). Differences are displayed on
the output video to give the user feedback about how much their form deviates
from the reference.

Usage example::

    python examples/pose_corrector.py \
        --source 0 \
        --correct-data correct_angles.csv \
        --weights yolo26n-pose.pt \
        --output-path feedback.mp4

The reference CSV should have a column for each monitored joint name such as
"left_elbow" or "right_knee".  Each row corresponds to a single example frame
(or time step) of the correct technique.  Angles are measured in degrees.

"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch

from ultralytics import YOLO
from ultralytics.data.loaders import get_best_youtube_url
from ultralytics.utils.torch_utils import select_device
from ultralytics.utils.plotting import Annotator


# choose the triplets we care about (middle index is the joint at which the
# angle is measured).  Using COCO keypoint indices.
JOINT_TRIPLETS = [
    ("left_elbow", 5, 7, 9),   # shoulder-elbow-wrist
    ("right_elbow", 6, 8, 10),
    ("left_shoulder", 7, 5, 11),  # elbow-shoulder-hip
    ("right_shoulder", 8, 6, 12),
    ("left_knee", 11, 13, 15), # hip-knee-ankle
    ("right_knee", 12, 14, 16),
]


# helper functions

def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return the angle (in degrees) formed by points a-b-c with vertex at b."""
    ba = a - b
    bc = c - b
    # safeguard against zero-length vectors
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-8
    cosang = np.dot(ba, bc) / denom
    # clip numerical noise
    cosang = float(np.clip(cosang, -1.0, 1.0))
    return math.degrees(math.acos(cosang))


def compute_angles(kps_xy: np.ndarray, kps_conf: np.ndarray | None = None) -> dict[str, float | None]:
    """Compute named joint angles given keypoints.

    Args:
        kps_xy: (K,2) array of keypoint coordinates.
        kps_conf: optional (K,) confidences.  Angles are only computed when all
            three points have confidence >0.3.

    Returns:
        mapping from joint name to angle (degrees) or None if not available.
    """
    angles: dict[str, float | None] = {}
    for name, i, j, k in JOINT_TRIPLETS:
        if kps_conf is not None:
            if kps_conf[i] < 0.3 or kps_conf[j] < 0.3 or kps_conf[k] < 0.3:
                angles[name] = None
                continue
        pts = kps_xy[[i, j, k], :]
        if np.any(pts == 0) or pts.shape[0] != 3:
            angles[name] = None
        else:
            angles[name] = calculate_angle(pts[0], pts[1], pts[2])
    return angles


def load_correct_data(path: Path) -> pd.DataFrame:
    """Load a reference CSV containing angle columns.

    The file is expected to have one column per joint name used in
    ``JOINT_TRIPLETS``.  Additional columns are ignored.  The returned DataFrame
    will only contain the matching columns.
    """
    df = pd.read_csv(path)
    # keep only those names appearing in our triplets
    cols = [name for (name, *_rest) in JOINT_TRIPLETS if name in df.columns]
    if not cols:
        raise ValueError(f"reference file {path} contains no required columns")
    return df[cols]


def compare_to_reference(
    current: dict[str, float | None],
    ref_df: pd.DataFrame,
) -> tuple[dict[str, float | None], dict[str, float]]:
    """Return difference between current angles and the reference mean.

    Note: we simply compare to the column-wise mean of ``ref_df`` for
    simplicity.  For a more advanced system one could compute a nearest neighbour
    or dynamic time warping.
    """
    mean = ref_df.mean(axis=0)
    diffs: dict[str, float | None] = {}
    for name, val in current.items():
        if val is None or name not in mean:
            diffs[name] = None
        else:
            diffs[name] = val - float(mean[name])
    return diffs, mean.to_dict()


# main pipeline

def _resolve_source(src: str) -> str:
    """If a YouTube URL is provided, replace it with a direct stream URL."""
    if src.startswith("http"):
        try:
            parsed = get_best_youtube_url(src)
            return parsed
        except Exception:
            return src
    return src


def generate_reference_csv(
    weights: str,
    source: str,
    device: str,
    output_csv: str,
) -> None:
    """Capture frames from ``source`` and save computed angles to CSV.

    The resulting file can be supplied later as ``--correct-data`` when running
    :func:`run`.

    You can supply a YouTube link directly; the helper will fetch the best
    stream URL before opening it with OpenCV.
    """
    device = select_device(device)
    model = YOLO(weights).to(device)
    source = _resolve_source(source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"could not open source {source}")
    rows: list[dict[str, Any]] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame)
        if results and results[0].keypoints is not None:
            kps_xy = results[0].keypoints.xy.cpu().numpy()[0]
            kps_conf = results[0].keypoints.conf.cpu().numpy()[0]
            angles = compute_angles(kps_xy, kps_conf)
            rows.append({k: v for k, v in angles.items() if v is not None})
        # show live so user can stop when satisfied
        cv2.imshow("Recording reference", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_csv, index=False)
        print(f"Saved reference angles to {output_csv}")
    else:
        print("No valid poses detected; no CSV written")


def run(
    weights: str,
    source: str,
    correct_data: str,
    device: str = "",
    output_path: str | None = None,
) -> None:
    """Run live pose comparison and feedback.

    ``correct_data`` must point to a precomputed CSV of reference angles.  If the
    user wants to generate such a CSV from a video they can call
    :func:`generate_reference_csv` instead by passing the ``--create-reference``
    flag in the CLI (see :func:`parse_opt`).
    """
    device = select_device(device)
    model = YOLO(weights).to(device)

    ref_df = load_correct_data(Path(correct_data))

    source = _resolve_source(source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"could not open source {source}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
    else:
        writer = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame)
        annotator = Annotator(frame, line_width=3, font_size=12, pil=False)
        if results and results[0].keypoints is not None:
            kps_xy = results[0].keypoints.xy.cpu().numpy()[0]  # (K,2)
            kps_conf = results[0].keypoints.conf.cpu().numpy()[0]
            angles = compute_angles(kps_xy, kps_conf)
            diffs, ref_mean = compare_to_reference(angles, ref_df)

            # draw skeleton
            for (x, y), c in zip(kps_xy, kps_conf):
                if c > 0.3:
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
            for name, diff in diffs.items():
                if diff is None:
                    continue
                # display difference value on screen
                text = f"{name}:{diff:+.1f}\u00b0"
                cv2.putText(frame, text, (10, 30 + 20 * list(diffs).index(name)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255) if abs(diff) < 10 else (0,0,255), 2)

        if writer is not None:
            writer.write(frame)
        cv2.imshow("Pose correction", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


def parse_opt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolo26n-pose.pt", help="pose model weights")
    parser.add_argument("--device", default="", help="cuda device e.g. 0 or cpu/mps")
    parser.add_argument("--source", type=str, default="0", help="video file, camera index, or YouTube URL")
    parser.add_argument(
        "--correct-data",
        type=str,
        default=None,
        help="path to CSV with reference angles (required unless --create-reference is set)",
    )
    parser.add_argument("--output-path", type=str, default=None, help="save annotated video")
    parser.add_argument(
        "--create-reference",
        action="store_true",
        help="record angles from source and write reference CSV instead of running feedback",
    )
    parser.add_argument(
        "--ref-output",
        type=str,
        default="reference.csv",
        help="file path to write reference data when --create-reference is used",
    )
    return parser.parse_args()


def main(opt: argparse.Namespace) -> None:
    if opt.create_reference:
        generate_reference_csv(opt.weights, opt.source, opt.device, opt.ref_output)
    else:
        if opt.correct_data is None:
            raise ValueError("--correct-data must be provided when not creating a reference")
        run(opt.weights, opt.source, opt.correct_data, opt.device, opt.output_path)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
