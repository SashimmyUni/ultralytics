# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from ultralytics import YOLO
from ultralytics.data.loaders import get_best_youtube_url
from ultralytics.utils.plotting import Annotator
from ultralytics.utils.torch_utils import select_device


class TorchVisionVideoClassifier:
    """Video classifier using pretrained TorchVision models for action recognition.

    This class provides an interface for video classification using various pretrained models from TorchVision's video
    model collection, supporting models like S3D, R3D, Swin3D, and MViT architectures.

    Attributes:
        model (torch.nn.Module): The loaded TorchVision model for video classification.
        weights (torchvision.models.video.Weights): The weights used for the model.
        device (torch.device): The device on which the model is loaded.

    Methods:
        available_model_names: Returns a list of available model names.
        preprocess_crops_for_video_cls: Preprocesses crops for video classification.
        __call__: Performs inference on the given sequences.
        postprocess: Postprocesses the model's output.

    Examples:
        >>> classifier = TorchVisionVideoClassifier("s3d", device="cpu")
        >>> crops = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(8)]
        >>> tensor = classifier.preprocess_crops_for_video_cls(crops)
        >>> outputs = classifier(tensor)
        >>> labels, confidences = classifier.postprocess(outputs)

    References:
        https://pytorch.org/vision/stable/
    """

    from torchvision.models.video import (
        MViT_V1_B_Weights,
        MViT_V2_S_Weights,
        R3D_18_Weights,
        S3D_Weights,
        Swin3D_B_Weights,
        Swin3D_T_Weights,
        mvit_v1_b,
        mvit_v2_s,
        r3d_18,
        s3d,
        swin3d_b,
        swin3d_t,
    )

    model_name_to_model_and_weights = {
        "s3d": (s3d, S3D_Weights.DEFAULT),
        "r3d_18": (r3d_18, R3D_18_Weights.DEFAULT),
        "swin3d_t": (swin3d_t, Swin3D_T_Weights.DEFAULT),
        "swin3d_b": (swin3d_b, Swin3D_B_Weights.DEFAULT),
        "mvit_v1_b": (mvit_v1_b, MViT_V1_B_Weights.DEFAULT),
        "mvit_v2_s": (mvit_v2_s, MViT_V2_S_Weights.DEFAULT),
    }

    def __init__(self, model_name: str, device: str | torch.device = ""):
        """Initialize the VideoClassifier with the specified model name and device.

        Args:
            model_name (str): The name of the model to use. Must be one of the available models.
            device (str | torch.device): The device to run the model on.
        """
        if model_name not in self.model_name_to_model_and_weights:
            raise ValueError(f"Invalid model name '{model_name}'. Available models: {self.available_model_names()}")
        model, self.weights = self.model_name_to_model_and_weights[model_name]
        self.device = select_device(device)
        self.model = model(weights=self.weights).to(self.device).eval()

    @staticmethod
    def available_model_names() -> list[str]:
        """Get the list of available model names.

        Returns:
            (list[str]): List of available model names that can be used with this classifier.
        """
        return list(TorchVisionVideoClassifier.model_name_to_model_and_weights.keys())

    def preprocess_crops_for_video_cls(
        self, crops: list[np.ndarray], input_size: list[int] | None = None
    ) -> torch.Tensor:
        """Preprocess a list of crops for video classification.

        Args:
            crops (list[np.ndarray]): List of crops to preprocess. Each crop should have dimensions (H, W, C).
            input_size (list[int], optional): The target input size for the model.

        Returns:
            (torch.Tensor): Preprocessed crops as a tensor with dimensions (1, C, T, H, W).
        """
        if input_size is None:
            input_size = [224, 224]
        from torchvision.transforms import v2

        transform = v2.Compose(
            [
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize(input_size, antialias=True),
                v2.Normalize(mean=self.weights.transforms().mean, std=self.weights.transforms().std),
            ]
        )

        processed_crops = [transform(torch.from_numpy(crop).permute(2, 0, 1)) for crop in crops]
        return torch.stack(processed_crops).unsqueeze(0).permute(0, 2, 1, 3, 4).to(self.device)

    def __call__(self, sequences: torch.Tensor) -> torch.Tensor:
        """Perform inference on the given sequences.

        Args:
            sequences (torch.Tensor): The input sequences for the model with dimensions (B, C, T, H, W) for batched
                video frames or (C, T, H, W) for single video frames.

        Returns:
            (torch.Tensor): The model's output logits.
        """
        with torch.inference_mode():
            return self.model(sequences)

    def postprocess(self, outputs: torch.Tensor) -> tuple[list[str], list[float]]:
        """Postprocess the model's batch output.

        Args:
            outputs (torch.Tensor): The model's output logits.

        Returns:
            (tuple[list[str], list[float]]): Predicted labels and their confidence scores.
        """
        pred_labels = []
        pred_confs = []
        for output in outputs:
            pred_class = output.argmax(0).item()
            pred_label = self.weights.meta["categories"][pred_class]
            pred_labels.append(pred_label)
            pred_conf = output.softmax(0)[pred_class].item()
            pred_confs.append(pred_conf)

        return pred_labels, pred_confs


class HuggingFaceVideoClassifier:
    """Zero-shot video classifier using Hugging Face transformer models.

    This class provides an interface for zero-shot video classification using Hugging Face models, supporting custom
    label sets and various transformer architectures for video understanding.

    Attributes:
        fp16 (bool): Whether to use FP16 for inference.
        labels (list[str]): List of labels for zero-shot classification.
        device (torch.device): The device on which the model is loaded.
        processor (transformers.AutoProcessor): The processor for the model.
        model (transformers.AutoModel): The loaded Hugging Face model.

    Methods:
        preprocess_crops_for_video_cls: Preprocesses crops for video classification.
        __call__: Performs inference on the given sequences.
        postprocess: Postprocesses the model's output.

    Examples:
        >>> labels = ["walking", "running", "dancing"]
        >>> classifier = HuggingFaceVideoClassifier(labels, device="cpu")
        >>> crops = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(8)]
        >>> tensor = classifier.preprocess_crops_for_video_cls(crops)
        >>> outputs = classifier(tensor)
        >>> labels, confidences = classifier.postprocess(outputs)
    """

    def __init__(
        self,
        labels: list[str],
        model_name: str = "microsoft/xclip-base-patch16-zero-shot",
        device: str | torch.device = "",
        fp16: bool = False,
    ):
        """Initialize the HuggingFaceVideoClassifier with the specified model name.

        Args:
            labels (list[str]): List of labels for zero-shot classification.
            model_name (str): The name of the model to use.
            device (str | torch.device): The device to run the model on.
            fp16 (bool): Whether to use FP16 for inference.
        """
        self.fp16 = fp16
        self.labels = labels
        self.device = select_device(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(self.device)
        if fp16:
            model = model.half()
        self.model = model.eval()

    def preprocess_crops_for_video_cls(
        self, crops: list[np.ndarray], input_size: list[int] | None = None
    ) -> torch.Tensor:
        """Preprocess a list of crops for video classification.

        Args:
            crops (list[np.ndarray]): List of crops to preprocess. Each crop should have dimensions (H, W, C).
            input_size (list[int], optional): The target input size for the model.

        Returns:
            (torch.Tensor): Preprocessed crops as a tensor with dimensions (1, T, C, H, W).
        """
        if input_size is None:
            input_size = [224, 224]
        from torchvision import transforms

        transform = transforms.Compose(
            [
                transforms.Lambda(lambda x: x.float() / 255.0),
                transforms.Resize(input_size),
                transforms.Normalize(
                    mean=self.processor.image_processor.image_mean, std=self.processor.image_processor.image_std
                ),
            ]
        )

        processed_crops = [transform(torch.from_numpy(crop).permute(2, 0, 1)) for crop in crops]  # (T, C, H, W)
        output = torch.stack(processed_crops).unsqueeze(0).to(self.device)  # (1, T, C, H, W)
        if self.fp16:
            output = output.half()
        return output

    def __call__(self, sequences: torch.Tensor) -> torch.Tensor:
        """Perform inference on the given sequences.

        Args:
            sequences (torch.Tensor): Batched input video frames with shape (B, T, C, H, W).

        Returns:
            (torch.Tensor): The model's output logits.
        """
        input_ids = self.processor(text=self.labels, return_tensors="pt", padding=True)["input_ids"].to(self.device)

        inputs = {"pixel_values": sequences, "input_ids": input_ids}

        with torch.inference_mode():
            outputs = self.model(**inputs)

        return outputs.logits_per_video

    def postprocess(self, outputs: torch.Tensor) -> tuple[list[list[str]], list[list[float]]]:
        """Postprocess the model's batch output.

        Args:
            outputs (torch.Tensor): The model's output logits.

        Returns:
            (tuple[list[list[str]], list[list[float]]]): Predicted top-2 labels and confidence scores for each sample.
        """
        pred_labels = []
        pred_confs = []

        with torch.no_grad():
            logits_per_video = outputs  # Assuming outputs is already the logits tensor
            probs = logits_per_video.softmax(dim=-1)  # Use softmax to convert logits to probabilities

        for prob in probs:
            top2_indices = prob.topk(2).indices.tolist()
            top2_labels = [self.labels[idx] for idx in top2_indices]
            top2_confs = prob[top2_indices].tolist()
            pred_labels.append(top2_labels)
            pred_confs.append(top2_confs)

        return pred_labels, pred_confs




def save_keypoint_sequences(track_kpts_history: dict[int, list[np.ndarray]], out_dir: str) -> None:
    """Save stored keypoints for each track to disk.

    Each track's sequence is written as a NumPy ``.npy`` file named
    ``track_<id>.npy``.  The output directory is created if necessary.
    """
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    for tid, seq in track_kpts_history.items():
        if len(seq) == 0:
            continue
        arr = np.stack(seq)  # shape (T, num_kpts, 2|3)
        np.save(path / f"track_{tid}.npy", arr)


def draw_pose_on_blank(kpts: np.ndarray, shape: tuple[int,int]) -> np.ndarray:
    """Return a blank image with the pose skeleton drawn.

    Args:
        kpts (np.ndarray): Array of keypoints shape (N,2) or (N,3).
        shape (tuple[int,int]): (height, width) of canvas.
    """
    canvas = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    annot = Annotator(canvas, line_width=3, font_size=10, pil=False)
    annot.kpts(kpts, shape=shape)
    return annot.im


def crop_and_pad(frame: np.ndarray, box: list[float], margin_percent: int) -> np.ndarray:
    """Crop box with margin and take square crop from frame.

    Args:
        frame (np.ndarray): The input frame to crop from.
        box (list[float]): The bounding box coordinates [x1, y1, x2, y2].
        margin_percent (int): The percentage of margin to add around the box.

    Returns:
        (np.ndarray): The cropped and resized square image.
    """
    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1

    # Add margin
    margin_x, margin_y = int(w * margin_percent / 100), int(h * margin_percent / 100)
    x1, y1 = max(0, x1 - margin_x), max(0, y1 - margin_y)
    x2, y2 = min(frame.shape[1], x2 + margin_x), min(frame.shape[0], y2 + margin_y)

    # Take square crop from frame
    size = max(y2 - y1, x2 - x1)
    center_y, center_x = (y1 + y2) // 2, (x1 + x2) // 2
    half_size = size // 2
    square_crop = frame[
        max(0, center_y - half_size) : min(frame.shape[0], center_y + half_size),
        max(0, center_x - half_size) : min(frame.shape[1], center_x + half_size),
    ]

    return cv2.resize(square_crop, (224, 224), interpolation=cv2.INTER_LINEAR)


def run(
    weights: str = "yolo26n-pose.pt",
    device: str = "",
    source: str = "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    output_path: str | None = None,
    crop_margin_percentage: int = 10,
    num_video_sequence_samples: int = 8,
    skip_frame: int = 2,
    video_cls_overlap_ratio: float = 0.25,
    fp16: bool = False,
    video_classifier_model: str = "microsoft/xclip-base-patch32",
    labels: list[str] | None = None,
    save_kpts_dir: str | None = None,
    visualize_pose: bool = False,
    pose_output_path: str | None = None,
    draw_boxes: bool = True,
    overlay_pose: bool = True,
) -> None:
    """Run action recognition on a video source using YOLO for object detection (or pose
    estimation) and a video classifier.

    A pose-capable YOLO model (e.g. any YOLO26-pose weights) will produce keypoints which are
    optionally stored and drawn on the output frames.  The rest of the pipeline continues to use
    cropped person patches for action classification as before.

    Args:
        weights (str): Path to the YOLO model weights.  Specifying a "-pose" file will enable
            pose keypoint extraction.
        device (str): Device to run the model on. Use 'cuda' for NVIDIA GPU, 'mps' for Apple Silicon, or 'cpu'.
        source (str): Path to mp4 video file or YouTube URL.
        output_path (str, optional): Path to save the output video.
        crop_margin_percentage (int): Percentage of margin to add around detected objects.
        num_video_sequence_samples (int): Number of video frames to use for classification.
        skip_frame (int): Number of frames to skip between detections.
        video_cls_overlap_ratio (float): Overlap ratio between video sequences.
        fp16 (bool): Whether to use half-precision floating point.
        video_classifier_model (str): Name or path of the video classifier model.
        labels (list[str], optional): List of labels for zero-shot classification.
        save_kpts_dir (str, optional): Directory in which to store per-track keypoint
            sequences.  If specified the folder will be created and each track will be
            written as ``track_<id>.npy`` after processing completes.
        visualize_pose (bool): If True, show a second window containing only the
            skeleton overlay (black background) for each detected person.
        pose_output_path (str, optional): Path to save a video of the pose-only window.
            Only used when ``visualize_pose`` is True.  The file will have the same
            resolution as the input frames.
            If not provided the default ``datasets/pose_video.mp4`` will be used
            (directory is created if necessary).
        draw_boxes (bool): If False, bounding-box annotations will not be drawn on the
            main video; useful when using pose models and you only want skeletons.
    """

    if labels is None or len(labels) == 0:
        # fall back to a martial arts combat vocabulary if nothing was provided
        labels = [
            "jab",              # quick straight punch
            "cross",            # rear-hand straight punch
            "hook",             # circular punch
            "uppercut",         # upward punch
            "front kick",
            "roundhouse kick",
            "side kick",
            "block",
            "dodge",
            "sparring",
        ]
    # Initialize models and device
    device = select_device(device)
    # detect whether we're running a pose model so we can pull keypoints
    is_pose_model = "pose" in weights
    yolo_model = YOLO(weights).to(device)
    if video_classifier_model in TorchVisionVideoClassifier.available_model_names():
        print("'fp16' is not supported for TorchVisionVideoClassifier. Setting fp16 to False.")
        print(
            "'labels' is not used for TorchVisionVideoClassifier. Ignoring the provided labels and using Kinetics-400 labels."
        )
        video_classifier = TorchVisionVideoClassifier(video_classifier_model, device=device)
    else:
        video_classifier = HuggingFaceVideoClassifier(
            labels, model_name=video_classifier_model, device=device, fp16=fp16
        )

    # Initialize video capture
    if source.startswith("http") and urlparse(source).hostname in {"www.youtube.com", "youtube.com", "youtu.be"}:
        try:
            source = get_best_youtube_url(source)
        except Exception as e:
            # some YouTube streams may have None resolutions; fall back to original URL
            print(f"warning: failed to select best YouTube stream ({e}), using original URL")
    elif not source.endswith(".mp4"):
        raise ValueError("Invalid source. Supported sources are YouTube URLs and MP4 files.")
    cap = cv2.VideoCapture(source)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialize VideoWriter
    if output_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Initialize track history (crops + optional keypoints)
    track_history: dict[int, list[np.ndarray]] = defaultdict(list)
    track_kpts_history: dict[int, list[np.ndarray]] = defaultdict(list) if is_pose_model else {}
    frame_counter = 0
    # prepare pose video writer if requested
    pose_writer = None
    if visualize_pose:
        if pose_output_path is None:
            # default to datasets folder
            default_path = Path("datasets") / "pose_video.mp4"
            default_path.parent.mkdir(parents=True, exist_ok=True)
            pose_output_path = str(default_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        pose_writer = cv2.VideoWriter(pose_output_path, fourcc, fps, (frame_width, frame_height))

    track_ids_to_infer = []
    crops_to_infer: list[torch.Tensor] = []
    pred_labels: list = []
    pred_confs: list = []

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_counter += 1

            # Run YOLO tracking (works for pose models too)
            results = yolo_model.track(frame, persist=True, classes=[0])  # Track only person class

            if results[0].boxes.is_track:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy()

                # pull keypoints if provided by the model
                kpt_arr = None
                if is_pose_model and hasattr(results[0], "keypoints"):
                    kpt_arr = results[0].keypoints.cpu().numpy()  # (N,17,3)
                    # debug: report shapes so we know keypoints are arriving
                    print(f"[debug] frame {frame_counter}: kpt_arr shape={kpt_arr.shape}")

                # Visualize prediction
                annotator = Annotator(frame, line_width=3, font_size=10, pil=False)

                if frame_counter % skip_frame == 0:
                    crops_to_infer = []
                    track_ids_to_infer = []

                for i, (box, track_id) in enumerate(zip(boxes, track_ids)):
                    if frame_counter % skip_frame == 0:
                        crop = crop_and_pad(frame, box, crop_margin_percentage)
                        track_history[track_id].append(crop)
                        # store keypoints history if pose model
                        if kpt_arr is not None:
                            history = track_kpts_history.setdefault(track_id, [])
                            history.append(kpt_arr[i])
                            # optionally flush to disk so data isn't lost
                            if save_kpts_dir is not None and is_pose_model:
                                out_path = Path(save_kpts_dir)
                                out_path.mkdir(parents=True, exist_ok=True)
                                try:
                                    np.save(out_path / f"track_{track_id}.npy", np.stack(history))
                                except Exception:
                                    pass

                    if len(track_history[track_id]) > num_video_sequence_samples:
                        track_history[track_id].pop(0)
                        if is_pose_model:
                            track_kpts_history[track_id].pop(0)

                    if len(track_history[track_id]) == num_video_sequence_samples and frame_counter % skip_frame == 0:
                        start_time = time.time()
                        crops = video_classifier.preprocess_crops_for_video_cls(track_history[track_id])
                        end_time = time.time()
                        preprocess_time = end_time - start_time
                        print(f"video cls preprocess time: {preprocess_time:.4f} seconds")
                        crops_to_infer.append(crops)
                        track_ids_to_infer.append(track_id)

                if crops_to_infer and (
                    not pred_labels
                    or frame_counter % int(num_video_sequence_samples * skip_frame * (1 - video_cls_overlap_ratio)) == 0
                ):
                    crops_batch = torch.cat(crops_to_infer, dim=0)

                    start_inference_time = time.time()
                    output_batch = video_classifier(crops_batch)
                    end_inference_time = time.time()
                    inference_time = end_inference_time - start_inference_time
                    print(f"video cls inference time: {inference_time:.4f} seconds")

                    pred_labels, pred_confs = video_classifier.postprocess(output_batch)

                if track_ids_to_infer and crops_to_infer:
                    if draw_boxes:
                        for box, track_id, pred_label, pred_conf in zip(boxes, track_ids_to_infer, pred_labels, pred_confs):
                            top2_preds = sorted(zip(pred_label, pred_conf), key=lambda x: x[1], reverse=True)
                            label_text = " | ".join([f"{label} ({conf:.2f})" for label, conf in top2_preds])
                            annotator.box_label(box, label_text, color=(0, 0, 255))

                # overlay keypoints if present
                # show main annotated frame
                cv2.imshow("Video", frame)
                if output_path is not None:
                    out.write(frame)
                last_kp_arr = None
                if kpt_arr is not None and overlay_pose:
                    # iterate elements individually and coerce each one to numpy; some
                    # YOLO variants may emit ragged/irregular lists which would confuse
                    # a full-array conversion.
                    for raw in kpt_arr:
                        try:
                            kp_arr = np.asarray(raw)
                        except Exception:
                            continue
                        # collapse any extra leading dims so we always end up with
                        # shape (M,2|3)
                        if kp_arr.ndim > 2:
                            try:
                                kp_arr = kp_arr.reshape(-1, kp_arr.shape[-2], kp_arr.shape[-1])
                            except Exception:
                                # if reshape fails, just skip this entry
                                continue
                        last_kp_arr = kp_arr
                        for kp in kp_arr:
                            annotator.kpts(kp, shape=(frame_height, frame_width))
                if visualize_pose:
                    # draw last kp_arr on separate window, or blank if missing
                    if last_kp_arr is not None:
                        pose_img = draw_pose_on_blank(last_kp_arr, (frame_height, frame_width))
                    else:
                        # no keypoints this frame, produce an empty canvas
                        pose_img = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                    cv2.imshow("Pose", pose_img)
                    if pose_writer is not None:
                        pose_writer.write(pose_img)
                # refresh windows and allow quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        # always release resources
        cap.release()
        if output_path is not None:
            out.release()
        if pose_writer is not None:
            pose_writer.release()
        cv2.destroyAllWindows()
def parse_opt() -> argparse.Namespace:
    """Parse command line arguments for action recognition pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=str,
        default="yolo26n-pose.pt",
        help="ultralytics detector model path (e.g. yolo26n-pose.pt for pose)",
    )
    parser.add_argument("--device", default="", help='cuda device, i.e. 0 or 0,1,2,3 or cpu/mps, "" for auto-detection')
    parser.add_argument(
        "--source",
        type=str,
        default="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        help="video file path or youtube URL",
    )
    parser.add_argument("--output-path", type=str, default="output_video.mp4", help="output video file path")
    parser.add_argument(
        "--crop-margin-percentage", type=int, default=10, help="percentage of margin to add around detected objects"
    )
    parser.add_argument(
        "--num-video-sequence-samples", type=int, default=8, help="number of video frames to use for classification"
    )
    parser.add_argument("--skip-frame", type=int, default=2, help="number of frames to skip between detections")
    parser.add_argument(
        "--video-cls-overlap-ratio", type=float, default=0.25, help="overlap ratio between video sequences"
    )
    parser.add_argument("--fp16", action="store_true", help="use FP16 for inference")
    parser.add_argument(
        "--video-classifier-model", type=str, default="microsoft/xclip-base-patch32", help="video classifier model name"
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        type=str,
        default=["jab", "cross", "hook", "front kick"],
        help="labels for zero-shot video classification",
    )
    parser.add_argument(
        "--save-kpts-dir",
        type=str,
        default=None,
        help="optional directory where per-track keypoint sequences will be saved",
    )
    parser.add_argument(
        "--visualize-pose",
        action="store_true",
        help="display a separate window showing only the pose skeleton on black",
    )
    parser.add_argument(
        "--pose-output-path",
        type=str,
        default=None,
        help="file to write the pose-only video (requires --visualize-pose). ``datasets/pose_video.mp4`` if omitted",
    )
    parser.add_argument(
        "--no-boxes",
        dest="draw_boxes",
        action="store_false",
        help="do not draw bounding boxes on the main video (useful for pose models)",
    )
    parser.add_argument(
        "--no-overlay-pose",
        dest="overlay_pose",
        action="store_false",
        help="do not overlay pose skeletons on the main video",
    )
    return parser.parse_args()


def main(opt: argparse.Namespace) -> None:
    """Run the action recognition pipeline with parsed command line arguments."""
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

#python action_recognition.py --weights yolo26n-pose.pt --device 0 --source "https://www.youtube.com/watch?v=ymNcMLqzNwI" --output-path output_video.mp4 --video-classifier-model microsoft/xclip-base-patch32 --save-kpts-dir keypoints --visualize-pose --pose-output-path datasets/pose_video.mp4

#python action_recognition.py --weights yolo26n-pose.pt --device 0 --source "https://www.youtube.com/watch?v=ymNcMLqzNwI" --output-path output_video.mp4 --video-classifier-model microsoft/xclip-base-patch32 --save-kpts-dir keypoints --visualize-pose --pose-output-path datasets/pose_video.mp4 --no-boxes   