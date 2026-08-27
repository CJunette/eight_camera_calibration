import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


DEFAULT_CAMERA_COUNT = 8
DEFAULT_BOARD_SIZE = (9, 6)
DEFAULT_SQUARE_SIZE = 0.025
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 30


@dataclass
class CameraInfo:
    index: int
    opened: bool
    width: float = 0.0
    height: float = 0.0
    fps: float = 0.0
    backend: str = ""


def parse_board_size(value: str) -> Tuple[int, int]:
    separators = ("x", "X", ",")
    for separator in separators:
        if separator in value:
            left, right = value.split(separator, 1)
            return int(left), int(right)
    raise argparse.ArgumentTypeError("board size must look like 9x6")


def parse_indices(value: str) -> List[int]:
    indices: List[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        indices.append(int(part))
    if not indices:
        raise argparse.ArgumentTypeError("at least one camera index is required")
    return indices


def make_output_dirs(root: Path) -> Dict[str, Path]:
    dirs = {
        "root": root,
        "images": root / "images",
        "previews": root / "previews",
        "calibration": root / "calibration",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def open_camera(index: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(index)
    if capture.isOpened():
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, fps)
    return capture


def detect_cameras(max_index: int, width: int, height: int, fps: int) -> List[CameraInfo]:
    cameras: List[CameraInfo] = []
    for index in range(max_index + 1):
        capture = open_camera(index, width, height, fps)
        opened = capture.isOpened()
        if opened:
            ok, _ = capture.read()
            opened = bool(ok)
        cameras.append(
            CameraInfo(
                index=index,
                opened=opened,
                width=capture.get(cv2.CAP_PROP_FRAME_WIDTH) if opened else 0.0,
                height=capture.get(cv2.CAP_PROP_FRAME_HEIGHT) if opened else 0.0,
                fps=capture.get(cv2.CAP_PROP_FPS) if opened else 0.0,
                backend=capture.getBackendName() if opened else "",
            )
        )
        capture.release()
    return cameras


def print_camera_report(cameras: Sequence[CameraInfo], required_count: int) -> None:
    opened = [camera for camera in cameras if camera.opened]
    print(f"Found {len(opened)} available camera(s); required: {required_count}")
    for camera in cameras:
        status = "OK" if camera.opened else "--"
        print(
            f"[{status}] index={camera.index:<2} "
            f"size={camera.width:.0f}x{camera.height:.0f} "
            f"fps={camera.fps:.1f} backend={camera.backend}"
        )
    if len(opened) < required_count:
        raise SystemExit(
            f"Only {len(opened)} camera(s) are available. Connect/enable {required_count} cameras first."
        )


def open_camera_set(indices: Sequence[int], width: int, height: int, fps: int) -> Dict[int, cv2.VideoCapture]:
    captures: Dict[int, cv2.VideoCapture] = {}
    try:
        for index in indices:
            capture = open_camera(index, width, height, fps)
            if not capture.isOpened():
                raise RuntimeError(f"camera index {index} could not be opened")
            for _ in range(5):
                capture.read()
            captures[index] = capture
        return captures
    except Exception:
        release_all(captures.values())
        raise


def release_all(captures: Iterable[cv2.VideoCapture]) -> None:
    for capture in captures:
        capture.release()
    cv2.destroyAllWindows()


def read_frames(captures: Dict[int, cv2.VideoCapture]) -> Dict[int, np.ndarray]:
    frames: Dict[int, np.ndarray] = {}
    for index, capture in captures.items():
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"failed to read frame from camera {index}")
        frames[index] = frame
    return frames


def chessboard_object_points(board_size: Tuple[int, int], square_size: float) -> np.ndarray:
    points = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    points[:, :2] = np.mgrid[0 : board_size[0], 0 : board_size[1]].T.reshape(-1, 2)
    points *= square_size
    return points


def find_chessboard(frame: np.ndarray, board_size: Tuple[int, int]) -> Tuple[bool, Optional[np.ndarray]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    found, corners = cv2.findChessboardCorners(gray, board_size, flags)
    if not found or corners is None:
        return False, None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, refined


def draw_status(
    frame: np.ndarray,
    camera_index: int,
    sample_count: int,
    found: bool,
    corners: Optional[np.ndarray],
    board_size: Tuple[int, int],
) -> np.ndarray:
    preview = frame.copy()
    if corners is not None:
        cv2.drawChessboardCorners(preview, board_size, corners, found)
    color = (40, 220, 40) if found else (40, 40, 230)
    text = f"cam {camera_index} | samples {sample_count} | {'board OK' if found else 'no board'}"
    cv2.rectangle(preview, (8, 8), (520, 48), (0, 0, 0), -1)
    cv2.putText(preview, text, (18, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    return preview


def tile_previews(previews: Sequence[np.ndarray], columns: int = 4, cell_width: int = 480) -> np.ndarray:
    resized: List[np.ndarray] = []
    for preview in previews:
        height, width = preview.shape[:2]
        scale = cell_width / float(width)
        resized.append(cv2.resize(preview, (cell_width, int(height * scale))))
    if not resized:
        return np.zeros((100, 100, 3), np.uint8)
    cell_height = max(image.shape[0] for image in resized)
    rows: List[np.ndarray] = []
    for start in range(0, len(resized), columns):
        row_images = resized[start : start + columns]
        while len(row_images) < columns:
            row_images.append(np.zeros((cell_height, cell_width, 3), np.uint8))
        padded = []
        for image in row_images:
            if image.shape[0] < cell_height:
                pad = np.zeros((cell_height - image.shape[0], cell_width, 3), np.uint8)
                image = np.vstack([image, pad])
            padded.append(image)
        rows.append(np.hstack(padded))
    return np.vstack(rows)


def capture_samples(args: argparse.Namespace) -> None:
    dirs = make_output_dirs(Path(args.output))
    indices = args.indices or [camera.index for camera in detect_cameras(args.max_index, args.width, args.height, args.fps) if camera.opened]
    if len(indices) != args.camera_count:
        raise SystemExit(f"Expected exactly {args.camera_count} camera indices, got {indices}")

    object_points = chessboard_object_points(args.board_size, args.square_size)
    captures = open_camera_set(indices, args.width, args.height, args.fps)
    samples: List[dict] = []
    per_camera_counts = {index: 0 for index in indices}
    last_auto_capture = 0.0

    print("Controls: SPACE=save valid set, A=toggle auto capture, Q/ESC=quit")
    print("A sample is saved only when every configured camera sees the chessboard.")
    auto_capture = args.auto

    try:
        while True:
            frames = read_frames(captures)
            detections = {index: find_chessboard(frame, args.board_size) for index, frame in frames.items()}
            all_found = all(found for found, _ in detections.values())
            previews = [
                draw_status(frames[index], index, per_camera_counts[index], detections[index][0], detections[index][1], args.board_size)
                for index in indices
            ]
            tiled = tile_previews(previews)
            cv2.imshow("8-camera calibration capture", tiled)

            key = cv2.waitKey(1) & 0xFF
            should_save = key == ord(" ")
            now = time.time()
            if auto_capture and all_found and now - last_auto_capture >= args.interval:
                should_save = True
                last_auto_capture = now
            if key in (ord("a"), ord("A")):
                auto_capture = not auto_capture
                print(f"Auto capture: {'on' if auto_capture else 'off'}")
            if key in (ord("q"), ord("Q"), 27):
                break

            if should_save:
                if not all_found:
                    print("Skipped: not all cameras see the chessboard.")
                    continue
                sample_id = len(samples)
                sample_record = {"sample_id": sample_id, "time": now, "cameras": {}}
                for index in indices:
                    image_name = f"sample_{sample_id:04d}_cam_{index}.png"
                    preview_name = f"sample_{sample_id:04d}_cam_{index}_corners.png"
                    image_path = dirs["images"] / image_name
                    preview_path = dirs["previews"] / preview_name
                    found, corners = detections[index]
                    cv2.imwrite(str(image_path), frames[index])
                    cv2.imwrite(str(preview_path), draw_status(frames[index], index, sample_id + 1, found, corners, args.board_size))
                    sample_record["cameras"][str(index)] = {
                        "image": str(image_path.as_posix()),
                        "preview": str(preview_path.as_posix()),
                    }
                    per_camera_counts[index] += 1
                samples.append(sample_record)
                print(f"Saved sample {sample_id:04d}; total valid samples: {len(samples)}")
                if len(samples) >= args.samples:
                    break
    finally:
        release_all(captures.values())

    manifest = {
        "camera_indices": indices,
        "board_size": args.board_size,
        "square_size": args.square_size,
        "object_points": object_points.tolist(),
        "samples": samples,
    }
    manifest_path = dirs["root"] / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")


def load_manifest(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def calibrate_intrinsics(manifest: dict) -> Tuple[dict, Dict[str, list], Dict[str, list], Tuple[int, int]]:
    board_size = tuple(manifest["board_size"])
    square_size = float(manifest["square_size"])
    object_points_template = chessboard_object_points(board_size, square_size)
    camera_indices = [str(index) for index in manifest["camera_indices"]]

    object_points_by_camera: Dict[str, list] = {index: [] for index in camera_indices}
    image_points_by_camera: Dict[str, list] = {index: [] for index in camera_indices}
    image_size: Optional[Tuple[int, int]] = None

    for sample in manifest["samples"]:
        for camera_index in camera_indices:
            image_path = Path(sample["cameras"][camera_index]["image"])
            frame = cv2.imread(str(image_path))
            if frame is None:
                print(f"Warning: could not read {image_path}")
                continue
            image_size = (frame.shape[1], frame.shape[0])
            found, corners = find_chessboard(frame, board_size)
            if found and corners is not None:
                object_points_by_camera[camera_index].append(object_points_template)
                image_points_by_camera[camera_index].append(corners)

    if image_size is None:
        raise RuntimeError("no calibration images could be read")

    results = {}
    for camera_index in camera_indices:
        if len(image_points_by_camera[camera_index]) < 8:
            raise RuntimeError(
                f"camera {camera_index} has only {len(image_points_by_camera[camera_index])} valid samples; collect at least 8"
            )
        rms, matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
            object_points_by_camera[camera_index], image_points_by_camera[camera_index], image_size, None, None
        )
        results[camera_index] = {
            "rms": float(rms),
            "camera_matrix": matrix.tolist(),
            "distortion_coefficients": distortion.tolist(),
            "valid_samples": len(image_points_by_camera[camera_index]),
            "rvecs": [rvec.tolist() for rvec in rvecs],
            "tvecs": [tvec.tolist() for tvec in tvecs],
        }
        print(f"Intrinsic cam {camera_index}: RMS={rms:.4f}, samples={len(image_points_by_camera[camera_index])}")

    return results, object_points_by_camera, image_points_by_camera, image_size


def calibrate_stereo_pairs(
    manifest: dict,
    intrinsics: dict,
    image_size: Tuple[int, int],
    reference_camera: str,
) -> dict:
    board_size = tuple(manifest["board_size"])
    square_size = float(manifest["square_size"])
    object_points_template = chessboard_object_points(board_size, square_size)
    stereo_results = {}

    for camera_index in [str(index) for index in manifest["camera_indices"]]:
        if camera_index == reference_camera:
            continue
        object_points = []
        reference_points = []
        target_points = []
        for sample in manifest["samples"]:
            reference_image = cv2.imread(str(Path(sample["cameras"][reference_camera]["image"])))
            target_image = cv2.imread(str(Path(sample["cameras"][camera_index]["image"])))
            if reference_image is None or target_image is None:
                continue
            reference_found, reference_corners = find_chessboard(reference_image, board_size)
            target_found, target_corners = find_chessboard(target_image, board_size)
            if reference_found and target_found and reference_corners is not None and target_corners is not None:
                object_points.append(object_points_template)
                reference_points.append(reference_corners)
                target_points.append(target_corners)

        if len(object_points) < 8:
            print(f"Skipping stereo {reference_camera}->{camera_index}: only {len(object_points)} shared samples")
            continue

        ref_intrinsic = np.array(intrinsics[reference_camera]["camera_matrix"], dtype=np.float64)
        ref_distortion = np.array(intrinsics[reference_camera]["distortion_coefficients"], dtype=np.float64)
        cam_intrinsic = np.array(intrinsics[camera_index]["camera_matrix"], dtype=np.float64)
        cam_distortion = np.array(intrinsics[camera_index]["distortion_coefficients"], dtype=np.float64)
        flags = cv2.CALIB_FIX_INTRINSIC
        rms, _, _, _, _, rotation, translation, essential, fundamental = cv2.stereoCalibrate(
            object_points,
            reference_points,
            target_points,
            ref_intrinsic,
            ref_distortion,
            cam_intrinsic,
            cam_distortion,
            image_size,
            flags=flags,
        )
        stereo_results[camera_index] = {
            "reference_camera": reference_camera,
            "target_camera": camera_index,
            "rms": float(rms),
            "rotation_reference_to_target": rotation.tolist(),
            "translation_reference_to_target": translation.tolist(),
            "essential_matrix": essential.tolist(),
            "fundamental_matrix": fundamental.tolist(),
            "shared_samples": len(object_points),
        }
        print(f"Stereo {reference_camera}->{camera_index}: RMS={rms:.4f}, samples={len(object_points)}")

    return stereo_results


def run_calibration(args: argparse.Namespace) -> None:
    output_root = Path(args.output)
    dirs = make_output_dirs(output_root)
    manifest = load_manifest(output_root / "manifest.json")
    intrinsics, _, _, image_size = calibrate_intrinsics(manifest)
    reference_camera = str(args.reference)
    if reference_camera not in intrinsics:
        raise RuntimeError(f"reference camera {reference_camera} was not captured")
    stereo = calibrate_stereo_pairs(manifest, intrinsics, image_size, reference_camera)
    results = {
        "image_size": image_size,
        "board_size": manifest["board_size"],
        "square_size": manifest["square_size"],
        "reference_camera": reference_camera,
        "intrinsics": intrinsics,
        "extrinsics_relative_to_reference": stereo,
    }
    result_path = dirs["calibration"] / "calibration_result.json"
    result_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {result_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect, capture, and calibrate an 8-camera OpenCV rig.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--camera-count", type=int, default=DEFAULT_CAMERA_COUNT)
    common.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    common.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    common.add_argument("--fps", type=int, default=DEFAULT_FPS)
    common.add_argument("--max-index", type=int, default=16)

    detect = subparsers.add_parser("detect", parents=[common], help="List available camera device indices.")
    detect.set_defaults(func=lambda args: print_camera_report(detect_cameras(args.max_index, args.width, args.height, args.fps), args.camera_count))

    capture = subparsers.add_parser("capture", parents=[common], help="Capture synchronized chessboard images.")
    capture.add_argument("--indices", type=parse_indices, help="Comma-separated camera indices, for example 0,1,2,3,4,5,6,7")
    capture.add_argument("--board-size", type=parse_board_size, default=DEFAULT_BOARD_SIZE, help="Inner corners, e.g. 9x6")
    capture.add_argument("--square-size", type=float, default=DEFAULT_SQUARE_SIZE, help="Chessboard square size in meters")
    capture.add_argument("--samples", type=int, default=30, help="Valid sample sets to save")
    capture.add_argument("--interval", type=float, default=1.5, help="Auto-capture minimum interval in seconds")
    capture.add_argument("--auto", action="store_true", help="Start in auto-capture mode")
    capture.add_argument("--output", default="runs/latest", help="Output directory")
    capture.set_defaults(func=capture_samples)

    calibrate = subparsers.add_parser("calibrate", help="Calibrate intrinsics and extrinsics from saved images.")
    calibrate.add_argument("--output", default="runs/latest", help="Directory containing manifest.json")
    calibrate.add_argument("--reference", type=int, default=0, help="Reference camera index for extrinsics")
    calibrate.set_defaults(func=run_calibration)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()