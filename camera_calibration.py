import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


DEFAULT_CAMERA_COUNT = 8
DEFAULT_BOARD_SIZE = (9, 6)
DEFAULT_SQUARE_SIZE = 0.025
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 30
DEFAULT_FOURCC = "MJPG"
DEFAULT_BOARD_TYPE = "chessboard"
DEFAULT_CHARUCO_SQUARES = (9, 7)
DEFAULT_MARKER_SIZE = 0.018
DEFAULT_ARUCO_DICT = "DICT_4X4_50"
DEFAULT_MIN_CHARUCO_CORNERS = 12


@dataclass
class CameraInfo:
    index: int
    opened: bool
    width: float = 0.0
    height: float = 0.0
    fps: float = 0.0
    backend: str = ""


@dataclass
class TargetDetection:
    found: bool
    corners: Optional[np.ndarray]
    ids: Optional[np.ndarray] = None
    marker_corners: Optional[list] = None
    marker_ids: Optional[np.ndarray] = None


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


def open_camera(index: int, width: int, height: int, fps: int, fourcc: str) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(index)
    if capture.isOpened():
        if fourcc:
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc[:4]))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, fps)
    return capture


def detect_cameras(max_index: int, width: int, height: int, fps: int, fourcc: str) -> List[CameraInfo]:
    cameras: List[CameraInfo] = []
    for index in range(max_index + 1):
        capture = open_camera(index, width, height, fps, fourcc)
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


def open_camera_set(indices: Sequence[int], width: int, height: int, fps: int, fourcc: str) -> Dict[int, cv2.VideoCapture]:
    captures: Dict[int, cv2.VideoCapture] = {}
    try:
        for index in indices:
            capture = open_camera(index, width, height, fps, fourcc)
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


def make_aruco_dictionary(dictionary_name: str) -> Any:
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("cv2.aruco is unavailable. Install opencv-contrib-python instead of opencv-python.")
    if not hasattr(cv2.aruco, dictionary_name):
        names = sorted(name for name in dir(cv2.aruco) if name.startswith("DICT_"))
        raise ValueError(f"unknown ArUco dictionary {dictionary_name}; choose one of: {', '.join(names)}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))


def make_charuco_board(
    squares: Tuple[int, int], square_size: float, marker_size: float, dictionary_name: str, legacy_pattern: bool = False
) -> Tuple[Any, Any]:
    if squares[0] < 2 or squares[1] < 2:
        raise ValueError("ChArUco board needs at least 2x2 squares")
    if square_size <= 0 or marker_size <= 0 or marker_size >= square_size:
        raise ValueError("marker size must be positive and smaller than square size")
    dictionary = make_aruco_dictionary(dictionary_name)
    if hasattr(cv2.aruco, "CharucoBoard"):
        board = cv2.aruco.CharucoBoard(squares, square_size, marker_size, dictionary)
    else:
        board = cv2.aruco.CharucoBoard_create(squares[0], squares[1], square_size, marker_size, dictionary)
    if legacy_pattern:
        if not hasattr(board, "setLegacyPattern"):
            raise RuntimeError("this OpenCV version cannot enable the legacy ChArUco pattern; use OpenCV 4.6 or newer")
        board.setLegacyPattern(True)
    return board, dictionary


def create_detector_parameters() -> Any:
    if hasattr(cv2.aruco, "DetectorParameters"):
        return cv2.aruco.DetectorParameters()
    return cv2.aruco.DetectorParameters_create()


def board_object_points(board: Any, ids: np.ndarray) -> np.ndarray:
    object_points = board.getChessboardCorners() if hasattr(board, "getChessboardCorners") else board.chessboardCorners
    return np.array([object_points[int(charuco_id)] for charuco_id in ids.flatten()], dtype=np.float32)


def find_chessboard(frame: np.ndarray, board_size: Tuple[int, int]) -> TargetDetection:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    found, corners = cv2.findChessboardCorners(gray, board_size, flags)
    if not found or corners is None:
        return TargetDetection(False, None)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return TargetDetection(True, refined)


def find_charuco(
    frame: np.ndarray, board: Any, dictionary: Any, min_corners: int, detector: Optional[Any] = None
) -> TargetDetection:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if hasattr(cv2.aruco, "CharucoDetector"):
        charuco_detector = detector or cv2.aruco.CharucoDetector(board)
        try:
            charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)
        except cv2.error:
            # OpenCV 5 can occasionally invalidate a long-lived Python wrapper.
            # Recreate it once; a second failure is a real image/OpenCV error.
            charuco_detector = cv2.aruco.CharucoDetector(board)
            try:
                charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)
            except cv2.error as error:
                raise RuntimeError(f"OpenCV failed while detecting this ChArUco image: {error}") from error
    else:
        parameters = create_detector_parameters()
        if hasattr(cv2.aruco, "ArucoDetector"):
            detector = cv2.aruco.ArucoDetector(dictionary, parameters)
            marker_corners, marker_ids, _ = detector.detectMarkers(gray)
        else:
            marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)
        if marker_ids is None or len(marker_ids) == 0:
            return TargetDetection(False, None, marker_ids, marker_corners, marker_ids)
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, gray, board)
    if charuco_corners is not None:
        charuco_corners = np.asarray(charuco_corners, dtype=np.float32).reshape(-1, 1, 2)
    if charuco_ids is not None:
        charuco_ids = np.asarray(charuco_ids, dtype=np.int32).reshape(-1, 1)
    if marker_ids is not None:
        marker_ids = np.asarray(marker_ids, dtype=np.int32).reshape(-1, 1)
    found = charuco_ids is not None and charuco_corners is not None and len(charuco_ids) >= min_corners
    return TargetDetection(found, charuco_corners if found else None, charuco_ids, marker_corners, marker_ids)


def calibrate_charuco(
    corners_by_image: Sequence[np.ndarray], ids_by_image: Sequence[np.ndarray], board: Any, image_size: Tuple[int, int]
) -> Tuple[float, np.ndarray, np.ndarray, list, list]:
    """Calibrate from ChArUco corners on both OpenCV 4.x and 5.x."""
    if hasattr(cv2.aruco, "calibrateCameraCharuco"):
        return cv2.aruco.calibrateCameraCharuco(corners_by_image, ids_by_image, board, image_size, None, None)
    object_points = [board_object_points(board, ids) for ids in ids_by_image]
    return cv2.calibrateCamera(object_points, list(corners_by_image), image_size, None, None)


def detect_target(frame: np.ndarray, target: dict) -> TargetDetection:
    if target["type"] == "charuco":
        return find_charuco(
            frame, target["board"], target["dictionary"], target["min_corners"], target.get("detector")
        )
    return find_chessboard(frame, target["board_size"])


def build_target_from_args(args: argparse.Namespace) -> dict:
    if args.board_type == "charuco":
        if args.min_charuco_corners < 4:
            raise ValueError("--min-charuco-corners must be at least 4")
        board, dictionary = make_charuco_board(
            args.charuco_squares, args.square_size, args.marker_size, args.aruco_dict, args.charuco_legacy_pattern
        )
        return {
            "type": "charuco",
            "charuco_squares": args.charuco_squares,
            "square_size": args.square_size,
            "marker_size": args.marker_size,
            "aruco_dict": args.aruco_dict,
            "legacy_pattern": args.charuco_legacy_pattern,
            "min_corners": args.min_charuco_corners,
            "board": board,
            "dictionary": dictionary,
            "detector": cv2.aruco.CharucoDetector(board) if hasattr(cv2.aruco, "CharucoDetector") else None,
        }
    return {
        "type": "chessboard",
        "board_size": args.board_size,
        "square_size": args.square_size,
    }


def build_target_from_manifest(manifest: dict) -> dict:
    board_type = manifest.get("board_type", "chessboard")
    if board_type == "charuco":
        board, dictionary = make_charuco_board(
            tuple(manifest["charuco_squares"]),
            float(manifest["square_size"]),
            float(manifest["marker_size"]),
            manifest["aruco_dict"],
            bool(manifest.get("charuco_legacy_pattern", False)),
        )
        return {
            "type": "charuco",
            "charuco_squares": tuple(manifest["charuco_squares"]),
            "square_size": float(manifest["square_size"]),
            "marker_size": float(manifest["marker_size"]),
            "aruco_dict": manifest["aruco_dict"],
            "legacy_pattern": bool(manifest.get("charuco_legacy_pattern", False)),
            "min_corners": int(manifest.get("min_charuco_corners", DEFAULT_MIN_CHARUCO_CORNERS)),
            "board": board,
            "dictionary": dictionary,
            "detector": cv2.aruco.CharucoDetector(board) if hasattr(cv2.aruco, "CharucoDetector") else None,
        }
    return {
        "type": "chessboard",
        "board_size": tuple(manifest["board_size"]),
        "square_size": float(manifest["square_size"]),
    }


def chessboard_corner_variants(corners: np.ndarray, board_size: Tuple[int, int]) -> Dict[str, np.ndarray]:
    columns, rows = board_size
    grid = corners.reshape(rows, columns, 1, 2)
    return {
        "normal": grid.copy().reshape(-1, 1, 2),
        "reverse": grid[::-1, ::-1].copy().reshape(-1, 1, 2),
        "flip_x": grid[:, ::-1].copy().reshape(-1, 1, 2),
        "flip_y": grid[::-1, :].copy().reshape(-1, 1, 2),
    }


def draw_status(
    frame: np.ndarray,
    camera_index: int,
    sample_count: int,
    detection: TargetDetection,
    target: dict,
) -> np.ndarray:
    preview = frame.copy()
    if target["type"] == "charuco":
        if detection.marker_corners is not None:
            cv2.aruco.drawDetectedMarkers(preview, detection.marker_corners, detection.marker_ids)
        if detection.corners is not None and detection.ids is not None:
            cv2.aruco.drawDetectedCornersCharuco(preview, detection.corners, detection.ids)
        found_text = f"charuco {len(detection.ids) if detection.ids is not None else 0}"
    else:
        if detection.corners is not None:
            cv2.drawChessboardCorners(preview, target["board_size"], detection.corners, detection.found)
        found_text = "board OK"
    color = (40, 220, 40) if detection.found else (40, 40, 230)
    text = f"cam {camera_index} | samples {sample_count} | {found_text if detection.found else 'no board'}"
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
    indices = args.indices or [
        camera.index for camera in detect_cameras(args.max_index, args.width, args.height, args.fps, args.fourcc) if camera.opened
    ]
    if len(indices) != args.camera_count:
        raise SystemExit(f"Expected exactly {args.camera_count} camera indices, got {indices}")

    target = build_target_from_args(args)
    captures = open_camera_set(indices, args.width, args.height, args.fps, args.fourcc)
    samples: List[dict] = []
    per_camera_counts = {index: 0 for index in indices}
    last_auto_capture = 0.0

    print("Controls: SPACE=save valid set, A=toggle auto capture, Q/ESC=quit")
    print(f"A sample is saved only when every configured camera sees the {target['type']} board.")
    auto_capture = args.auto

    try:
        while True:
            frames = read_frames(captures)
            detections = {index: detect_target(frame, target) for index, frame in frames.items()}
            all_found = all(detection.found for detection in detections.values())
            previews = [
                draw_status(frames[index], index, per_camera_counts[index], detections[index], target)
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
                    missing = [str(index) for index, detection in detections.items() if not detection.found]
                    print(f"Skipped: {target['type']} was not detected by camera(s): {', '.join(missing)}.")
                    continue
                sample_id = len(samples)
                sample_record = {"sample_id": sample_id, "time": now, "cameras": {}}
                for index in indices:
                    image_name = f"sample_{sample_id:04d}_cam_{index}.png"
                    preview_name = f"sample_{sample_id:04d}_cam_{index}_corners.png"
                    image_path = dirs["images"] / image_name
                    preview_path = dirs["previews"] / preview_name
                    cv2.imwrite(str(image_path), frames[index])
                    cv2.imwrite(str(preview_path), draw_status(frames[index], index, sample_id + 1, detections[index], target))
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
        "board_type": target["type"],
        "square_size": args.square_size,
        "samples": samples,
    }
    if target["type"] == "charuco":
        manifest.update(
            {
                "charuco_squares": list(args.charuco_squares),
                "marker_size": args.marker_size,
                "aruco_dict": args.aruco_dict,
                "charuco_legacy_pattern": args.charuco_legacy_pattern,
                "min_charuco_corners": args.min_charuco_corners,
            }
        )
    else:
        object_points = chessboard_object_points(args.board_size, args.square_size)
        manifest.update({"board_size": args.board_size, "object_points": object_points.tolist()})
    manifest_path = dirs["root"] / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")


def load_manifest(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_image_path(manifest_path: Path, image_path: str) -> Path:
    path = Path(image_path)
    if path.is_absolute():
        return path
    output_root = manifest_path.parent
    candidate = output_root / path
    if candidate.exists():
        return candidate
    return output_root.parent.parent / path


def calibrate_intrinsics(manifest: dict, manifest_path: Path) -> Tuple[dict, Tuple[int, int]]:
    target = build_target_from_manifest(manifest)
    object_points_template = chessboard_object_points(target["board_size"], target["square_size"]) if target["type"] == "chessboard" else None
    camera_indices = [str(index) for index in manifest["camera_indices"]]

    object_points_by_camera: Dict[str, list] = {index: [] for index in camera_indices}
    image_points_by_camera: Dict[str, list] = {index: [] for index in camera_indices}
    charuco_corners_by_camera: Dict[str, list] = {index: [] for index in camera_indices}
    charuco_ids_by_camera: Dict[str, list] = {index: [] for index in camera_indices}
    image_size: Optional[Tuple[int, int]] = None

    for sample in manifest["samples"]:
        for camera_index in camera_indices:
            image_path = resolve_image_path(manifest_path, sample["cameras"][camera_index]["image"])
            frame = cv2.imread(str(image_path))
            if frame is None:
                print(f"Warning: could not read {image_path}")
                continue
            image_size = (frame.shape[1], frame.shape[0])
            detection = detect_target(frame, target)
            if detection.found and detection.corners is not None:
                if target["type"] == "charuco" and detection.ids is not None:
                    charuco_corners_by_camera[camera_index].append(detection.corners)
                    charuco_ids_by_camera[camera_index].append(detection.ids)
                elif object_points_template is not None:
                    object_points_by_camera[camera_index].append(object_points_template)
                    image_points_by_camera[camera_index].append(detection.corners)

    if image_size is None:
        raise RuntimeError("no calibration images could be read")

    results = {}
    for camera_index in camera_indices:
        if target["type"] == "charuco":
            valid_samples = len(charuco_corners_by_camera[camera_index])
            if valid_samples < 8:
                raise RuntimeError(f"camera {camera_index} has only {valid_samples} valid ChArUco samples; collect at least 8")
            rms, matrix, distortion, rvecs, tvecs = calibrate_charuco(
                charuco_corners_by_camera[camera_index],
                charuco_ids_by_camera[camera_index],
                target["board"],
                image_size,
            )
        else:
            valid_samples = len(image_points_by_camera[camera_index])
            if valid_samples < 8:
                raise RuntimeError(
                    f"camera {camera_index} has only {valid_samples} valid samples; collect at least 8"
                )
            rms, matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
                object_points_by_camera[camera_index], image_points_by_camera[camera_index], image_size, None, None
            )
        results[camera_index] = {
            "rms": float(rms),
            "camera_matrix": matrix.tolist(),
            "distortion_coefficients": distortion.tolist(),
            "valid_samples": valid_samples,
            "rvecs": [rvec.tolist() for rvec in rvecs],
            "tvecs": [tvec.tolist() for tvec in tvecs],
        }
        print(f"Intrinsic cam {camera_index}: RMS={rms:.4f}, samples={valid_samples}")

    return results, image_size


def calibrate_stereo_pairs(
    manifest: dict,
    manifest_path: Path,
    intrinsics: dict,
    image_size: Tuple[int, int],
    reference_camera: str,
) -> dict:
    target = build_target_from_manifest(manifest)
    object_points_template = (
        chessboard_object_points(target["board_size"], target["square_size"])
        if target["type"] == "chessboard"
        else None
    )
    stereo_results = {}

    for camera_index in (str(index) for index in manifest["camera_indices"]):
        if camera_index == reference_camera:
            continue
        object_points: list = []
        reference_points: list = []
        target_points_by_variant: Dict[str, list] = {"normal": [], "reverse": [], "flip_x": [], "flip_y": []}
        for sample in manifest["samples"]:
            ref_frame = cv2.imread(str(resolve_image_path(manifest_path, sample["cameras"][reference_camera]["image"])))
            cam_frame = cv2.imread(str(resolve_image_path(manifest_path, sample["cameras"][camera_index]["image"])))
            if ref_frame is None or cam_frame is None:
                continue
            reference_detection = detect_target(ref_frame, target)
            camera_detection = detect_target(cam_frame, target)
            if target["type"] == "charuco":
                sample_object_points, sample_reference_points, sample_target_points = matched_charuco_points(
                    reference_detection, camera_detection, target["board"]
                )
                if len(sample_object_points) < 4:
                    continue
                object_points.append(sample_object_points)
                reference_points.append(sample_reference_points)
                target_points_by_variant["normal"].append(sample_target_points)
            elif reference_detection.found and camera_detection.found and object_points_template is not None:
                object_points.append(object_points_template)
                reference_points.append(reference_detection.corners)
                for name, corners in chessboard_corner_variants(camera_detection.corners, target["board_size"]).items():
                    target_points_by_variant[name].append(corners)

        if len(object_points) < 8:
            print(f"Skipping stereo {reference_camera}->{camera_index}: only {len(object_points)} shared samples")
            continue

        ref_intrinsic = np.array(intrinsics[reference_camera]["camera_matrix"], dtype=np.float64)
        ref_distortion = np.array(intrinsics[reference_camera]["distortion_coefficients"], dtype=np.float64)
        cam_intrinsic = np.array(intrinsics[camera_index]["camera_matrix"], dtype=np.float64)
        cam_distortion = np.array(intrinsics[camera_index]["distortion_coefficients"], dtype=np.float64)
        best_result = None
        for corner_order, target_points in target_points_by_variant.items():
            if len(target_points) != len(object_points):
                continue
            rms, _, _, _, _, rotation, translation, essential, fundamental = cv2.stereoCalibrate(
                object_points, reference_points, target_points, ref_intrinsic, ref_distortion,
                cam_intrinsic, cam_distortion, image_size, flags=cv2.CALIB_FIX_INTRINSIC,
            )
            if best_result is None or rms < best_result["rms"]:
                best_result = {"corner_order": corner_order, "rms": float(rms), "rotation": rotation,
                               "translation": translation, "essential": essential, "fundamental": fundamental}
        if best_result is None:
            print(f"Skipping stereo {reference_camera}->{camera_index}: no matched target points")
            continue
        stereo_results[camera_index] = {
            "reference_camera": reference_camera, "target_camera": camera_index, "rms": best_result["rms"],
            "corner_order": best_result["corner_order"], "rotation_reference_to_target": best_result["rotation"].tolist(),
            "translation_reference_to_target": best_result["translation"].tolist(),
            "essential_matrix": best_result["essential"].tolist(), "fundamental_matrix": best_result["fundamental"].tolist(),
            "shared_samples": len(object_points),
        }
        print(f"Stereo {reference_camera}->{camera_index}: RMS={best_result['rms']:.4f}, samples={len(object_points)}, corner_order={best_result['corner_order']}")
    return stereo_results


def matched_charuco_points(
    reference_detection: TargetDetection, target_detection: TargetDetection, board: Any
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if (
        reference_detection.corners is None
        or reference_detection.ids is None
        or target_detection.corners is None
        or target_detection.ids is None
    ):
        return np.empty((0, 3), np.float32), np.empty((0, 1, 2), np.float32), np.empty((0, 1, 2), np.float32)
    reference_by_id = {
        int(charuco_id): corner for charuco_id, corner in zip(reference_detection.ids.flatten(), reference_detection.corners)
    }
    target_by_id = {int(charuco_id): corner for charuco_id, corner in zip(target_detection.ids.flatten(), target_detection.corners)}
    common_ids = sorted(set(reference_by_id).intersection(target_by_id))
    if not common_ids:
        return np.empty((0, 3), np.float32), np.empty((0, 1, 2), np.float32), np.empty((0, 1, 2), np.float32)
    ids = np.array(common_ids, dtype=np.int32).reshape(-1, 1)
    return (
        board_object_points(board, ids),
        np.array([reference_by_id[charuco_id] for charuco_id in common_ids], dtype=np.float32),
        np.array([target_by_id[charuco_id] for charuco_id in common_ids], dtype=np.float32),
    )
def run_calibration(args: argparse.Namespace) -> None:
    output_root = Path(args.output)
    dirs = make_output_dirs(output_root)
    manifest_path = output_root / "manifest.json"
    manifest = load_manifest(manifest_path)
    intrinsics, image_size = calibrate_intrinsics(manifest, manifest_path)
    reference_camera = str(args.reference)
    if reference_camera not in intrinsics:
        raise RuntimeError(f"reference camera {reference_camera} was not captured")
    stereo = calibrate_stereo_pairs(manifest, manifest_path, intrinsics, image_size, reference_camera)
    results = {
        "image_size": image_size,
        "board_type": manifest.get("board_type", "chessboard"),
        "square_size": manifest["square_size"],
        "reference_camera": reference_camera,
        "intrinsics": intrinsics,
        "extrinsics_relative_to_reference": stereo,
    }
    if manifest.get("board_type", "chessboard") == "charuco":
        results.update({key: manifest[key] for key in ("charuco_squares", "marker_size", "aruco_dict")})
        results["charuco_legacy_pattern"] = bool(manifest.get("charuco_legacy_pattern", False))
    else:
        results["board_size"] = manifest["board_size"]
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
    common.add_argument("--fourcc", default=DEFAULT_FOURCC, help="Camera pixel format, usually MJPG or YUY2")
    common.add_argument("--max-index", type=int, default=16)

    detect = subparsers.add_parser("detect", parents=[common], help="List available camera device indices.")
    detect.set_defaults(
        func=lambda args: print_camera_report(
            detect_cameras(args.max_index, args.width, args.height, args.fps, args.fourcc), args.camera_count
        )
    )

    capture = subparsers.add_parser("capture", parents=[common], help="Capture synchronized chessboard or ChArUco images.")
    capture.add_argument("--indices", type=parse_indices, help="Comma-separated camera indices, for example 0,1,2,3,4,5,6,7")
    capture.add_argument("--board-size", type=parse_board_size, default=DEFAULT_BOARD_SIZE, help="Inner corners, e.g. 9x6")
    capture.add_argument("--square-size", type=float, default=DEFAULT_SQUARE_SIZE, help="Chessboard square size in meters")
    capture.add_argument("--board-type", choices=("chessboard", "charuco"), default=DEFAULT_BOARD_TYPE)
    capture.add_argument("--charuco-squares", type=parse_board_size, default=DEFAULT_CHARUCO_SQUARES,
                         help="ChArUco square count, e.g. 9x7 (not inner corners)")
    capture.add_argument("--marker-size", type=float, default=DEFAULT_MARKER_SIZE,
                         help="ChArUco marker side length in meters; must be smaller than --square-size")
    capture.add_argument("--aruco-dict", default=DEFAULT_ARUCO_DICT, help="OpenCV ArUco dictionary, e.g. DICT_4X4_50")
    capture.add_argument("--charuco-legacy-pattern", action="store_true",
                         help="Use the pre-OpenCV-4.6 ChArUco layout (even row count starts with a white upper-left square)")
    capture.add_argument("--min-charuco-corners", type=int, default=DEFAULT_MIN_CHARUCO_CORNERS,
                         help="Minimum interpolated ChArUco corners required per camera")
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
