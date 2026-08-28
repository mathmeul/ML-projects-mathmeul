"""Comms.py — open the iPhone (Iriun) feed instead of the built-in webcam.

Usage:
    python Comms.py            # auto-detect Iriun, else ask which camera to use
    python Comms.py 2          # force camera index 2
    python Comms.py --list     # only list the cameras that were found
"""
from __future__ import annotations

import platform
import sys

import cv2

IS_WINDOWS = platform.system() == "Windows"
# DSHOW first: Iriun's virtual camera is a DirectShow device and MSMF often
# fails to open it. MSMF is kept as a fallback for cameras DSHOW misses.
BACKENDS = [cv2.CAP_DSHOW, cv2.CAP_MSMF] if IS_WINDOWS else [cv2.CAP_ANY]


def device_names() -> list[str]:
    """DirectShow device names, index-aligned. Empty if unavailable."""
    if not IS_WINDOWS:
        return []
    try:
        from pygrabber.dshow_graph import FilterGraph
    except ImportError:
        return []
    try:
        return FilterGraph().get_input_devices()
    except Exception:
        return []


def open_camera(index: int) -> cv2.VideoCapture | None:
    """Return an opened capture that actually delivers a frame, or None."""
    for backend in BACKENDS:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened() and cap.read()[0]:
            return cap
        cap.release()
    return None


def scan(max_index: int = 8) -> list[tuple[int, str]]:
    """Find working camera indices with their name (if we can resolve one)."""
    names = device_names()
    found: list[tuple[int, str]] = []
    for index in range(max_index):
        cap = open_camera(index)
        if cap is None:
            continue
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        name = names[index] if index < len(names) else f"camera {index}"
        found.append((index, f"{name} ({width}x{height})"))
        print(f"index {index}: {found[-1][1]}")
    return found


def iriun_index() -> int | None:
    """Index of the Iriun virtual camera, by device name."""
    for index, name in enumerate(device_names()):
        if "iriun" in name.lower():
            return index
    return None


def pick_index(cameras: list[tuple[int, str]]) -> int | None:
    """Prefer an Iriun device; otherwise let the user choose."""
    for index, label in cameras:
        if "iriun" in label.lower():
            print(f"Iriun found on index {index}.")
            return index

    if not device_names():
        print(
            "\nCould not read camera names (install pygrabber for automatic "
            "Iriun detection: pip install pygrabber)."
        )
    if len(cameras) == 1:
        print("Only one camera found; it is probably the built-in webcam.")

    print("\nWhich index is the iPhone feed?")
    answer = input(f"index {[i for i, _ in cameras]} (Enter to cancel): ").strip()
    return int(answer) if answer.isdigit() else None


def main() -> None:
    args = sys.argv[1:]

    index = int(args[0]) if args and args[0].isdigit() else None

    # Fast path: resolve Iriun by name, so we never grab the built-in webcam.
    if index is None and "--list" not in args:
        index = iriun_index()
        if index is not None:
            print(f"Iriun found on index {index}.")

    if index is None:
        cameras = scan()
        if not cameras:
            raise SystemExit(
                "No working camera found — check that Iriun Webcam Server "
                "is running on the PC and the app is open on the phone."
            )
        if "--list" in args:
            return
        index = pick_index(cameras)
        if index is None:
            raise SystemExit("No camera selected.")

    cap = open_camera(index)
    if cap is None:
        raise SystemExit(f"Could not open camera index {index}.")

    print(f"Using camera index {index}. Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame grab failed.")
            break
        cv2.imshow("iPhone feed (Iriun)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
