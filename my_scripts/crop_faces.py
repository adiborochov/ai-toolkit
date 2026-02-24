#!/usr/bin/env python3
"""
Crop before/after image pairs around the detected face (including beard).
Uses MediaPipe for face detection on the 'before' image, expands the box
to include the beard, forces 1888:2208 aspect ratio, then applies the
identical crop to both before and after. Resizes to 1888x2208.

Usage:
    python crop_faces.py <before_dir> <after_dir>

Output is written to <before_dir>_cropped/ and <after_dir>_cropped/
alongside the input folders.
"""

import os
import sys
import mediapipe as mp
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "blaze_face_short_range.tflite")
TARGET_W = 1888
TARGET_H = 2208
TARGET_RATIO = TARGET_W / TARGET_H  # ~0.8551


def create_detector(min_confidence=0.5):
    """Create a MediaPipe FaceDetector with the tasks API."""
    base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp.tasks.vision.FaceDetectorOptions(
        base_options=base_options,
        min_detection_confidence=min_confidence,
    )
    return mp.tasks.vision.FaceDetector.create_from_options(options)


def detect_face(image_path):
    """Detect face bounding box using MediaPipe. Returns (x, y, w, h) in pixels or None."""
    mp_image = mp.Image.create_from_file(image_path)
    img_w = mp_image.width
    img_h = mp_image.height

    detector = create_detector(0.5)
    result = detector.detect(mp_image)
    detector.close()

    if not result.detections:
        detector = create_detector(0.3)
        result = detector.detect(mp_image)
        detector.close()

    if not result.detections:
        return None

    # Take the detection with highest confidence
    best = max(result.detections, key=lambda d: d.categories[0].score)
    bbox = best.bounding_box

    fx = bbox.origin_x
    fy = bbox.origin_y
    fw = bbox.width
    fh = bbox.height

    return fx, fy, fw, fh


def compute_crop_box(face_box, img_w, img_h):
    """
    Given a face bounding box (x, y, w, h), expand it to include the beard
    and surrounding context, then adjust to TARGET_RATIO aspect ratio.
    Returns (left, top, right, bottom) clamped to image bounds.
    """
    fx, fy, fw, fh = face_box

    # Face center
    cx = fx + fw / 2
    cy = fy + fh / 2

    # Expand: 30% padding on top (headroom), 70% downward (beard), 40% on each side
    top_pad = fh * 0.50
    bottom_pad = fh * 0.60
    side_pad = fw * 0.50

    left = cx - fw / 2 - side_pad
    top = fy - top_pad
    right = cx + fw / 2 + side_pad
    bottom = fy + fh + bottom_pad

    # Current crop dimensions
    crop_w = right - left
    crop_h = bottom - top
    crop_cx = (left + right) / 2
    crop_cy = (top + bottom) / 2

    # Adjust to target aspect ratio
    current_ratio = crop_w / crop_h
    if current_ratio > TARGET_RATIO:
        # Too wide -> increase height
        new_h = crop_w / TARGET_RATIO
        crop_cy = crop_cy  # keep centered
        top = crop_cy - new_h / 2
        bottom = crop_cy + new_h / 2
    else:
        # Too tall -> increase width
        new_w = crop_h * TARGET_RATIO
        left = crop_cx - new_w / 2
        right = crop_cx + new_w / 2

    # Round to integers
    left = int(round(left))
    top = int(round(top))
    right = int(round(right))
    bottom = int(round(bottom))

    # Shift into bounds if needed (prefer shifting over clamping to preserve ratio)
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > img_w:
        left -= (right - img_w)
        right = img_w
    if bottom > img_h:
        top -= (bottom - img_h)
        bottom = img_h

    # Final clamp (in case image is smaller than crop)
    left = max(0, left)
    top = max(0, top)
    right = min(img_w, right)
    bottom = min(img_h, bottom)

    return left, top, right, bottom


def crop_and_resize(img_path, box, output_path):
    """Crop the image to `box`, resize to TARGET dimensions preserving aspect ratio."""
    img = Image.open(img_path).convert("RGB")
    left, top, right, bottom = box
    cropped = img.crop((left, top, right, bottom))

    # Resize to target. Since we forced the aspect ratio, this should be clean.
    # Use LANCZOS for high-quality downsampling.
    crop_w = right - left
    crop_h = bottom - top
    crop_ratio = crop_w / crop_h

    if abs(crop_ratio - TARGET_RATIO) < 0.001:
        # Aspect ratio matches, direct resize
        result = cropped.resize((TARGET_W, TARGET_H), Image.LANCZOS)
    else:
        # Slight mismatch from clamping - fit inside target and pad
        cropped.thumbnail((TARGET_W, TARGET_H), Image.LANCZOS)
        result = Image.new("RGB", (TARGET_W, TARGET_H), (0, 0, 0))
        paste_x = (TARGET_W - cropped.width) // 2
        paste_y = (TARGET_H - cropped.height) // 2
        result.paste(cropped, (paste_x, paste_y))

    result.save(output_path, quality=95)


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <before_dir> <after_dir>")
        sys.exit(1)

    before_dir = os.path.abspath(sys.argv[1])
    after_dir = os.path.abspath(sys.argv[2])

    if not os.path.isdir(before_dir):
        print(f"Error: before directory not found: {before_dir}")
        sys.exit(1)
    if not os.path.isdir(after_dir):
        print(f"Error: after directory not found: {after_dir}")
        sys.exit(1)

    before_out = before_dir + "_cropped"
    after_out = after_dir + "_cropped"

    os.makedirs(before_out, exist_ok=True)
    os.makedirs(after_out, exist_ok=True)

    print(f"Before:  {before_dir}")
    print(f"After:   {after_dir}")
    print(f"Output:  {before_out}")
    print(f"         {after_out}\n")

    before_files = sorted(
        f for f in os.listdir(before_dir) if not f.startswith(".")
    )

    failed = []
    for fname in before_files:
        before_path = os.path.join(before_dir, fname)
        after_path = os.path.join(after_dir, fname)

        if not os.path.exists(after_path):
            print(f"  SKIP {fname}: no matching after image")
            continue

        print(f"Processing {fname}...", end=" ")

        # Get image dimensions
        img = Image.open(before_path)
        img_w, img_h = img.size

        # Detect face in before image
        face_box = detect_face(before_path)
        if face_box is None:
            print("FAILED - no face detected")
            failed.append(fname)
            continue

        fx, fy, fw, fh = face_box
        print(f"face at ({fx},{fy}) {fw}x{fh}", end=" -> ")

        # Compute crop box
        box = compute_crop_box(face_box, img_w, img_h)
        left, top, right, bottom = box
        print(f"crop ({left},{top})-({right},{bottom}) {right-left}x{bottom-top}")

        # Crop and resize both
        crop_and_resize(before_path, box, os.path.join(before_out, fname))
        crop_and_resize(after_path, box, os.path.join(after_out, fname))

    if failed:
        print(f"\nFailed to detect faces in: {', '.join(failed)}")
        print("These images were skipped. Consider manual cropping.")
    else:
        print(f"\nAll {len(before_files)} pairs processed successfully!")


if __name__ == "__main__":
    main()
