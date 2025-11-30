#!/usr/bin/env python3
"""
Export script for YOLOv8 dental X-ray segmentation model.
Exports trained model to ONNX format for deployment.
"""

import argparse
import os
from pathlib import Path
from ultralytics import YOLO


def export_to_onnx(
    model_path,
    imgsz=640,
    dynamic=False,
    simplify=False,
    opset=12,
    half=False
):
    """
    Export YOLOv8 model to ONNX format.

    Args:
        model_path: Path to trained model weights
        imgsz: Input image size
        dynamic: Enable dynamic input shapes
        simplify: Simplify ONNX model
        opset: ONNX opset version
        half: Export in FP16 half precision

    Returns:
        Path to exported ONNX model
    """
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    print(f"Exporting model to ONNX format...")
    print(f"  Image size: {imgsz}")
    print(f"  Dynamic shapes: {dynamic}")
    print(f"  Simplify: {simplify}")
    print(f"  ONNX opset: {opset}")
    print(f"  Half precision: {half}")

    onnx_path = model.export(
        format="onnx",
        imgsz=imgsz,
        dynamic=dynamic,
        simplify=simplify,
        opset=opset,
        half=half
    )

    print(f"\nExport completed successfully!")
    print(f"ONNX model saved to: {onnx_path}")

    return onnx_path


def export_multiple_formats(
    model_path,
    formats=None,
    imgsz=640,
    half=False
):
    """
    Export model to multiple formats.

    Args:
        model_path: Path to trained model weights
        formats: List of export formats (onnx, torchscript, coreml, etc.)
        imgsz: Input image size
        half: Export in FP16 half precision
    """
    if formats is None:
        formats = ["onnx"]

    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    exported_paths = {}

    for fmt in formats:
        print(f"\nExporting to {fmt.upper()} format...")
        try:
            exported_path = model.export(
                format=fmt,
                imgsz=imgsz,
                half=half
            )
            exported_paths[fmt] = exported_path
            print(f"{fmt.upper()} export successful: {exported_path}")
        except Exception as e:
            print(f"Error exporting to {fmt}: {str(e)}")

    return exported_paths


def main():
    parser = argparse.ArgumentParser(
        description="Export YOLOv8 segmentation model to ONNX and other formats"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model weights (e.g., runs/segment/opg_dental/weights/best.pt)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size for export"
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Enable dynamic input shapes for ONNX"
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Simplify ONNX model"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=12,
        help="ONNX opset version"
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Export in FP16 half precision"
    )
    parser.add_argument(
        "--formats",
        type=str,
        nargs="+",
        default=["onnx"],
        help="Export formats (onnx, torchscript, coreml, engine, tflite, etc.)"
    )

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        print("Please provide a valid path to trained model weights.")
        return

    # Export based on number of formats
    if len(args.formats) == 1 and args.formats[0] == "onnx":
        # Single ONNX export with full options
        onnx_path = export_to_onnx(
            model_path=args.model,
            imgsz=args.imgsz,
            dynamic=args.dynamic,
            simplify=args.simplify,
            opset=args.opset,
            half=args.half
        )
    else:
        # Multiple format export
        exported_paths = export_multiple_formats(
            model_path=args.model,
            formats=args.formats,
            imgsz=args.imgsz,
            half=args.half
        )

        print("\n" + "="*50)
        print("Export Summary")
        print("="*50)
        for fmt, path in exported_paths.items():
            print(f"{fmt.upper()}: {path}")


if __name__ == "__main__":
    main()
