#!/usr/bin/env python3
"""
Export script for YOLOv8 dental X-ray segmentation model.
Exports trained model to ONNX format for deployment to Rust.
"""

import argparse
import os
from pathlib import Path
from ultralytics import YOLO


def export_to_onnx(
    model_path,
    image_size=512,
    dynamic=False,
    simplify=False,
    opset=12,
    half=False
):
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    print(f"Exporting model to ONNX format...")
    print(f"  Image size: {image_size}")
    print(f"  Dynamic shapes: {dynamic}")
    print(f"  Simplify: {simplify}")
    print(f"  ONNX opset: {opset}")
    print(f"  Half precision: {half}")
    onnx_path = model.export(
        format="onnx",
        imgsz=image_size,
        dynamic=dynamic,
        simplify=simplify,
        opset=opset,
        half=half
    )
    print(f"\nExport completed successfully!")
    print(f"ONNX model saved to: {onnx_path}")
    return onnx_path


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
        "--image-size",
        type=int,
        default=512,
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
        "--format",
        type=str,
        default="onnx",
        help="Export format (onnx)"
    )
    args = parser.parse_args()
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        print("Please provide a valid path to trained model weights.")
        return
    if args.format:
        # ONNX export
        onnx_path = export_to_onnx(
            model_path=args.model,
            image_size=args.image_size,
            dynamic=args.dynamic,
            simplify=args.simplify,
            opset=args.opset,
            half=args.half
        )
        print("\n" + "="*50)
        print("Export Summary")
        print("="*50)
        print(f"{args.format.upper()}: {onnx_path}")


if __name__ == "__main__":
    main()
