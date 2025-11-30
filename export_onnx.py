#!/usr/bin/env python3

import sys
import os
from pathlib import Path
from ultralytics import YOLO


def export_to_onnx(model_path):
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    print(f"Exporting model to ONNX format.")
    onnx_path = model.export(format="onnx")
    print(f"\nExport completed successfully!")
    print(f"ONNX model saved to: {onnx_path}")
    return onnx_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_model.pt>")
        return

    model_path = sys.argv[1]

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    return export_to_onnx(model_path=model_path)



if __name__ == "__main__":
    onnx_path = main()
    print("\n" + "="*50)
    print("Export Summary")
    print("="*50)
    print(f"ONNX: {onnx_path}")

