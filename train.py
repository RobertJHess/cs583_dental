#!/usr/bin/env python3
"""
Training script for YOLOv8 dental X-ray segmentation model.
Downloads the dataset from Roboflow and trains the model.
"""

import argparse
import os
from roboflow import Roboflow
from ultralytics import YOLO


def download_dataset(api_key="szzYe8PLeSgGZR4e6h7C", download_location="data/opg"):
    """Download the dental X-ray dataset from Roboflow."""
    print("Downloading dataset from Roboflow...")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("detect-inpnn").project("opg-detect")
    dataset = project.version(1).download("yolov8", location=download_location)
    print(dataset)
    print(f"Dataset downloaded to: {dataset.location}")
    return dataset


def train_model(
    data_yaml="opg.yaml",
    model_name="yolov8s-seg.pt",
    epochs=25,
    image_size=512,
    batch_size=2,
    device="cpu",
    project="runs/segment",
    name="opg_dental"
):
    print(f"Loading model: {model_name}")
    model = YOLO(model_name)
    print(f"Starting training for {epochs} epochs...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        device=device,
        project=project,
        name=name,
        patience=50,
        save=True,
        save_period=10,
        cache=False,
        plots=True
    )
    print("Training completed!")
    print(f"Best model saved at: {model.trainer.best}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 segmentation model on dental X-ray dataset"
    )
    parser.add_argument(
        "--download-dataset",
        action="store_true",
        help="Download dataset from Roboflow before training"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="szzYe8PLeSgGZR4e6h7C",
        help="Roboflow API key"
    )
    parser.add_argument(
        "--download_location",
        type=str,
        default="data/opg",
        help="Dataset download location"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="opg.yaml",
        help="Path to dataset YAML file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8s-seg.pt",
        help="YOLOv8 model to use (yolov8n-seg.pt, yolov8s-seg.pt, yolov8m-seg.pt, etc.)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Input image size"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=2,
        help="Batch size"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="CUDA device (0, 1, 2, etc.) or 'cpu'"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/segment",
        help="Project directory"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="opg_dental",
        help="Experiment name"
    )
    args = parser.parse_args()
    if args.download_dataset:
        download_dataset(api_key=args.api_key, download_location=args.download_location)
    results = train_model(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        image_size=args.image_size,
        batch_size=args.batch,
        device=args.device,
        project=args.project,
        name=args.name
    )
    print("\n" + "="*50)
    print("Training Summary")
    print("="*50)
    print(f"Results directory: {args.project}/{args.name}")


if __name__ == "__main__":
    main()
