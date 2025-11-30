#!/usr/bin/env python3

import argparse
import os
from roboflow import Roboflow
from ultralytics import YOLO


def download_dataset(api_key="szzYe8PLeSgGZR4e6h7C", download_location="data/opg"):
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
        help="YOLOv8 model to use."
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
        help="CUDA device 0 or 'cpu'"
    )
    args = parser.parse_args()
    if args.download_dataset:
        download_dataset(api_key=args.api_key, download_location=args.download_location)
    train_model(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        image_size=args.image_size,
        batch_size=args.batch,
        device=args.device
    )
    print("\n" + "="*50)
    print("Training Summary")
    print("="*50)
    print("Results directory: runs/segment/")


if __name__ == "__main__":
    main()
