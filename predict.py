#!/usr/bin/env python3
"""
Prediction script for YOLOv8 dental X-ray segmentation model.
Runs inference on dental X-ray images and saves results.
Used to validate training before using the Rust inference.
"""

import argparse
import os
from pathlib import Path
from ultralytics import YOLO


def predict(
    model_path,
    source,
    conf=0.25,
    iou=0.7,
    image_size=512,
    save=True,
    save_txt=False,
    save_conf=False,
    save_crop=False,
    show_labels=True,
    show_conf=True,
    show_boxes=True,
    project="runs/predict",
    name="opg_predictions",
    visualize=False
):
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    print(f"Running prediction on: {source}")
    results = model.predict(
        source=source,
        conf=conf,
        iou=iou,
        imgsz=image_size,
        save=save,
        save_txt=save_txt,
        save_conf=save_conf,
        save_crop=save_crop,
        show_labels=show_labels,
        show_conf=show_conf,
        show_boxes=show_boxes,
        project=project,
        name=name,
        visualize=visualize,
        stream=False
    )
    print(f"\nPrediction completed!")
    print(f"Results saved to: {project}/{name}")
    print("\n" + "="*50)
    print("Detection Summary")
    print("="*50)
    for index, result in enumerate(results):
        if result.masks is not None:
            num_detections = len(result.masks)
            print(f"Image {index + 1}: {num_detections} objects detected")
            if result.boxes is not None and len(result.boxes) > 0:
                classes = result.boxes.cls.cpu().numpy()
                names = result.names
                detected_classes = {}
                for cls in classes:
                    class_name = names[int(cls)]
                    detected_classes[class_name] = detected_classes.get(class_name, 0) + 1
                for class_name, count in detected_classes.items():
                    print(f"  - {class_name}: {count}")
        else:
            print(f"Image {index + 1}: No objects detected")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run YOLOv8 segmentation predictions on dental X-ray images"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model weights (e.g., runs/segment/opg_dental/weights/best.pt)"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to image, directory"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (0.0-1.0)"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="IoU threshold for NMS (0.0-1.0)"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Input image size"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save prediction results"
    )
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Save results as text files"
    )
    parser.add_argument(
        "--save-conf",
        action="store_true",
        help="Save confidence scores in text files"
    )
    parser.add_argument(
        "--save-crop",
        action="store_true",
        help="Save cropped predictions"
    )
    parser.add_argument(
        "--hide-labels",
        action="store_true",
        help="Hide labels on predictions"
    )
    parser.add_argument(
        "--hide-conf",
        action="store_true",
        help="Hide confidence scores on predictions"
    )
    parser.add_argument(
        "--hide-boxes",
        action="store_true",
        help="Hide bounding boxes"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/predict",
        help="Project directory"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="opg_predictions",
        help="Experiment name"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize model features"
    )
    args = parser.parse_args()
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        print("Please provide a valid path to trained model weights.")
        return
    results = predict(
        model_path=args.model,
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        image_size=args.image_size,
        save=not args.no_save,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        save_crop=args.save_crop,
        show_labels=not args.hide_labels,
        show_conf=not args.hide_conf,
        show_boxes=not args.hide_boxes,
        project=args.project,
        name=args.name,
        visualize=args.visualize
    )


if __name__ == "__main__":
    main()
    