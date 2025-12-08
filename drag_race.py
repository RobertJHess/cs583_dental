import os
import glob
import time
import asyncio
from datetime import datetime
from pathlib import Path
import numpy as np
import cv2
import onnxruntime as ort
import yaml
ort.preload_dlls()

def process_yolov8_output(predictions, proto, orig_shape, conf_threshold=0.25, iou_threshold=0.45):
    """
    Process YOLOv8 segmentation output.
    
    Args:
        predictions: np.ndarray shape (1, 60, 5376)
        proto: np.ndarray shape (1, 32, 128, 128)
        orig_shape: tuple (H, W) of original image
        conf_threshold: confidence threshold
        iou_threshold: IoU threshold for NMS
    
    Returns:
        List of detections with masks
    """
    predictions = predictions[0]  # (60, 5376)
    proto = proto[0]  # (32, 128, 128)
    
    num_attrs = predictions.shape[0]
    num_detections = predictions.shape[1]
    
    num_mask_coeffs = 32
    num_classes = num_attrs - 4 - num_mask_coeffs
    
    detections = []
    
    for i in range(num_detections):
        # Extract box (xywh format, normalized to 512x512)
        x, y, w, h = predictions[0:4, i]
        
        # Extract class scores
        class_scores = predictions[4:4+num_classes, i]
        class_id = int(np.argmax(class_scores))
        score = float(class_scores[class_id])
        
        if score < conf_threshold:
            continue
        
        # Convert xywh to xyxy (still in 512x512 space)
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        
        # Extract mask coefficients
        mask_coeffs = predictions[4+num_classes:4+num_classes+32, i]
        
        detections.append({
            'bbox': [x1, y1, x2, y2],
            'score': score,
            'class_id': class_id,
            'mask_coeffs': mask_coeffs
        })
    
    # Apply NMS
    detections = nms(detections, iou_threshold)
    
    # Generate masks and scale to original size
    orig_h, orig_w = orig_shape
    scale_x = orig_w / 512.0
    scale_y = orig_h / 512.0
    
    for det in detections:
        # Scale bbox to original size
        det['bbox'][0] *= scale_x
        det['bbox'][1] *= scale_y
        det['bbox'][2] *= scale_x
        det['bbox'][3] *= scale_y
        
        # Generate mask: mask_coeffs @ proto
        raw_mask = np.einsum('c,chw->hw', det['mask_coeffs'], proto)
        mask = 1 / (1 + np.exp(-raw_mask))  # sigmoid
        
        # Resize mask to original image size
        mask_resized = cv2.resize(mask, (orig_w, orig_h))
        det['mask'] = (mask_resized > 0.5).astype(np.uint8)
    
    return detections


def nms(detections, iou_threshold):
    """Non-maximum suppression"""
    if len(detections) == 0:
        return []
    
    # Sort by score
    detections = sorted(detections, key=lambda x: x['score'], reverse=True)
    
    keep = []
    suppressed = [False] * len(detections)
    
    for i in range(len(detections)):
        if suppressed[i]:
            continue
        
        keep.append(detections[i])
        box_a = detections[i]['bbox']
        
        for j in range(i + 1, len(detections)):
            if suppressed[j]:
                continue
            
            box_b = detections[j]['bbox']
            
            # Calculate IoU
            x1 = max(box_a[0], box_b[0])
            y1 = max(box_a[1], box_b[1])
            x2 = min(box_a[2], box_b[2])
            y2 = min(box_a[3], box_b[3])
            
            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
            area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
            union = area_a + area_b - intersection
            
            iou = intersection / union if union > 0 else 0
            
            if iou > iou_threshold:
                suppressed[j] = True
    
    return keep


def visualize_yolo_segmentation(detections, img):
    """
    Creates an Ultralytics-style segmentation overlay.
    
    Args:
        detections: list of detection dicts with 'bbox', 'score', 'class_id', 'mask'
        img: original image (H,W,3) np.uint8
    
    Returns:
        np.ndarray: colored overlay image
    """
    result = img.copy()
    overlay = img.copy()
    
    # Generate colors for classes (matching Rust implementation)
    np.random.seed(42)
    max_class = max([d['class_id'] for d in detections]) if detections else 0
    colors = {}
    for i in range(max_class + 1):
        hue = (i * 137.5) % 360.0
        colors[i] = hsv_to_rgb(hue, 0.8, 0.95)
    
    for det in detections:
        color = colors[det['class_id']]
        mask = det['mask']
        bbox = det['bbox']
        
        # Apply mask overlay
        overlay[mask == 1] = (overlay[mask == 1] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
        
        # Draw bounding box
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
    
    # Blend overlay with original
    result = cv2.addWeighted(result, 0.6, overlay, 0.4, 0)
    
    return result


def hsv_to_rgb(h, s, v):
    """Convert HSV to RGB (matching Rust implementation)"""
    c = v * s
    x = c * (1.0 - abs((h / 60.0) % 2.0 - 1.0))
    m = v - c
    
    if h < 60:
        r, g, b = c, x, 0
    elif h < 120:
        r, g, b = x, c, 0
    elif h < 180:
        r, g, b = 0, c, x
    elif h < 240:
        r, g, b = 0, x, c
    elif h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    
    return (int((b + m) * 255), int((g + m) * 255), int((r + m) * 255))


def preprocess_image(img):
    """Preprocess image for ONNX inference"""
    img_resized = cv2.resize(img, (512, 512))
    img_input = img_resized.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return img_input


# Thread-local storage for sessions
import threading
thread_local = threading.local()


def get_session(model_path, providers):
    """Get or create a session for the current thread"""
    if not hasattr(thread_local, 'session'):
        thread_local.session = ort.InferenceSession(model_path, providers=providers)
    return thread_local.session


async def process_single_image(model_path, providers, img_data, output_dir, conf_threshold, iou_threshold, processed_count, total_images, lock):
    """Process a single image asynchronously"""
    img, preprocessed, source_name = img_data
    
    # Run inference in thread pool (blocking operation)
    loop = asyncio.get_event_loop()
    
    def run_inference():
        session = get_session(model_path, providers)
        inputs = {session.get_inputs()[0].name: preprocessed}
        outputs = session.run(None, inputs)
        return outputs
    
    outputs = await loop.run_in_executor(None, run_inference)
    
    # Process and visualize (CPU-bound, also run in executor)
    def process_and_save():
        # Process YOLOv8 output
        detections = process_yolov8_output(
            outputs[0], 
            outputs[1], 
            img.shape[:2],
            conf_threshold,
            iou_threshold
        )
        
        # Visualize
        result_img = visualize_yolo_segmentation(detections, img)
        
        # Save
        output_path = os.path.join(output_dir, f"{source_name}.jpg")
        cv2.imwrite(output_path, result_img)
    
    await loop.run_in_executor(None, process_and_save)
    
    # Update progress
    async with lock:
        processed_count[0] += 1
        if processed_count[0] % 100 == 0 or processed_count[0] == total_images:
            print(f"Processed {processed_count[0]}/{total_images} images")


async def main():
    # Load config
    config_path = "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_path = config['model_path']
    device = config['device'].lower()
    output_dir = config['output_dir']
    conf_threshold = config.get('conf_threshold', 0.25)
    iou_threshold = config.get('iou_threshold', 0.45)
    input_folder = config['input_folder']
    num_copies = config.get('num_copies', 300)
    num_threads = config.get('num_threads', None)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up ONNX Runtime providers
    if device == 'gpu':
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        print("Using GPU (CUDA) execution provider")
    else:
        providers = ['CPUExecutionProvider']
        print("Using CPU execution provider")
    
    print("Model will be loaded per-thread on first use")
    
    if num_threads:
        print(f"Using {num_threads} threads")
    else:
        num_threads = os.cpu_count()
        print(f"Using default thread pool size: {num_threads}")
    
    # Set the executor thread pool size
    from concurrent.futures import ThreadPoolExecutor
    loop = asyncio.get_event_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=num_threads))
    
    # Step 1: Load all images from input folder
    print("\n=== DRAG RACE SETUP ===")
    print(f"Loading images from: {input_folder}")
    
    load_start = time.time()
    original_images = []
    
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        for img_path in glob.glob(os.path.join(input_folder, ext)):
            img = cv2.imread(img_path)
            if img is not None:
                source_name = Path(img_path).stem
                original_images.append((img, source_name))
    
    num_original = len(original_images)
    print(f"Loaded {num_original} original images")
    
    if num_original == 0:
        print("No images found in input folder")
        return
    
    # Step 2: Create copies and preprocess
    print(f"Creating {num_copies} copies of each image...")
    all_image_data = []
    
    for img, source_name in original_images:
        for copy_id in range(num_copies):
            preprocessed = preprocess_image(img)
            all_image_data.append((
                img,
                preprocessed,
                f"{source_name}_{copy_id:04d}"
            ))
    
    total_images = len(all_image_data)
    load_duration = time.time() - load_start
    
    print(f"Total images prepared: {total_images}")
    print(f"Preparation time: {load_duration:.2f}s")
    
    # Step 3: START THE DRAG RACE!
    print("\n=== STARTING DRAG RACE ===")
    input("Press Enter to start inference...")
    
    race_start = time.time()
    
    print(f"Processing {total_images} images in parallel...")
    
    # Track progress
    processed_count = [0]  # Mutable container for async updates
    lock = asyncio.Lock()
    
    # Process ALL images in parallel
    tasks = []
    for img_data in all_image_data:
        task = process_single_image(
            model_path,
            providers,
            img_data,
            output_dir,
            conf_threshold,
            iou_threshold,
            processed_count,
            total_images,
            lock
        )
        tasks.append(task)
    
    # Wait for all tasks to complete
    await asyncio.gather(*tasks)
    
    race_duration = time.time() - race_start
    
    # Step 4: Save drag race results
    print("\n=== DRAG RACE COMPLETE ===")
    
    total_duration_sec = race_duration
    avg_per_image_ms = (race_duration * 1000) / total_images
    images_per_sec = total_images / total_duration_sec
    
    print(f"Total time: {total_duration_sec:.2f}s")
    print(f"Average per image: {avg_per_image_ms:.2f}ms")
    print(f"Throughput: {images_per_sec:.2f} images/sec")
    
    stats_file = os.path.join(output_dir, "drag_race_data.txt")
    with open(stats_file, 'w') as f:
        f.write("=== PYTHON YOLOV8-SEG DRAG RACE RESULTS (ASYNC) ===\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n")
        f.write("CONFIGURATION:\n")
        f.write(f"  Model: {model_path}\n")
        f.write(f"  Device: {device}\n")
        f.write(f"  Threads: {num_threads}\n")
        f.write(f"  Confidence threshold: {conf_threshold}\n")
        f.write(f"  IOU threshold: {iou_threshold}\n")
        f.write(f"  Async runtime: asyncio\n")
        f.write("\n")
        f.write("DATASET:\n")
        f.write(f"  Original images: {num_original}\n")
        f.write(f"  Copies per image: {num_copies}\n")
        f.write(f"  Total images: {total_images}\n")
        f.write(f"  Preparation time: {load_duration:.2f}s\n")
        f.write("\n")
        f.write("RESULTS:\n")
        f.write(f"  Total inference time: {total_duration_sec:.2f}s\n")
        f.write(f"  Average per image: {avg_per_image_ms:.2f}ms\n")
        f.write(f"  Throughput: {images_per_sec:.2f} images/sec\n")
    
    print(f"\nResults saved to: {stats_file}")


if __name__ == "__main__":
    asyncio.run(main())