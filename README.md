# cs583_dental

Python scripts for training, predicting, and exporting YOLOv8 dental X-ray segmentation models.

## Installation

```bash

pip install -r requirements.txt

```

Note: requirements.txt will only install the version of pytorch that is compatable with CPU usage.
Install a CUDA enabled version first then install via requirements.txt.

## Usage

### 1. Train a Model

Train a YOLOv8 segmentation model on dental X-ray data:

```bash
python train.py
```

Optional arguments:

- `--epochs`: Number of training epochs (default: 25)

- `--batch-size`: Batch size (default: 2)

- `--image-size`: Image size (default: 512)

- `--device`: Device to use (default: cpu)

Example:

```bash
python train.py --epochs 50 --batch-size 4 --device 0
```

### 2. Run Predictions

Run inference on dental X-ray images:

```bash
python predict.py <model_path> <source>
```

Optional arguments:

- `--conf`: Confidence threshold (default: 0.25)

- `--image-size`: Input image size (default: 512)

Example:

```bash
python predict.py runs/segment/train/weights/best.pt data/opg/test/images --conf 0.50
```

### 3. Export to ONNX

Export trained model to ONNX format for deployment:

```bash
python export_onnx.py <model_path>
```

Example:

```bash
python export_onnx.py runs/segment/train/weights/best.pt
```

## Output Locations

- **Training**: `runs/segment/train/`

- **Predictions**: `runs/segment/predict/`

- **ONNX export**: Same directory as input model with `.onnx` extension