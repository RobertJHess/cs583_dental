use anyhow::{Context, Result};
use clap::Parser;
use ort::{
    execution_providers::{CPUExecutionProvider, CUDAExecutionProvider},
    inputs,
    init,
    session::Session,
    value::Tensor,
};
use serde::Deserialize;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::cell::UnsafeCell;
use std::time::Instant;
use chrono::Utc;
use image::{DynamicImage, Rgb, RgbImage};
use rayon::prelude::*;

#[derive(Parser, Debug)]
#[command(name = "onnx_image_infer")]
struct Cli {
    #[arg(short, long)]
    config: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct Config {
    model_path: String,
    device: String,
    #[serde(default = "default_output_dir")]
    output_dir: String,
    #[serde(default = "default_conf_threshold")]
    conf_threshold: f32,
    #[serde(default = "default_iou_threshold")]
    iou_threshold: f32,
    input_folder: String,
    #[serde(default = "default_num_copies")]
    num_copies: usize,
    num_threads: Option<usize>,
}

fn default_output_dir() -> String {
    "./results".to_string()
}

fn default_conf_threshold() -> f32 {
    0.25
}

fn default_iou_threshold() -> f32 {
    0.45
}

fn default_num_copies() -> usize {
    300
}

#[derive(Debug, Clone)]
struct Detection {
    bbox: [f32; 4],
    score: f32,
    class_id: usize,
    mask: Vec<Vec<u8>>,
}

#[derive(Clone)]
struct ImageData {
    original_img: DynamicImage,
    preprocessed: Vec<f32>,
    orig_width: u32,
    orig_height: u32,
    source_name: String,
}

struct BatchStats {
    batch_id: usize,
    num_images: usize,
    duration_ms: u128,
    avg_per_image_ms: f64,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn nms(mut detections: Vec<Detection>, iou_threshold: f32) -> Vec<Detection> {
    detections.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    
    let mut keep = Vec::new();
    let mut suppressed = vec![false; detections.len()];
    
    for i in 0..detections.len() {
        if suppressed[i] {
            continue;
        }
        keep.push(detections[i].clone());
        
        let box_a = &detections[i].bbox;
        
        for j in (i + 1)..detections.len() {
            if suppressed[j] {
                continue;
            }
            
            let box_b = &detections[j].bbox;
            
            let x1 = box_a[0].max(box_b[0]);
            let y1 = box_a[1].max(box_b[1]);
            let x2 = box_a[2].min(box_b[2]);
            let y2 = box_a[3].min(box_b[3]);
            
            let intersection = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
            let area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]);
            let area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]);
            let union = area_a + area_b - intersection;
            
            let iou = intersection / union;
            
            if iou > iou_threshold {
                suppressed[j] = true;
            }
        }
    }
    
    keep
}

fn process_yolov8_output(
    predictions: &[f32],
    predictions_shape: &[usize],
    proto: &[f32],
    _proto_shape: &[usize],
    orig_width: u32,
    orig_height: u32,
    conf_threshold: f32,
    iou_threshold: f32,
) -> Result<Vec<Detection>> {
    let num_attrs = predictions_shape[1];
    let num_detections = predictions_shape[2];
    
    let num_mask_coeffs = 32;
    let num_classes = num_attrs - 4 - num_mask_coeffs;
    
    let mut detections = Vec::new();
    
    for i in 0..num_detections {
        let x = predictions[0 * num_detections + i];
        let y = predictions[1 * num_detections + i];
        let w = predictions[2 * num_detections + i];
        let h = predictions[3 * num_detections + i];
        
        let mut max_score = 0.0f32;
        let mut max_class = 0usize;
        
        for c in 0..num_classes {
            let score = predictions[(4 + c) * num_detections + i];
            if score > max_score {
                max_score = score;
                max_class = c;
            }
        }
        
        if max_score < conf_threshold {
            continue;
        }
        
        let x1 = x - w / 2.0;
        let y1 = y - h / 2.0;
        let x2 = x + w / 2.0;
        let y2 = y + h / 2.0;
        
        let mask_coeffs: Vec<f32> = (0..num_mask_coeffs)
            .map(|j| predictions[(4 + num_classes + j) * num_detections + i])
            .collect();
        
        let mut mask_128 = vec![vec![0.0f32; 128]; 128];
        
        for y in 0..128 {
            for x in 0..128 {
                let mut sum = 0.0;
                for c in 0..32 {
                    let idx = c * 128 * 128 + y * 128 + x;
                    sum += mask_coeffs[c] * proto[idx];
                }
                mask_128[y][x] = sigmoid(sum);
            }
        }
        
        let mut mask_full = vec![vec![0u8; orig_width as usize]; orig_height as usize];
        
        for y in 0..orig_height as usize {
            for x in 0..orig_width as usize {
                let src_y = ((y as f32 / orig_height as f32) * 128.0) as usize;
                let src_x = ((x as f32 / orig_width as f32) * 128.0) as usize;
                let src_y = src_y.min(127);
                let src_x = src_x.min(127);
                
                mask_full[y][x] = if mask_128[src_y][src_x] > 0.5 { 1 } else { 0 };
            }
        }
        
        let scale_x = orig_width as f32 / 512.0;
        let scale_y = orig_height as f32 / 512.0;
        
        detections.push(Detection {
            bbox: [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y],
            score: max_score,
            class_id: max_class,
            mask: mask_full,
        });
    }
    
    Ok(nms(detections, iou_threshold))
}

fn generate_colors(num_classes: usize) -> Vec<Rgb<u8>> {
    let mut colors = Vec::new();
    for i in 0..num_classes {
        let hue = (i as f32 * 137.5) % 360.0;
        let rgb = hsv_to_rgb(hue, 0.8, 0.95);
        colors.push(rgb);
    }
    colors
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> Rgb<u8> {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;
    
    let (r, g, b) = if h < 60.0 {
        (c, x, 0.0)
    } else if h < 120.0 {
        (x, c, 0.0)
    } else if h < 180.0 {
        (0.0, c, x)
    } else if h < 240.0 {
        (0.0, x, c)
    } else if h < 300.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };
    
    Rgb([
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    ])
}

fn visualize_detections(
    img: &DynamicImage,
    detections: &[Detection],
    colors: &[Rgb<u8>],
) -> RgbImage {
    let mut result = img.to_rgb8();
    let mut overlay = result.clone();
    
    for det in detections {
        let color = colors[det.class_id % colors.len()];
        
        for y in 0..det.mask.len() {
            for x in 0..det.mask[0].len() {
                if det.mask[y][x] == 1 {
                    let pixel = overlay.get_pixel_mut(x as u32, y as u32);
                    pixel[0] = ((pixel[0] as f32 * 0.5) + (color[0] as f32 * 0.5)) as u8;
                    pixel[1] = ((pixel[1] as f32 * 0.5) + (color[1] as f32 * 0.5)) as u8;
                    pixel[2] = ((pixel[2] as f32 * 0.5) + (color[2] as f32 * 0.5)) as u8;
                }
            }
        }
        
        let x1 = det.bbox[0] as i32;
        let y1 = det.bbox[1] as i32;
        let x2 = det.bbox[2] as i32;
        let y2 = det.bbox[3] as i32;
        
        draw_box(&mut result, x1, y1, x2, y2, color, 2);
    }
    
    for y in 0..result.height() {
        for x in 0..result.width() {
            let orig = result.get_pixel(x, y);
            let over = overlay.get_pixel(x, y);
            let blended = Rgb([
                ((orig[0] as f32 * 0.6) + (over[0] as f32 * 0.4)) as u8,
                ((orig[1] as f32 * 0.6) + (over[1] as f32 * 0.4)) as u8,
                ((orig[2] as f32 * 0.6) + (over[2] as f32 * 0.4)) as u8,
            ]);
            result.put_pixel(x, y, blended);
        }
    }
    
    result
}

fn draw_box(img: &mut RgbImage, x1: i32, y1: i32, x2: i32, y2: i32, color: Rgb<u8>, thickness: i32) {
    let width = img.width() as i32;
    let height = img.height() as i32;
    
    for t in 0..thickness {
        for x in x1..=x2 {
            if x >= 0 && x < width {
                if y1 + t >= 0 && y1 + t < height {
                    img.put_pixel(x as u32, (y1 + t) as u32, color);
                }
                if y2 - t >= 0 && y2 - t < height {
                    img.put_pixel(x as u32, (y2 - t) as u32, color);
                }
            }
        }
        for y in y1..=y2 {
            if y >= 0 && y < height {
                if x1 + t >= 0 && x1 + t < width {
                    img.put_pixel((x1 + t) as u32, y as u32, color);
                }
                if x2 - t >= 0 && x2 - t < width {
                    img.put_pixel((x2 - t) as u32, y as u32, color);
                }
            }
        }
    }
}

fn preprocess_image(img: &DynamicImage) -> Vec<f32> {
    let img_resized = img.resize_exact(512, 512, image::imageops::FilterType::Lanczos3);
    let rgb = img_resized.to_rgb8();
    let (w, h) = rgb.dimensions();
    
    let mut data = vec![0.0f32; 3 * (h as usize) * (w as usize)];
    
    for y in 0..h as usize {
        for x in 0..w as usize {
            let p = rgb.get_pixel(x as u32, y as u32).0;
            data[0 * h as usize * w as usize + y * w as usize + x] = p[0] as f32 / 255.0;
            data[1 * h as usize * w as usize + y * w as usize + x] = p[1] as f32 / 255.0;
            data[2 * h as usize * w as usize + y * w as usize + x] = p[2] as f32 / 255.0;
        }
    }
    
    data
}

fn main() -> Result<()> {
    env_logger::init();

    let cli = Cli::parse();
    let config_path = cli.config.unwrap_or_else(|| PathBuf::from("config.yaml"));

    let config_text = fs::read_to_string(&config_path)
        .with_context(|| format!("Failed to read config file: {}", config_path.display()))?;
    let config: Config = serde_yaml::from_str(&config_text)
        .with_context(|| format!("Failed to parse YAML config: {}", config_path.display()))?;

    let device = config.device.to_lowercase();
    if device != "cpu" && device != "gpu" {
        anyhow::bail!("config.device must be 'cpu' or 'gpu'");
    }

    // Set thread pool size if specified
    if let Some(num_threads) = config.num_threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global()
            .context("Failed to set thread pool size")?;
        println!("Using {} threads", num_threads);
    } else {
        println!("Using default thread pool size: {}", rayon::current_num_threads());
    }

    let out_dir = Path::new(&config.output_dir);
    fs::create_dir_all(out_dir)
        .with_context(|| format!("Failed to create output dir: {}", out_dir.display()))?;

    if device == "gpu" {
        init()
            .with_execution_providers([CUDAExecutionProvider::default().build()])
            .commit()
            .context("Failed to create ORT environment with CUDA execution provider.")?;
        println!("Initialized ORT environment with CUDA execution provider (GPU).");
    } else {
        init()
            .with_execution_providers([CPUExecutionProvider::default().build()])
            .commit()
            .context("Failed to create ORT environment with CPU execution provider.")?;
        println!("Initialized ORT environment with CPU execution provider.");
    }

    let model_path = Path::new(&config.model_path);
    if !model_path.exists() {
        anyhow::bail!("Model file not found: {}", model_path.display());
    }

    println!("Loading model...");
    let model_path_for_threads = model_path.to_path_buf();

    println!("Model loaded successfully.");

    // Step 1: Load all images from input folder
    println!("\n=== DRAG RACE SETUP ===");
    println!("Loading images from: {}", config.input_folder);
    
    let load_start = Instant::now();
    let mut original_images = Vec::new();
    
    for entry in fs::read_dir(&config.input_folder)? {
        let entry = entry?;
        let path = entry.path();
        
        if path.is_file() {
            if let Some(ext) = path.extension() {
                let ext_str = ext.to_string_lossy().to_lowercase();
                if ext_str == "jpg" || ext_str == "jpeg" || ext_str == "png" {
                    match image::open(&path) {
                        Ok(img) => {
                            let file_name = path.file_stem()
                                .and_then(|s| s.to_str())
                                .unwrap_or("image")
                                .to_string();
                            original_images.push((img, file_name));
                        }
                        Err(e) => {
                            eprintln!("Failed to load {}: {}", path.display(), e);
                        }
                    }
                }
            }
        }
    }
    
    let num_original = original_images.len();
    println!("Loaded {} original images", num_original);
    
    if num_original == 0 {
        anyhow::bail!("No images found in input folder");
    }
    
    // Step 2: Create copies and preprocess
    println!("Creating {} copies of each image...", config.num_copies);
    let mut all_image_data = Vec::new();
    
    for (orig_img, source_name) in &original_images {
        for copy_id in 0..config.num_copies {
            let preprocessed = preprocess_image(orig_img);
            all_image_data.push(ImageData {
                original_img: orig_img.clone(),
                preprocessed,
                orig_width: orig_img.width(),
                orig_height: orig_img.height(),
                source_name: format!("{}_{:04}", source_name, copy_id),
            });
        }
    }
    
    let total_images = all_image_data.len();
    let load_duration = load_start.elapsed();
    
    println!("Total images prepared: {}", total_images);
    println!("Preparation time: {:.2}s", load_duration.as_secs_f64());
    
    // Generate colors once
    let colors = generate_colors(100);
    
    // Step 3: START THE DRAG RACE!
    println!("\n=== STARTING DRAG RACE ===");
    println!("Press Enter to start inference...");
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    
    let race_start = Instant::now();
    
    println!("Processing {} images in parallel...", total_images);
    
    let model_path_clone = model_path_for_threads.clone();
    let conf_thresh = config.conf_threshold;
    let iou_thresh = config.iou_threshold;
    let out_dir_clone = out_dir.to_path_buf();
    let colors_clone = colors.clone();
    
    // Track progress
    let processed_count = Arc::new(Mutex::new(0usize));
    
    // Process ALL images in parallel
    all_image_data.par_iter().enumerate().try_for_each(|(idx, img_data)| -> Result<()> {
        let batch_id = idx / 10;  // For tracking purposes
        let batch_start = Instant::now();
        
        // Each thread creates its own session - ONNX Runtime sessions are thread-safe
        // Creating per-thread avoids lock contention
        thread_local! {
            static SESSION_CACHE: UnsafeCell<Option<Session>> = UnsafeCell::new(None);
        }
        
        let outputs = SESSION_CACHE.with(|cache| {
            let cache_ptr = cache.get();
            unsafe {
                // Initialize session on first use in this thread
                if (*cache_ptr).is_none() {
                    *cache_ptr = Some(
                        Session::builder()
                            .unwrap()
                            .commit_from_file(&model_path_clone)
                            .expect("Failed to create session")
                    );
                }
                
                let session = (*cache_ptr).as_mut().unwrap();
                
                // Run inference
                let shape = vec![1, 3, 512, 512];
                let input_tensor = Tensor::from_array((shape.as_slice(), img_data.preprocessed.clone()))
                    .expect("Failed to create tensor");
                
                let inputs = inputs![input_tensor];
                session.run(inputs).expect("Inference failed")
            }
        });
        
        // Extract tensor data
        let (predictions_shape, predictions) = outputs[0].try_extract_tensor::<f32>()
            .expect("Failed to extract predictions");
        let (proto_shape, proto) = outputs[1].try_extract_tensor::<f32>()
            .expect("Failed to extract proto");
        
        let predictions_shape_usize: Vec<usize> = predictions_shape.iter().map(|&x| x as usize).collect();
        let proto_shape_usize: Vec<usize> = proto_shape.iter().map(|&x| x as usize).collect();
        let predictions_vec = predictions.to_vec();
        let proto_vec = proto.to_vec();
        
        // Process detections
        let detections = process_yolov8_output(
            &predictions_vec,
            &predictions_shape_usize,
            &proto_vec,
            &proto_shape_usize,
            img_data.orig_width,
            img_data.orig_height,
            conf_thresh,
            iou_thresh,
        )?;
        
        // Visualize
        let result_img = visualize_detections(&img_data.original_img, &detections, &colors_clone);
        
        // Save
        let img_file = out_dir_clone.join(format!("{}.jpg", img_data.source_name));
        result_img.save(&img_file).expect("Failed to save image");
        
        let duration = batch_start.elapsed();
        
        // Update progress counter
        let mut count = processed_count.lock().unwrap();
        *count += 1;
        if *count % 100 == 0 || *count == total_images {
            println!("Processed {}/{} images", *count, total_images);
        }
        
        Ok(())
    }).expect("Image processing failed");
    
    let race_duration = race_start.elapsed();
    
    // Step 4: Save drag race results
    println!("\n=== DRAG RACE COMPLETE ===");
    
    let total_duration_sec = race_duration.as_secs_f64();
    let avg_per_image_ms = race_duration.as_millis() as f64 / total_images as f64;
    let images_per_sec = total_images as f64 / total_duration_sec;
    
    println!("Total time: {:.2}s", total_duration_sec);
    println!("Average per image: {:.2}ms", avg_per_image_ms);
    println!("Throughput: {:.2} images/sec", images_per_sec);
    
    let stats_file = out_dir.join("drag_race_data.txt");
    let mut f = File::create(&stats_file)?;
    
    writeln!(f, "=== RUST YOLOV8-SEG DRAG RACE RESULTS ===")?;
    writeln!(f, "Timestamp: {}", Utc::now().format("%Y-%m-%d %H:%M:%S"))?;
    writeln!(f, "")?;
    writeln!(f, "CONFIGURATION:")?;
    writeln!(f, "  Model: {}", config.model_path)?;
    writeln!(f, "  Device: {}", config.device)?;
    writeln!(f, "  Threads: {}", rayon::current_num_threads())?;
    writeln!(f, "  Confidence threshold: {}", config.conf_threshold)?;
    writeln!(f, "  IOU threshold: {}", config.iou_threshold)?;
    writeln!(f, "")?;
    writeln!(f, "DATASET:")?;
    writeln!(f, "  Original images: {}", num_original)?;
    writeln!(f, "  Copies per image: {}", config.num_copies)?;
    writeln!(f, "  Total images: {}", total_images)?;
    writeln!(f, "  Preparation time: {:.2}s", load_duration.as_secs_f64())?;
    writeln!(f, "")?;
    writeln!(f, "RESULTS:")?;
    writeln!(f, "  Total inference time: {:.2}s", total_duration_sec)?;
    writeln!(f, "  Average per image: {:.2}ms", avg_per_image_ms)?;
    writeln!(f, "  Throughput: {:.2} images/sec", images_per_sec)?;
    writeln!(f, "")?;
    
    println!("\nResults saved to: {}", stats_file.display());
    
    Ok(())
}