use anyhow::{Context, Result};
use clap::Parser;
use ndarray::Array4;
use ort::{
    execution_providers::{CPUExecutionProvider, CUDAExecutionProvider},
    inputs,
    init,
    session::Session,
    value::Tensor,
};
use serde::{Deserialize, Serialize};
use serde_json;
use std::fs::{self, File};
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;
use chrono::Utc;

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
}

fn default_output_dir() -> String {
    "./results".to_string()
}

#[derive(Debug, Serialize)]
struct InferenceResult {
    image_path: String,
    model_path: String,
    device: String,
    timestamp_ms: i64,
    duration_ms: u128,
    outputs: Vec<OutputTensorSummary>,
}

#[derive(Debug, Serialize)]
struct OutputTensorSummary {
    name: Option<String>,
    shape: Vec<usize>,
    values: Vec<f32>,
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

    let out_dir = Path::new(&config.output_dir);
    fs::create_dir_all(out_dir)
        .with_context(|| format!("Failed to create output dir: {}", out_dir.display()))?;

    if device == "gpu" {
        init()
            .with_execution_providers([CUDAExecutionProvider::default().build()])
            .commit()
            .context("Failed to create ORT environment with CUDA execution provider. Ensure ONNX Runtime was built with CUDA and your system has matching CUDA libs.")?;
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

    let mut session = Session::builder()?
        .commit_from_file(model_path)
        .with_context(|| format!("Failed to create session from model '{}'", model_path.display()))?;

    println!(
        "Loaded model '{}'. Enter image paths to run inference (Ctrl-C to exit).",
        model_path.display()
    );

    let stdin = io::stdin();
    let mut lock = stdin.lock();

    loop {
        print!("Image path > ");
        io::stdout().flush().ok();

        let mut line = String::new();
        if lock.read_line(&mut line)? == 0 {
            println!("\nEOF — exiting.");
            break;
        }
        let img_path = line.trim();
        if img_path.is_empty() {
            continue;
        }

        let img_path_obj = Path::new(img_path);
        if !img_path_obj.exists() {
            eprintln!("File does not exist: {}", img_path);
            continue;
        }

        let img = match image::open(img_path_obj) {
            Ok(im) => im,
            Err(e) => {
                eprintln!("Failed to open image '{}': {}", img_path, e);
                continue;
            }
        };

        let img_resized = img.resize_exact(512, 512, image::imageops::FilterType::Lanczos3);
        
        let rgb = img_resized.to_rgb8();
        let (w, h) = rgb.dimensions(); 
        let mut arr = Array4::<f32>::zeros((1usize, 3usize, h as usize, w as usize));

        for y in 0..(h as usize) {
            for x in 0..(w as usize) {
                let p = rgb.get_pixel(x as u32, y as u32).0;
                arr[[0, 0, y, x]] = p[0] as f32 / 255.0;
                arr[[0, 1, y, x]] = p[1] as f32 / 255.0;
                arr[[0, 2, y, x]] = p[2] as f32 / 255.0;
            }
        }

        let shape = arr.shape().to_vec();
        let data: Vec<f32> = arr.into_raw_vec();
        
        let input_tensor = Tensor::from_array((shape.as_slice(), data))?;

        let inputs = inputs![input_tensor];

        let output_names: Vec<Option<String>> = session.outputs
            .iter()
            .map(|o| Some(o.name.clone()))
            .collect();

        let t0 = Instant::now();
        let outputs = match session.run(inputs) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("Inference error for '{}': {:?}", img_path, e);
                continue;
            }
        };
        let duration = t0.elapsed();

        let mut out_summaries: Vec<OutputTensorSummary> = Vec::with_capacity(outputs.len());

        for (i, out_val) in outputs.into_iter().enumerate() {
            if let Ok(arr_out) = out_val.1.try_extract_array::<f32>() {
                let shape = arr_out.shape().to_vec();
                let mut values: Vec<f32> = arr_out.iter().copied().collect();
                if values.len() > 8192 {
                    values.truncate(8192);
                }
                let name = output_names.get(i).and_then(|n| n.clone());
                out_summaries.push(OutputTensorSummary {
                    name,
                    shape,
                    values,
                });
            } else {
                let name = output_names.get(i).and_then(|n| n.clone());
                out_summaries.push(OutputTensorSummary {
                    name,
                    shape: vec![],
                    values: vec![],
                });
            }
        }

        let result = InferenceResult {
            image_path: img_path.to_string(),
            model_path: config.model_path.clone(),
            device: config.device.clone(),
            timestamp_ms: Utc::now().timestamp_millis(),
            duration_ms: duration.as_millis(),
            outputs: out_summaries,
        };

        let stem = img_path_obj
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("image");
        let ts = Utc::now().format("%Y%m%dT%H%M%S%.3f");
        let file_name = format!("{}_{}.json", stem, ts);
        let out_file = out_dir.join(file_name);

        let mut f = File::create(&out_file)
            .with_context(|| format!("Failed to create output file: {}", out_file.display()))?;
        let pretty = serde_json::to_string_pretty(&result)?;
        f.write_all(pretty.as_bytes())?;

        println!(
            "Inference finished in {} ms; results written to {}",
            duration.as_millis(),
            out_file.display()
        );
    }

    Ok(())
}