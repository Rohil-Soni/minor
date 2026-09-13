use std::collections::HashMap;
use std::path::{self, PathBuf};
use std::io::BufWriter;
use walkdir::WalkDir;
use std::fs::File;
use std::io::{BufReader, Write}; 
use exif::{In, Reader, Tag};
use rayon::prelude::*;
use image::{GenericImageView, Pixel};

mod arw;

#[derive(Default, Debug, serde::Serialize)]
struct ImgPair {
    jpeg: Option<PathBuf>,
    arw: Option<PathBuf>,
}

#[derive(Debug, serde::Serialize)]
struct ImageRecord {
    path: PathBuf,
    arw_path: Option<PathBuf>,
    focal_length: Option<String>,
    iso: Option<String>,
    shutter_speed: Option<String>,
    date_time: Option<String>,
    sharpness_score: f64,
}

fn find_and_send(directory: &str) -> Vec<ImgPair> {
    let mut imagegroup: HashMap<(PathBuf, String), ImgPair> = HashMap::new();

    for entry in WalkDir::new(directory).into_iter().filter_map(|e| e.ok()) {
        let path = entry.into_path();

        if path.is_file() {
            if let Some(extension) = path.extension() {
                let ext_str = extension.to_string_lossy().to_lowercase();

                if matches!(ext_str.as_str(), "jpg" | "jpeg" | "arw") {
                    
                    if let (Some(parent), Some(file_stem)) = (path.parent(), path.file_stem()) {

                        let key = (parent.to_path_buf(), file_stem.to_string_lossy().to_string());
                        let pair = imagegroup.entry(key).or_default();

                        match ext_str.as_str() {
                            "jpg" | "jpeg" => pair.jpeg = Some(path.clone()),
                            "arw" => pair.arw = Some(path.clone()),
                            _ => {}
                        }
                    } 
                }
            }
        }
    }
    imagegroup.into_values().collect()
}

fn calculate_sharpness(path: &PathBuf) -> Option<f64>{

    let ext = path.extension().unwrap_or_default().to_string_lossy().to_lowercase();

    let img = if ext == "arw" {

        arw::extract_luma_from_arw(path)?

    } else {
        image::open(path).ok()?.into_luma8()
    };

    let (width, height) = img.dimensions();

    let mut laplacian_values = Vec::with_capacity((width*height) as usize);

    for y in 1..height-1 {
        for x in 1..width-1 {
            let top = img.get_pixel(x, y-1)[0] as f64;
            let bottom = img.get_pixel(x, y+1)[0] as f64;
            let left = img.get_pixel(x-1, y)[0] as f64;
            let right = img.get_pixel(x+1, y)[0] as f64;
            let center = img.get_pixel(x, y)[0] as f64;
            let laplacian = (top + bottom + left + right - 4.0 * center).abs();
            laplacian_values.push(laplacian);
        }
    }

    let count = laplacian_values.len() as f64;
    if count == 0.0 { return Some(0.0); }

    let mean = laplacian_values.iter().sum::<f64>() / count;

    let variance = laplacian_values.iter()
        .map(|value| {
            let diff = value - mean;
            diff * diff
        })
        .sum::<f64>() / count;
    Some(variance)
}

fn extract_metadata(pair: &ImgPair) -> Option<ImageRecord> {

    let target_path = pair.jpeg.as_ref().or(pair.arw.as_ref())?;

    let sharpness = calculate_sharpness(target_path)?;

    if sharpness < 15.0 {
        println!("Skipping {} due to low sharpness: {}", target_path.display(), sharpness);
        return None;
    }
    
    let file = File::open(target_path).ok()?;
    let mut bufreader = BufReader::new(&file);
    let exifreader = Reader::new();
    
    let exif = exifreader.read_from_container(&mut bufreader).ok()?;

    let focal_length = exif.get_field(Tag::FocalLength, In::PRIMARY)
        .map(|f| f.display_value().with_unit(&exif).to_string());
        
    let iso = exif.get_field(Tag::PhotographicSensitivity, In::PRIMARY)
        .map(|f| f.display_value().with_unit(&exif).to_string());
        
    let shutter_speed = exif.get_field(Tag::ExposureTime, In::PRIMARY)
        .map(|f| f.display_value().with_unit(&exif).to_string());

    let date_time = exif.get_field(Tag::DateTime, In::PRIMARY)
        .map(|f| f.display_value().with_unit(&exif).to_string());

        let rawholder = if pair.jpeg.is_some() {
            pair.arw.clone()
        } else {
            None
        };

    Some(ImageRecord {
        path: target_path.clone(),
        arw_path: rawholder,
        focal_length,
        iso,
        shutter_speed,
        date_time,
        sharpness_score: sharpness,
    })
}

fn main() {
    let target = "D:\\Photos\\DSC05262.ARW";
    println!("Scanning directory: {}", target);

    let image_paths = find_and_send(target);

    println!("Found {} image files. Extracting metadata...", image_paths.len());

    let discovered_filename = "discovered_images.txt";
    let discovered_file = File::create(discovered_filename).expect("Failed to create discovered file");
    let mut discovered_writer = BufWriter::new(discovered_file);

    for pair in &image_paths {
        if let Some(jpg) = &pair.jpeg {
            writeln!(discovered_writer, "JPEG: {}", jpg.display()).expect("Failed to write to file");
        }
        if let Some(arw) = &pair.arw {
            writeln!(discovered_writer, "RAW:  {}", arw.display()).expect("Failed to write to file");
        }
        writeln!(discovered_writer, "---").expect("Failed to write to file");
    }
    println!("Discovered images saved to {}", discovered_filename);
    println!("Extracting metadata across all cores...");

    let exif_data_list: Vec<ImageRecord> = image_paths
        .par_iter()
        .filter_map(|pair| extract_metadata(pair))
        .collect();

    let json_filename = "filtered_images.json";
    let json_file = File::create(json_filename).expect("Failed to create JSON file");
    serde_json::to_writer_pretty(json_file, &exif_data_list).expect("Failed to write JSON");
    println!("Structured data successfully serialized to {}", json_filename);
}