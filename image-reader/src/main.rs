use exif::{Exif, In, Reader, Tag, Value};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufWriter;
use std::io::{BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::SystemTime;
use walkdir::WalkDir;

mod arw;

const SCHEMA_VERSION: u32 = 1;
const DEFAULT_MIN_SHARPNESS: f64 = 15.0;
const RUN_DIR: &str = ".photoholic";
// output folders written by the python side; never re-scan them
const SKIP_DIRS: [&str; 3] = [RUN_DIR, "photoholic_picks", "photoholic_rejects"];

const USAGE: &str = "usage: image-reader <INPUT_DIR> [--out <JSON_PATH>] [--min-sharpness <F64>]

  --out             output JSON path (default: <INPUT_DIR>/.photoholic/filtered_images.json)
  --min-sharpness   reject images with Laplacian variance below this (default: 15.0)";

struct Args {
    input_dir: PathBuf,
    out: PathBuf,
    min_sharpness: f64,
}

#[derive(Default, Debug, serde::Serialize)]
struct ImgPair {
    jpeg: Option<PathBuf>,
    arw: Option<PathBuf>,
}

#[derive(Debug, serde::Serialize)]
struct ImageRecord {
    id: String,
    path: PathBuf,
    arw_path: Option<PathBuf>,
    focal_length: Option<String>,
    iso: Option<String>,
    shutter_speed: Option<String>,
    date_time: Option<String>,
    date_time_original: Option<String>,
    subsec_time_original: Option<String>,
    sharpness_score: f64,
}

#[derive(Debug, serde::Serialize)]
struct RejectedRecord {
    path: PathBuf,
    reason: &'static str,
    sharpness_score: Option<f64>,
}

#[derive(Debug, serde::Serialize)]
struct Output {
    schema_version: u32,
    generated_at: String,
    source_dir: PathBuf,
    min_sharpness: f64,
    images: Vec<ImageRecord>,
    rejected: Vec<RejectedRecord>,
}

fn parse_args() -> Result<Args, String> {
    let mut input_dir: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut min_sharpness = DEFAULT_MIN_SHARPNESS;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => return Err(String::new()),
            "--out" => {
                let value = args.next().ok_or("--out needs a value")?;
                out = Some(PathBuf::from(value));
            }
            "--min-sharpness" => {
                let value = args.next().ok_or("--min-sharpness needs a value")?;
                min_sharpness = value
                    .parse::<f64>()
                    .ok()
                    .filter(|v| v.is_finite() && *v >= 0.0)
                    .ok_or(format!("invalid --min-sharpness: {value}"))?;
            }
            flag if flag.starts_with("--") => return Err(format!("unknown option: {flag}")),
            _ if input_dir.is_some() => return Err(format!("unexpected argument: {arg}")),
            _ => input_dir = Some(PathBuf::from(arg)),
        }
    }

    let input_dir = input_dir.ok_or("missing <INPUT_DIR>")?;
    let input_dir = std::path::absolute(&input_dir).unwrap_or(input_dir);
    if !input_dir.is_dir() {
        return Err(format!(
            "input directory does not exist: {}",
            input_dir.display()
        ));
    }

    let out = out.unwrap_or_else(|| input_dir.join(RUN_DIR).join("filtered_images.json"));

    Ok(Args {
        input_dir,
        out,
        min_sharpness,
    })
}

fn find_and_send(directory: &Path) -> Vec<ImgPair> {
    let mut imagegroup: HashMap<(PathBuf, String), ImgPair> = HashMap::new();

    let walker = WalkDir::new(directory).into_iter().filter_entry(|e| {
        !(e.file_type().is_dir() && SKIP_DIRS.iter().any(|d| e.file_name() == *d))
    });

    for entry in walker.filter_map(|e| e.ok()) {
        let path = entry.into_path();

        if path.is_file() {
            if let Some(extension) = path.extension() {
                let ext_str = extension.to_string_lossy().to_lowercase();

                if matches!(ext_str.as_str(), "jpg" | "jpeg" | "arw") {
                    if let (Some(parent), Some(file_stem)) = (path.parent(), path.file_stem()) {
                        let key = (
                            parent.to_path_buf(),
                            file_stem.to_string_lossy().to_string(),
                        );
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

fn calculate_sharpness(path: &PathBuf) -> Option<f64> {
    let ext = path
        .extension()
        .unwrap_or_default()
        .to_string_lossy()
        .to_lowercase();

    let img = if ext == "arw" {
        arw::extract_luma_from_arw(path)?
    } else {
        image::open(path).ok()?.into_luma8()
    };

    let (width, height) = img.dimensions();

    let mut laplacian_values = Vec::with_capacity((width * height) as usize);

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let top = img.get_pixel(x, y - 1)[0] as f64;
            let bottom = img.get_pixel(x, y + 1)[0] as f64;
            let left = img.get_pixel(x - 1, y)[0] as f64;
            let right = img.get_pixel(x + 1, y)[0] as f64;
            let center = img.get_pixel(x, y)[0] as f64;
            let laplacian = (top + bottom + left + right - 4.0 * center).abs();
            laplacian_values.push(laplacian);
        }
    }

    let count = laplacian_values.len() as f64;
    if count == 0.0 {
        return Some(0.0);
    }

    let mean = laplacian_values.iter().sum::<f64>() / count;

    let variance = laplacian_values
        .iter()
        .map(|value| {
            let diff = value - mean;
            diff * diff
        })
        .sum::<f64>()
        / count;
    Some(variance)
}

fn read_exif(path: &Path) -> Option<Exif> {
    let file = File::open(path).ok()?;
    let mut bufreader = BufReader::new(&file);
    Reader::new().read_from_container(&mut bufreader).ok()
}

fn display_field(exif: &Exif, tag: Tag) -> Option<String> {
    exif.get_field(tag, In::PRIMARY)
        .map(|f| f.display_value().with_unit(exif).to_string())
}

// raw ascii value ("2026:06:01 10:15:22", "45") so python can parse it without display quoting
fn ascii_field(exif: &Exif, tag: Tag) -> Option<String> {
    match &exif.get_field(tag, In::PRIMARY)?.value {
        Value::Ascii(parts) => parts
            .first()
            .map(|s| String::from_utf8_lossy(s).trim().to_string())
            .filter(|s| !s.is_empty()),
        _ => None,
    }
}

fn extract_metadata(
    pair: &ImgPair,
    min_sharpness: f64,
) -> Option<Result<ImageRecord, RejectedRecord>> {
    let target_path = pair.jpeg.as_ref().or(pair.arw.as_ref())?;

    let Some(sharpness) = calculate_sharpness(target_path) else {
        println!("Skipping {}: could not decode image", target_path.display());
        return Some(Err(RejectedRecord {
            path: target_path.clone(),
            reason: "decode_error",
            sharpness_score: None,
        }));
    };

    if sharpness < min_sharpness {
        println!(
            "Skipping {} due to low sharpness: {}",
            target_path.display(),
            sharpness
        );
        return Some(Err(RejectedRecord {
            path: target_path.clone(),
            reason: "low_sharpness",
            sharpness_score: Some(sharpness),
        }));
    }

    // missing/unreadable exif keeps the image with null fields
    let exif = read_exif(target_path);
    let exif = exif.as_ref();

    let rawholder = if pair.jpeg.is_some() {
        pair.arw.clone()
    } else {
        None
    };

    Some(Ok(ImageRecord {
        id: String::new(),
        path: target_path.clone(),
        arw_path: rawholder,
        focal_length: exif.and_then(|e| display_field(e, Tag::FocalLength)),
        iso: exif.and_then(|e| display_field(e, Tag::PhotographicSensitivity)),
        shutter_speed: exif.and_then(|e| display_field(e, Tag::ExposureTime)),
        date_time: exif.and_then(|e| ascii_field(e, Tag::DateTime)),
        date_time_original: exif.and_then(|e| ascii_field(e, Tag::DateTimeOriginal)),
        subsec_time_original: exif.and_then(|e| ascii_field(e, Tag::SubSecTimeOriginal)),
        sharpness_score: sharpness,
    }))
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(args) => args,
        Err(msg) => {
            if msg.is_empty() {
                println!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            eprintln!("error: {msg}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };

    let run_dir = args
        .out
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
        .to_path_buf();
    if let Err(e) = std::fs::create_dir_all(&run_dir) {
        eprintln!(
            "error: cannot create output directory {}: {e}",
            run_dir.display()
        );
        return ExitCode::FAILURE;
    }

    println!("Scanning directory: {}", args.input_dir.display());

    let image_paths = find_and_send(&args.input_dir);

    println!(
        "Found {} image files. Extracting metadata...",
        image_paths.len()
    );

    let discovered_filename = run_dir.join("discovered_images.txt");
    let discovered_file =
        File::create(&discovered_filename).expect("Failed to create discovered file");
    let mut discovered_writer = BufWriter::new(discovered_file);

    for pair in &image_paths {
        if let Some(jpg) = &pair.jpeg {
            writeln!(discovered_writer, "JPEG: {}", jpg.display())
                .expect("Failed to write to file");
        }
        if let Some(arw) = &pair.arw {
            writeln!(discovered_writer, "RAW:  {}", arw.display())
                .expect("Failed to write to file");
        }
        writeln!(discovered_writer, "---").expect("Failed to write to file");
    }
    discovered_writer.flush().expect("Failed to write to file");
    println!(
        "Discovered images saved to {}",
        discovered_filename.display()
    );
    println!("Extracting metadata across all cores...");

    let (mut images, mut rejected): (Vec<ImageRecord>, Vec<RejectedRecord>) = image_paths
        .par_iter()
        .filter_map(|pair| extract_metadata(pair, args.min_sharpness))
        .partition_map(|r| match r {
            Ok(record) => rayon::iter::Either::Left(record),
            Err(record) => rayon::iter::Either::Right(record),
        });

    // deterministic output: sort by path, then assign ids
    images.sort_by(|a, b| a.path.cmp(&b.path));
    rejected.sort_by(|a, b| a.path.cmp(&b.path));
    for (i, record) in images.iter_mut().enumerate() {
        record.id = format!("img_{:06}", i + 1);
    }

    let output = Output {
        schema_version: SCHEMA_VERSION,
        generated_at: humantime::format_rfc3339_seconds(SystemTime::now()).to_string(),
        source_dir: args.input_dir,
        min_sharpness: args.min_sharpness,
        images,
        rejected,
    };

    let json_file = File::create(&args.out).expect("Failed to create JSON file");
    serde_json::to_writer_pretty(BufWriter::new(json_file), &output).expect("Failed to write JSON");
    println!(
        "Kept {} images, rejected {}. Structured data successfully serialized to {}",
        output.images.len(),
        output.rejected.len(),
        args.out.display()
    );
    ExitCode::SUCCESS
}
