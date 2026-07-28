use std::collections::HashMap;
use std::path::PathBuf;
use walkdir::WalkDir;

#[derive(Default, Debug)]
struct ImgType {
    jpeg: Option<PathBuf>,
    raw: Option<PathBuf>,
}

fn find_and_send(directory: &str) -> Vec<PathBuf> {
    let mut imagegroup : HashMap<String, ImgType> = HashMap::new();

    for entry in WalkDir::new(directory).into_iter().filter_map(|e| e.ok()) {
        let path = entry.into_path();

        if path.is_file() {
            if let (Some(stem), Some(ext)) = (path.file_stem(), path.extension()) {
                let stem_str = stem.to_string_lossy().to_string();
                let ext_str = ext.to_string_lossy().to_lowercase();

                let pair = imagegroup.entry(stem_str).or_default();

                match ext_str.as_str() {
                    "jpg" | "jpeg" => pair.jpeg = Some(path),
                    "raw" => pair.raw = Some(path),
                    _ => {}
            }
        }
    }
   }
    let mut result = Vec::new();
    for (_, pair) in imagegroup {
        if let Some(path) = pair.jpeg {
            result.push(path);
        }
        if let Some(path) = pair.raw {
            result.push(path);
        }
    }
    result
}

fn main() {
    let target = "";
    println!("Scanning directory: {}", target);

    let image_paths = find_and_send(target);

    println!("Found {} image files:", image_paths.len());
    
    for file in image_paths {
        println!(" - {:?}", file.display());
    }
}
