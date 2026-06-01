use crossbeam_channel::bounded;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::thread;
use walkdir::WalkDir;
use zune_jpeg::JpegDecoder;

#[derive(Debug, Clone)]
struct FileGroup {
    name: String,
    jpegpath: Option<PathBuf>,
    rawpath: Option<PathBuf>,
}

#[derive(Debug)]
struct readimage {
    group: FileGroup,
    pixels: Vec<u8>,
    width: u32,
    height: u32,
}


fn main() {
    println!("Photoholic Image Reader Starting (CPU Mode)");

    let (tx_discovery, rx_decode) = bounded::<FileGroup>(64);
    let (tx_decode, rx_inference) = bounded::<readimage>(64);
    let targetdir = "./images";

    // Spawn the discovery thread
    let discovery_handle = thread::spawn(move || {
        println!("[Stage 1] Starting file discovery in '{}'", targetdir);
        
        let mut file_groups: HashMap<String, FileGroup> = HashMap::new();
        
        for entry in WalkDir::new(targetdir).into_iter().filter_map(|e| e.ok()) {
            
            let path = entry.path();
            
            if !path.is_file() {
                continue;
            }
            
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
            let basename = path.file_stem().and_then(|n| n.to_str()).unwrap_or("").to_string();
            
            if ext = = "jpg" || ext == "jpeg" {
                let group = file_groups.entry(basename.clone()).or_insert(FileGroup {
                    name: basename.clone(),
                    jpegpath: None,
                    rawpath: None,
                });

                if ext == "arw" {
                    group.rawpath = Some(path.to_path_buf());
                } 
                else {
                    group.jpegpath = Some(path.to_path_buf());
                }
            }
        }

        let mut count = 0;
        for (_, group) in file_groups {

            if tx_discovery.send(group).is_err() {
                println!("[Stage 1] Discovery channel closed, stopping discovery thread.");
                break;
            }
            count += 1;
        }

        println!("[Stage 1] Discovered {} files.", count);
        drop(tx_discovery);
    });
}
