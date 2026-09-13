use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::PathBuf;
use exif::{In, Reader, Tag};
use image::GrayImage;

pub fn extract_luma_from_arw(arw_path: &PathBuf) -> Option<GrayImage> {

    let file = File::open(arw_path).ok()?;
    let mut bufreader = BufReader::new(file);
    let exifreader = Reader::new();

    let exif = exifreader.read_from_container(&mut bufreader).ok()?;

    let offset = exif.get_field(Tag::JPEGInterchangeFormat, In::PRIMARY)
        .or_else(|| exif.get_field(Tag::JPEGInterchangeFormat, In(1)))
        .and_then(|f| f.value.get_uint(0))?;

    let length = exif.get_field(Tag::JPEGInterchangeFormatLength, In::PRIMARY)
        .or_else(|| exif.get_field(Tag::JPEGInterchangeFormatLength, In(1)))
        .and_then(|f| f.value.get_uint(0))?;

    bufreader.seek(SeekFrom::Start(offset as u64)).ok()?;
    let mut jpeg_bytes = vec![0u8; length as usize];
    bufreader.read_exact(&mut jpeg_bytes).ok()?;

    image::load_from_memory(&jpeg_bytes).ok().map(|img| img.into_luma8())
}