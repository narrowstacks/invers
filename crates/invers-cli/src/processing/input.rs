//! Input file handling and path utilities.

use std::path::{Path, PathBuf};

/// Supported image extensions for batch processing
pub const SUPPORTED_EXTENSIONS: &[&str] = &["tif", "tiff", "png", "dng"];

/// Determine output path based on input, output dir, and export format
///
/// # Arguments
/// * `input` - Input file path
/// * `out` - Optional output directory or file path
/// * `export` - Export format ("tiff16" or "dng")
///
/// # Returns
/// The full output path for the converted image
pub fn determine_output_path(
    input: &Path,
    out: &Option<PathBuf>,
    export: &str,
) -> Result<PathBuf, String> {
    let extension = match export {
        "tiff16" | "tiff" => "tif",
        "dng" => "dng",
        _ => "tif",
    };

    if let Some(out_path) = out {
        // If out is a directory, use input filename with new extension
        if out_path.is_dir() {
            let filename = input
                .file_stem()
                .ok_or("Invalid input filename")?
                .to_string_lossy();
            Ok(out_path.join(format!("{}_positive.{}", filename, extension)))
        } else {
            // Use the specified path as-is
            Ok(out_path.clone())
        }
    } else {
        // Use input directory with modified filename
        let filename = input
            .file_stem()
            .ok_or("Invalid input filename")?
            .to_string_lossy();
        let parent = input.parent().unwrap_or(Path::new("."));
        Ok(parent.join(format!("{}_positive.{}", filename, extension)))
    }
}

/// Expand a list of inputs (files and directories) into a list of image files.
///
/// Directories are scanned for supported image files (.tif, .tiff, .png, .dng).
/// If `recursive` is true, subdirectories are also scanned.
pub fn expand_inputs(inputs: &[PathBuf], recursive: bool) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();

    for input in inputs {
        if input.is_dir() {
            collect_images_from_dir(input, recursive, &mut files)?;
        } else if input.is_file() {
            files.push(input.clone());
        } else {
            return Err(format!("Path not found: {}", input.display()));
        }
    }

    // Sort for consistent ordering
    files.sort();
    Ok(files)
}

/// Recursively collect image files from a directory.
fn collect_images_from_dir(
    dir: &Path,
    recursive: bool,
    files: &mut Vec<PathBuf>,
) -> Result<(), String> {
    let entries = std::fs::read_dir(dir)
        .map_err(|e| format!("Failed to read directory {}: {}", dir.display(), e))?;

    for entry in entries {
        let entry = entry.map_err(|e| format!("Error reading directory entry: {}", e))?;
        let path = entry.path();

        if path.is_dir() && recursive {
            collect_images_from_dir(&path, recursive, files)?;
        } else if path.is_file() {
            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                if SUPPORTED_EXTENSIONS.contains(&ext.to_lowercase().as_str()) {
                    files.push(path);
                }
            }
        }
    }
    Ok(())
}
