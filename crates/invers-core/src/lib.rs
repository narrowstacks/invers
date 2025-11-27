//! Invers Core Library
//!
//! Core functionality for film negative to positive conversion.

pub mod auto_adjust;
pub mod color;
pub mod config;
pub mod decoders;
pub mod diagnostics;
pub mod exporters;
pub mod models;
pub mod pipeline;
pub mod presets;
pub mod testing;

// Re-export commonly used types
pub use color::{Hsl, Lab};
pub use decoders::{decode_image_from_bytes, DecodedImage, ImageFormat};
pub use exporters::export_tiff16_to_bytes;
pub use models::{BaseEstimation, ConvertOptions, FilmPreset, HslAdjustments, ScanProfile};
pub use pipeline::{process_image, ProcessedImage};
pub use presets::{load_film_preset_from_str, load_scan_profile_from_str};
