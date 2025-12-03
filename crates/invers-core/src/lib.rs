//! Positize Core Library
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

// GPU acceleration module (optional, enabled with "gpu" feature)
#[cfg(feature = "gpu")]
pub mod gpu;

// Re-export commonly used types
pub use color::{Hsl, Lab};
pub use models::{BaseEstimation, ConvertOptions, FilmPreset, HslAdjustments, ScanProfile};

// Re-export GPU functions when available
#[cfg(feature = "gpu")]
pub use gpu::{gpu_info, is_gpu_available, process_image_gpu};
