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

// Re-export commonly used types
pub use color::{Hsl, Lab};
pub use models::{BaseEstimation, ConvertOptions, FilmPreset, HslAdjustments, ScanProfile};
