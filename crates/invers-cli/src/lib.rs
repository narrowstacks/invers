//! Shared utilities for invers-cli
//!
//! This module provides reusable functions and utilities that can be
//! shared between the CLI and GUI applications.

pub mod args;
pub mod builders;
pub mod parsers;
pub mod processing;
pub mod types;

// Re-export commonly used items at the crate root for convenience
pub use builders::{build_cb_options, build_convert_options_full_with_gpu};
pub use parsers::{parse_base_rgb, parse_inversion_mode, parse_pipeline_mode, parse_roi};
pub use processing::{
    determine_output_path, expand_inputs, make_base_from_rgb, process_single_image,
    SUPPORTED_EXTENSIONS,
};
pub use types::{ProcessingParams, WhiteBalance, WhiteBalanceSettings};
