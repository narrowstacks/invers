//! Image processing and input handling.

mod input;
mod single;

pub use input::{determine_output_path, expand_inputs, SUPPORTED_EXTENSIONS};
pub use single::{make_base_from_rgb, process_single_image};
