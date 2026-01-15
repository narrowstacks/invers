//! Parsing functions for CLI arguments.

mod base;
mod cb;
mod pipeline;

pub use base::{parse_base_rgb, parse_roi};
pub use cb::{
    parse_cb_color_model, parse_cb_enhanced_profile, parse_cb_film_character,
    parse_cb_tone_profile, parse_cb_wb_preset,
};
pub use pipeline::{parse_inversion_mode, parse_neutral_roi, parse_pipeline_mode};
