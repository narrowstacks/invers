//! RAW image decoder (delegates to invers-raw/LibRaw)

use std::path::Path;

use super::{detect_monochrome, DecodedImage};

/// Decode a RAW file using invers-raw (LibRaw wrapper)
pub(crate) fn decode_raw<P: AsRef<Path>>(path: P) -> Result<DecodedImage, String> {
    let raw = invers_raw::decode_raw(path)?;

    let is_monochrome = detect_monochrome(&raw.data, raw.width, raw.height);

    Ok(DecodedImage {
        width: raw.width,
        height: raw.height,
        data: raw.data,
        channels: raw.channels,
        black_level: raw.black_level,
        white_level: raw.white_level,
        color_matrix: raw.color_matrix,
        source_is_grayscale: false,
        is_monochrome,
    })
}
