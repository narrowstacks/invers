//! Scan profile types for capture source characteristics.

mod hints;
mod hsl;
mod mask;
mod profile;

#[cfg(test)]
mod tests;

pub use hints::{DemosaicHints, WhiteBalanceHints};
pub use hsl::HslAdjustments;
pub use mask::MaskProfile;
pub use profile::ScanProfile;
