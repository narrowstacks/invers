//! Preview rendering
//!
//! Real-time preview generation at reduced resolution.

use positize_core::{BaseEstimation, FilmPreset};

/// Preview renderer
pub struct PreviewRenderer {
    /// Preview resolution scale (0.0-1.0)
    scale: f32,

    /// Whether preview is currently valid
    valid: bool,
}

impl PreviewRenderer {
    pub fn new(scale: f32) -> Self {
        Self {
            scale: scale.clamp(0.1, 1.0),
            valid: false,
        }
    }

    /// Set preview scale
    pub fn set_scale(&mut self, scale: f32) {
        self.scale = scale.clamp(0.1, 1.0);
        self.invalidate();
    }

    /// Invalidate the current preview (force re-render)
    pub fn invalidate(&mut self) {
        self.valid = false;
    }

    /// Check if preview is valid
    pub fn is_valid(&self) -> bool {
        self.valid
    }

    /// Render preview with current settings
    ///
    /// TODO: In M2, integrate with positize_core pipeline
    /// - Load image at reduced resolution
    /// - Apply base estimation and preset
    /// - Render to display buffer
    pub fn render(
        &mut self,
        _image_path: &std::path::Path,
        _base: Option<&BaseEstimation>,
        _preset: Option<&FilmPreset>,
    ) -> Result<PreviewImage, String> {
        // Placeholder
        self.valid = true;
        Err("Preview rendering not yet implemented".to_string())
    }
}

/// Preview image data
pub struct PreviewImage {
    /// Image width
    pub width: u32,

    /// Image height
    pub height: u32,

    /// RGB8 data for display
    pub data: Vec<u8>,
}

/// Background preview renderer (runs in separate thread)
pub struct BackgroundRenderer;

impl BackgroundRenderer {
    // TODO: Implement background rendering with async updates
    // TODO: Use channels to communicate with UI thread
    // TODO: Cancel previous renders when settings change
}
