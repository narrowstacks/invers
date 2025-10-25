//! Image viewer component
//!
//! Displays the loaded image with zoom, pan, and overlay features.

use crate::app_state::PreviewSettings;

/// Image viewer widget
///
/// TODO: In M2, integrate with Qt's QGraphicsView or custom OpenGL viewer
pub struct ImageViewer {
    settings: PreviewSettings,
}

impl ImageViewer {
    pub fn new() -> Self {
        Self {
            settings: PreviewSettings::default(),
        }
    }

    pub fn settings(&self) -> &PreviewSettings {
        &self.settings
    }

    pub fn settings_mut(&mut self) -> &mut PreviewSettings {
        &mut self.settings
    }

    /// Set zoom level
    pub fn set_zoom(&mut self, zoom: f32) {
        self.settings.zoom = zoom.max(0.1).min(10.0);
    }

    /// Set pan offset
    pub fn set_pan(&mut self, x: f32, y: f32) {
        self.settings.pan_offset = (x, y);
    }

    /// Reset view to default
    pub fn reset_view(&mut self) {
        self.settings.zoom = 1.0;
        self.settings.pan_offset = (0.0, 0.0);
    }
}

/// Histogram overlay
pub struct Histogram;

impl Histogram {
    // TODO: Compute and display histogram
    // TODO: Per-channel histograms
    // TODO: RGB overlay mode
}

/// Pixel probe tool
pub struct PixelProbe {
    /// Current pixel position
    pub position: Option<(u32, u32)>,
}

impl PixelProbe {
    pub fn new() -> Self {
        Self { position: None }
    }

    // TODO: Display RGB values at cursor position
    // TODO: Show surrounding pixel grid (5x5 or 9x9)
}
