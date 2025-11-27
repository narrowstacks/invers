//! Application state using Sycamore signals
//!
//! This module defines all the reactive state for the invers-web application.
//! State is organized into logical groups for UI binding.

use invers_core::decoders::DecodedImage;
use invers_core::models::{BaseEstimation, FilmPreset};
use invers_core::pipeline::ProcessedImage;
use sycamore::prelude::*;
use std::rc::Rc;
use std::cell::RefCell;

/// Maximum preview dimension (width or height)
pub const MAX_PREVIEW_SIZE: u32 = 1024;

/// Maximum image size in megapixels
pub const MAX_IMAGE_MEGAPIXELS: u32 = 50;

/// UI interaction mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InteractionMode {
    /// Normal viewing mode
    Normal,
    /// Eyedropper mode for picking base color
    EyedropperBase,
    /// Eyedropper mode for picking white balance point
    EyedropperWhiteBalance,
}

impl Default for InteractionMode {
    fn default() -> Self {
        Self::Normal
    }
}

/// Processing status
#[derive(Debug, Clone, PartialEq)]
pub enum ProcessingStatus {
    /// No image loaded
    Empty,
    /// Processing in progress
    Processing(String),
    /// Processing complete
    Ready,
    /// Error occurred
    Error(String),
}

impl Default for ProcessingStatus {
    fn default() -> Self {
        Self::Empty
    }
}

/// Wrapper for image data that can be shared across signals
#[derive(Debug, Clone, Default)]
pub struct ImageHolder {
    pub image: Rc<RefCell<Option<DecodedImage>>>,
}

impl ImageHolder {
    pub fn new() -> Self {
        Self { image: Rc::new(RefCell::new(None)) }
    }

    pub fn set(&self, img: Option<DecodedImage>) {
        *self.image.borrow_mut() = img;
    }

    pub fn get(&self) -> Option<DecodedImage> {
        self.image.borrow().clone()
    }

    pub fn is_some(&self) -> bool {
        self.image.borrow().is_some()
    }
}

/// Wrapper for processed image data
#[derive(Debug, Clone, Default)]
pub struct ProcessedHolder {
    pub image: Rc<RefCell<Option<ProcessedImage>>>,
}

impl ProcessedHolder {
    pub fn new() -> Self {
        Self { image: Rc::new(RefCell::new(None)) }
    }

    pub fn set(&self, img: Option<ProcessedImage>) {
        *self.image.borrow_mut() = img;
    }

    pub fn get(&self) -> Option<ProcessedImage> {
        self.image.borrow().clone()
    }

    pub fn is_some(&self) -> bool {
        self.image.borrow().is_some()
    }
}

/// Application state context
#[derive(Clone)]
pub struct AppState {
    // === Image Data ===
    /// Original full-resolution decoded image
    pub original_image: Signal<ImageHolder>,

    /// Downsampled preview image (max 1024px)
    pub preview_image: Signal<ImageHolder>,

    /// Processed preview for display
    pub processed_preview: Signal<ProcessedHolder>,

    /// Original filename
    pub filename: Signal<String>,

    // === Base Estimation ===
    /// Film base RGB values (0.0-1.0)
    pub base_r: Signal<f32>,
    pub base_g: Signal<f32>,
    pub base_b: Signal<f32>,

    /// Base offsets (adjustments to base, -0.2 to +0.2)
    pub base_offset_r: Signal<f32>,
    pub base_offset_g: Signal<f32>,
    pub base_offset_b: Signal<f32>,

    // === White Balance ===
    /// White balance RGB multipliers (0.1-3.0)
    pub wb_r: Signal<f32>,
    pub wb_g: Signal<f32>,
    pub wb_b: Signal<f32>,

    // === Exposure ===
    /// Exposure compensation (0.1-3.0)
    pub exposure: Signal<f32>,

    // === Tone Curve ===
    /// Skip tone curve
    pub skip_tone_curve: Signal<bool>,

    /// Tone curve strength (0.0-1.0)
    pub tone_curve_strength: Signal<f32>,

    // === Color Matrix ===
    /// Skip color matrix
    pub skip_color_matrix: Signal<bool>,

    /// 3x3 color correction matrix
    pub color_matrix: Signal<[[f32; 3]; 3]>,

    // === Auto Adjustments ===
    /// Enable auto-levels
    pub enable_auto_levels: Signal<bool>,

    /// Auto-levels clip percentage
    pub auto_levels_clip: Signal<f32>,

    /// Enable auto-color
    pub enable_auto_color: Signal<bool>,

    /// Auto-color strength
    pub auto_color_strength: Signal<f32>,

    /// Enable auto-exposure
    pub enable_auto_exposure: Signal<bool>,

    // === Preset ===
    /// Selected preset slug
    pub selected_preset: Signal<String>,

    // === UI State ===
    /// Current interaction mode
    pub interaction_mode: Signal<InteractionMode>,

    /// Processing status
    pub status: Signal<ProcessingStatus>,

    /// Status message for display
    pub status_message: Signal<String>,

    /// Trigger signal for re-render
    pub render_trigger: Signal<u32>,
}

impl AppState {
    /// Create new application state with default values
    pub fn new() -> Self {
        Self {
            // Image data
            original_image: create_signal(ImageHolder::new()),
            preview_image: create_signal(ImageHolder::new()),
            processed_preview: create_signal(ProcessedHolder::new()),
            filename: create_signal(String::new()),

            // Base estimation - defaults will be set when image loads
            base_r: create_signal(0.5),
            base_g: create_signal(0.35),
            base_b: create_signal(0.25),

            // Base offsets
            base_offset_r: create_signal(0.0),
            base_offset_g: create_signal(0.0),
            base_offset_b: create_signal(0.0),

            // White balance
            wb_r: create_signal(1.0),
            wb_g: create_signal(1.0),
            wb_b: create_signal(1.0),

            // Exposure
            exposure: create_signal(1.0),

            // Tone curve
            skip_tone_curve: create_signal(false),
            tone_curve_strength: create_signal(0.55),

            // Color matrix - identity matrix
            skip_color_matrix: create_signal(false),
            color_matrix: create_signal([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]),

            // Auto adjustments
            enable_auto_levels: create_signal(true),
            auto_levels_clip: create_signal(0.25),
            enable_auto_color: create_signal(false),
            auto_color_strength: create_signal(0.6),
            enable_auto_exposure: create_signal(true),

            // Preset
            selected_preset: create_signal("optimized-standard".to_string()),

            // UI state
            interaction_mode: create_signal(InteractionMode::Normal),
            status: create_signal(ProcessingStatus::Empty),
            status_message: create_signal(String::new()),
            render_trigger: create_signal(0),
        }
    }

    /// Reset white balance to neutral
    pub fn reset_white_balance(&self) {
        self.wb_r.set(1.0);
        self.wb_g.set(1.0);
        self.wb_b.set(1.0);
    }

    /// Reset exposure to default
    pub fn reset_exposure(&self) {
        self.exposure.set(1.0);
    }

    /// Reset color matrix to identity
    pub fn reset_color_matrix(&self) {
        self.color_matrix.set([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]);
    }

    /// Reset base offsets to zero
    pub fn reset_base_offsets(&self) {
        self.base_offset_r.set(0.0);
        self.base_offset_g.set(0.0);
        self.base_offset_b.set(0.0);
    }

    /// Get the current base estimation
    pub fn get_base_estimation(&self) -> BaseEstimation {
        BaseEstimation {
            roi: None,
            medians: [self.base_r.get(), self.base_g.get(), self.base_b.get()],
            noise_stats: None,
            auto_estimated: true,
        }
    }

    /// Update base from eyedropper sample
    pub fn set_base_from_sample(&self, r: f32, g: f32, b: f32) {
        self.base_r.set(r);
        self.base_g.set(g);
        self.base_b.set(b);
        self.interaction_mode.set(InteractionMode::Normal);
    }

    /// Update white balance from neutral point sample
    pub fn set_wb_from_neutral_point(&self, r: f32, g: f32, b: f32) {
        // Calculate multipliers to make this point neutral (gray)
        let avg = (r + g + b) / 3.0;
        if avg > 0.01 {
            self.wb_r.set((avg / r).clamp(0.1, 3.0));
            self.wb_g.set((avg / g).clamp(0.1, 3.0));
            self.wb_b.set((avg / b).clamp(0.1, 3.0));
        }
        self.interaction_mode.set(InteractionMode::Normal);
    }

    /// Apply a film preset to the current state
    pub fn apply_preset(&self, preset: &FilmPreset) {
        // Apply base offsets
        self.base_offset_r.set(preset.base_offsets[0]);
        self.base_offset_g.set(preset.base_offsets[1]);
        self.base_offset_b.set(preset.base_offsets[2]);

        // Apply color matrix
        self.color_matrix.set(preset.color_matrix);

        // Apply tone curve settings
        self.tone_curve_strength.set(preset.tone_curve.strength);
    }

    /// Set status to processing
    pub fn set_processing(&self, message: &str) {
        self.status.set(ProcessingStatus::Processing(message.to_string()));
        self.status_message.set(message.to_string());
    }

    /// Set status to ready
    pub fn set_ready(&self) {
        self.status.set(ProcessingStatus::Ready);
        self.status_message.set("Ready".to_string());
    }

    /// Set status to error
    pub fn set_error(&self, message: &str) {
        self.status.set(ProcessingStatus::Error(message.to_string()));
        self.status_message.set(message.to_string());
    }

    /// Check if an image is loaded
    pub fn has_image(&self) -> bool {
        self.original_image.get_clone().is_some()
    }

    /// Trigger a re-render
    pub fn trigger_render(&self) {
        let current = self.render_trigger.get();
        self.render_trigger.set(current.wrapping_add(1));
    }

    /// Get image dimensions string
    pub fn get_dimensions_string(&self) -> String {
        match self.original_image.get_clone().get() {
            Some(img) => {
                let mp = (img.width as f64 * img.height as f64) / 1_000_000.0;
                format!("{}x{} ({:.1} MP)", img.width, img.height, mp)
            }
            None => String::new(),
        }
    }

    /// Get preview dimensions string
    pub fn get_preview_dimensions_string(&self) -> String {
        match self.preview_image.get_clone().get() {
            Some(img) => format!("Preview: {}x{}", img.width, img.height),
            None => String::new(),
        }
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}
