//! Positize GUI Application
//!
//! Interactive GUI for film negative to positive conversion using egui.

use eframe::egui;
use invers_core::{
    decoders::{decode_image, DecodedImage},
    models::{BaseEstimation, ConvertOptions, FilmPreset, ToneCurveParams},
    pipeline::{estimate_base, process_image, ProcessedImage},
};
use std::path::PathBuf;

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 800.0])
            .with_title("Invers - Film Negative Converter"),
        ..Default::default()
    };

    eframe::run_native(
        "Invers",
        options,
        Box::new(|_cc| Ok(Box::new(InversApp::default()))),
    )
}

struct InversApp {
    // Image data
    loaded_image: Option<DecodedImage>,
    preview_image: Option<DecodedImage>, // Downsampled for fast preview
    loaded_path: Option<PathBuf>,
    processed_result: Option<ProcessedImage>,
    display_texture: Option<egui::TextureHandle>,

    // Processing parameters
    base_estimation: Option<BaseEstimation>,
    base_r: f32,
    base_g: f32,
    base_b: f32,

    // Film preset parameters
    base_offset_r: f32,
    base_offset_g: f32,
    base_offset_b: f32,

    // White balance multipliers
    white_balance_r: f32,
    white_balance_g: f32,
    white_balance_b: f32,

    exposure_compensation: f32,
    tone_curve_strength: f32,
    skip_tone_curve: bool,
    skip_color_matrix: bool,

    // Color matrix (3x3)
    color_matrix: [[f32; 3]; 3],

    // UI state
    show_color_matrix: bool,
    processing_needed: bool,
    error_message: Option<String>,
    eyedropper_mode: bool,
    white_balance_mode: bool,
}

impl Default for InversApp {
    fn default() -> Self {
        Self {
            loaded_image: None,
            preview_image: None,
            loaded_path: None,
            processed_result: None,
            display_texture: None,

            base_estimation: None,
            base_r: 0.5,
            base_g: 0.5,
            base_b: 0.5,

            base_offset_r: 0.0,
            base_offset_g: 0.0,
            base_offset_b: 0.0,

            white_balance_r: 1.0,
            white_balance_g: 1.0,
            white_balance_b: 1.0,

            exposure_compensation: 1.0,
            tone_curve_strength: 0.5,
            skip_tone_curve: false,
            skip_color_matrix: false,

            // Identity matrix by default
            color_matrix: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],

            show_color_matrix: false,
            processing_needed: false,
            error_message: None,
            eyedropper_mode: false,
            white_balance_mode: false,
        }
    }
}

impl eframe::App for InversApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Top menu bar
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Open Image...").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("Images", &["tif", "tiff", "png"])
                            .pick_file()
                        {
                            self.load_image(path);
                        }
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Quit").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });

                ui.menu_button("View", |ui| {
                    ui.checkbox(&mut self.show_color_matrix, "Show Color Matrix");
                });
            });
        });

        // Left panel: Image preview
        egui::SidePanel::left("preview_panel")
            .default_width(1000.0)
            .resizable(true)
            .show(ctx, |ui| {
                ui.heading("Preview");
                ui.separator();

                egui::ScrollArea::both()
                    .id_salt("preview_scroll")
                    .show(ui, |ui| {
                        self.show_image_preview(ui, ctx);
                    });
            });

        // Right panel: Controls
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Parameters");
            ui.separator();

            egui::ScrollArea::vertical()
                .id_salt("controls_scroll")
                .show(ui, |ui| {
                    self.show_controls(ui);
                });
        });

        // Process image if parameters changed
        if self.processing_needed && self.loaded_image.is_some() {
            self.process_current_image();
            self.processing_needed = false;
        }

        // Show error message if any
        if self.error_message.is_some() {
            let error = self.error_message.clone().unwrap();
            let mut should_close = false;
            egui::Window::new("Error")
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label(&error);
                    if ui.button("OK").clicked() {
                        should_close = true;
                    }
                });
            if should_close {
                self.error_message = None;
            }
        }
    }
}

impl InversApp {
    fn load_image(&mut self, path: PathBuf) {
        match decode_image(&path) {
            Ok(image) => {
                // Create downsampled preview for fast processing (max 1024px)
                let preview = downsample_image(&image, 1024);

                eprintln!(
                    "[LOAD] Original: {}x{}, Preview: {}x{}",
                    image.width, image.height, preview.width, preview.height
                );

                // Reset all parameters to defaults for new image
                self.reset_parameters();

                // Auto-estimate base from the loaded image
                match estimate_base(&image, None) {
                    Ok(base) => {
                        self.base_r = base.medians[0];
                        self.base_g = base.medians[1];
                        self.base_b = base.medians[2];
                        eprintln!(
                            "[LOAD] Base estimation: RGB [{:.6}, {:.6}, {:.6}]",
                            base.medians[0], base.medians[1], base.medians[2]
                        );
                        self.base_estimation = Some(base);
                    }
                    Err(e) => {
                        self.error_message = Some(format!("Base estimation failed: {}", e));
                    }
                }

                self.loaded_path = Some(path);
                self.loaded_image = Some(image);
                self.preview_image = Some(preview);
                self.processing_needed = true;
                self.display_texture = None; // Clear old texture
            }
            Err(e) => {
                self.error_message = Some(format!("Failed to load image: {}", e));
            }
        }
    }

    fn reset_parameters(&mut self) {
        // Reset all processing parameters to defaults
        self.base_offset_r = 0.0;
        self.base_offset_g = 0.0;
        self.base_offset_b = 0.0;

        self.white_balance_r = 1.0;
        self.white_balance_g = 1.0;
        self.white_balance_b = 1.0;

        self.exposure_compensation = 1.0;
        self.tone_curve_strength = 0.5;
        self.skip_tone_curve = false;
        self.skip_color_matrix = false;

        // Reset color matrix to identity
        self.color_matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        eprintln!("[LOAD] Reset all parameters to defaults");
    }

    fn process_current_image(&mut self) {
        // Use preview image for fast processing
        let Some(ref image) = self.preview_image else {
            return;
        };

        // Build base estimation from current parameters
        let base = BaseEstimation {
            roi: None,
            medians: [self.base_r, self.base_g, self.base_b],
            noise_stats: None,
            auto_estimated: false,
        };

        // Build film preset with current parameters
        let preset = FilmPreset {
            name: "Custom".to_string(),
            base_offsets: [self.base_offset_r, self.base_offset_g, self.base_offset_b],
            color_matrix: self.color_matrix,
            tone_curve: ToneCurveParams {
                curve_type: "neutral".to_string(),
                strength: self.tone_curve_strength,
                params: std::collections::HashMap::new(),
            },
            notes: None,
        };

        // Build convert options
        let defaults = invers_core::config::pipeline_config_handle()
            .config
            .defaults
            .clone();

        let options = ConvertOptions {
            input_paths: vec![],
            output_dir: PathBuf::from("."),
            output_format: invers_core::models::OutputFormat::Tiff16,
            working_colorspace: "linear-rec2020".to_string(),
            bit_depth_policy: invers_core::models::BitDepthPolicy::MatchInput,
            film_preset: Some(preset),
            scan_profile: None,
            base_estimation: Some(base),
            num_threads: None,
            skip_tone_curve: self.skip_tone_curve || defaults.skip_tone_curve,
            skip_color_matrix: self.skip_color_matrix || defaults.skip_color_matrix,
            exposure_compensation: defaults.exposure_compensation * self.exposure_compensation,
            debug: false,
            enable_auto_levels: defaults.enable_auto_levels,
            auto_levels_clip_percent: defaults.auto_levels_clip_percent,
            enable_auto_color: defaults.enable_auto_color,
            auto_color_strength: defaults.auto_color_strength,
            auto_color_min_gain: defaults.auto_color_min_gain,
            auto_color_max_gain: defaults.auto_color_max_gain,
            base_brightest_percent: defaults.base_brightest_percent,
            base_sampling_mode: defaults.base_sampling_mode,
            inversion_mode: defaults.inversion_mode,
            shadow_lift_mode: defaults.shadow_lift_mode,
            shadow_lift_value: defaults.shadow_lift_value,
            highlight_compression: defaults.highlight_compression,
            enable_auto_exposure: defaults.enable_auto_exposure,
            auto_exposure_target_median: defaults.auto_exposure_target_median,
            auto_exposure_strength: defaults.auto_exposure_strength,
            auto_exposure_min_gain: defaults.auto_exposure_min_gain,
            auto_exposure_max_gain: defaults.auto_exposure_max_gain,
        };

        // Process the image
        match process_image(image.clone(), &options) {
            Ok(mut result) => {
                // Apply white balance to the processed result
                self.apply_white_balance(&mut result);

                self.processed_result = Some(result);
                self.display_texture = None; // Force texture rebuild
            }
            Err(e) => {
                self.error_message = Some(format!("Processing failed: {}", e));
            }
        }
    }

    fn apply_white_balance(&self, result: &mut ProcessedImage) {
        // Skip if white balance is neutral (all 1.0)
        if (self.white_balance_r - 1.0).abs() < 0.001
            && (self.white_balance_g - 1.0).abs() < 0.001
            && (self.white_balance_b - 1.0).abs() < 0.001
        {
            return;
        }

        // Apply white balance multipliers to each pixel
        for pixel in result.data.chunks_exact_mut(3) {
            pixel[0] = (pixel[0] * self.white_balance_r).clamp(0.0, 1.0);
            pixel[1] = (pixel[1] * self.white_balance_g).clamp(0.0, 1.0);
            pixel[2] = (pixel[2] * self.white_balance_b).clamp(0.0, 1.0);
        }
    }

    fn show_image_preview(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        if let Some(ref result) = self.processed_result {
            // Create or update texture
            if self.display_texture.is_none() {
                let color_image = self.create_color_image(result);
                let texture = ctx.load_texture("preview", color_image, Default::default());
                self.display_texture = Some(texture);
            }

            if let Some(ref texture) = self.display_texture {
                let size = texture.size_vec2();
                let available_size = ui.available_size();

                // Scale to fit while maintaining aspect ratio
                let scale = (available_size.x / size.x)
                    .min(available_size.y / size.y)
                    .min(1.0);
                let display_size = size * scale;

                // Make image interactive if any eyedropper is active
                let is_eyedropper_active = self.eyedropper_mode || self.white_balance_mode;
                let response = ui.add(egui::Image::new((texture.id(), display_size)).sense(
                    if is_eyedropper_active {
                        egui::Sense::click()
                    } else {
                        egui::Sense::hover()
                    },
                ));

                // Change cursor when eyedropper is active
                if is_eyedropper_active {
                    ctx.set_cursor_icon(egui::CursorIcon::Crosshair);
                }

                // Handle eyedropper clicks
                if is_eyedropper_active && response.clicked() {
                    if let Some(pos) = response.interact_pointer_pos() {
                        // Calculate position relative to image
                        let image_pos = pos - response.rect.min;

                        // Convert to texture coordinates (0.0-1.0)
                        let tex_x = image_pos.x / display_size.x;
                        let tex_y = image_pos.y / display_size.y;

                        // Sample from the original image at this position
                        if self.eyedropper_mode {
                            self.sample_base_color(tex_x, tex_y);
                            self.eyedropper_mode = false;
                        } else if self.white_balance_mode {
                            self.sample_white_balance(tex_x, tex_y);
                            self.white_balance_mode = false;
                        }
                    }
                }
            }
        } else if self.loaded_image.is_some() {
            ui.label("Processing...");
        } else {
            ui.label("No image loaded. Use File > Open Image to load a negative.");
        }
    }

    fn sample_white_balance(&mut self, tex_x: f32, tex_y: f32) {
        let Some(ref orig_image) = self.loaded_image else {
            return;
        };

        // Map normalized coordinates to original image coordinates
        let x = (tex_x * orig_image.width as f32) as u32;
        let y = (tex_y * orig_image.height as f32) as u32;

        // Sample a 7x7 region around the clicked point for more stable white balance
        let region_size = 7;
        let half_size = region_size / 2;

        let mut samples: Vec<[f32; 3]> = Vec::new();

        for dy in 0..region_size {
            for dx in 0..region_size {
                let sample_x = (x as i32 + dx as i32 - half_size as i32)
                    .clamp(0, orig_image.width as i32 - 1) as u32;
                let sample_y = (y as i32 + dy as i32 - half_size as i32)
                    .clamp(0, orig_image.height as i32 - 1) as u32;

                let idx = ((sample_y * orig_image.width + sample_x) * 3) as usize;
                if idx + 2 < orig_image.data.len() {
                    samples.push([
                        orig_image.data[idx],
                        orig_image.data[idx + 1],
                        orig_image.data[idx + 2],
                    ]);
                }
            }
        }

        // Calculate average RGB
        if !samples.is_empty() {
            let mut sum_r = 0.0;
            let mut sum_g = 0.0;
            let mut sum_b = 0.0;

            for sample in &samples {
                sum_r += sample[0];
                sum_g += sample[1];
                sum_b += sample[2];
            }

            let count = samples.len() as f32;
            let avg_r = sum_r / count;
            let avg_g = sum_g / count;
            let avg_b = sum_b / count;

            // Calculate white balance multipliers
            // Use the brightest channel as reference to avoid darkening
            let max_channel = avg_r.max(avg_g).max(avg_b).max(0.001); // Avoid division by zero

            self.white_balance_r = max_channel / avg_r.max(0.001);
            self.white_balance_g = max_channel / avg_g.max(0.001);
            self.white_balance_b = max_channel / avg_b.max(0.001);

            eprintln!(
                "[WHITE BALANCE] Sampled at ({}, {}) - RGB: [{:.6}, {:.6}, {:.6}]",
                x, y, avg_r, avg_g, avg_b
            );
            eprintln!(
                "[WHITE BALANCE] Multipliers: R={:.3}x G={:.3}x B={:.3}x",
                self.white_balance_r, self.white_balance_g, self.white_balance_b
            );

            self.processing_needed = true;
        }
    }

    fn sample_base_color(&mut self, tex_x: f32, tex_y: f32) {
        let Some(ref orig_image) = self.loaded_image else {
            return;
        };

        // Map normalized coordinates to original image coordinates
        let x = (tex_x * orig_image.width as f32) as u32;
        let y = (tex_y * orig_image.height as f32) as u32;

        // Sample a 5x5 region around the clicked point
        let region_size = 5;
        let half_size = region_size / 2;

        let mut samples: Vec<[f32; 3]> = Vec::new();

        for dy in 0..region_size {
            for dx in 0..region_size {
                let sample_x = (x as i32 + dx as i32 - half_size as i32)
                    .clamp(0, orig_image.width as i32 - 1) as u32;
                let sample_y = (y as i32 + dy as i32 - half_size as i32)
                    .clamp(0, orig_image.height as i32 - 1) as u32;

                let idx = ((sample_y * orig_image.width + sample_x) * 3) as usize;
                if idx + 2 < orig_image.data.len() {
                    samples.push([
                        orig_image.data[idx],
                        orig_image.data[idx + 1],
                        orig_image.data[idx + 2],
                    ]);
                }
            }
        }

        // Calculate average RGB
        if !samples.is_empty() {
            let mut sum_r = 0.0;
            let mut sum_g = 0.0;
            let mut sum_b = 0.0;

            for sample in &samples {
                sum_r += sample[0];
                sum_g += sample[1];
                sum_b += sample[2];
            }

            let count = samples.len() as f32;
            self.base_r = sum_r / count;
            self.base_g = sum_g / count;
            self.base_b = sum_b / count;

            eprintln!(
                "[EYEDROPPER] Sampled at ({}, {}) - RGB: [{:.6}, {:.6}, {:.6}] from {} pixels",
                x,
                y,
                self.base_r,
                self.base_g,
                self.base_b,
                samples.len()
            );

            self.processing_needed = true;
        }
    }

    fn create_color_image(&self, result: &ProcessedImage) -> egui::ColorImage {
        let width = result.width as usize;
        let height = result.height as usize;
        let mut pixels = Vec::with_capacity(width * height);

        // Convert f32 linear RGB to sRGB u8 for display
        for pixel in result.data.chunks_exact(3) {
            let r = linear_to_srgb(pixel[0]);
            let g = linear_to_srgb(pixel[1]);
            let b = linear_to_srgb(pixel[2]);
            pixels.push(egui::Color32::from_rgb(r, g, b));
        }

        egui::ColorImage {
            size: [width, height],
            pixels,
        }
    }

    fn show_controls(&mut self, ui: &mut egui::Ui) {
        if self.loaded_image.is_none() {
            ui.label("Load an image to adjust parameters");
            return;
        }

        ui.collapsing("Base Estimation", |ui| {
            ui.label("Film base color (from brightest area):");

            // Eyedropper button
            let eyedropper_text = if self.eyedropper_mode {
                "🎨 Eyedropper Active (click image)"
            } else {
                "🎨 Pick Base Color from Image"
            };
            if ui.button(eyedropper_text).clicked() {
                self.eyedropper_mode = !self.eyedropper_mode;
            }

            ui.separator();

            if ui
                .add(egui::Slider::new(&mut self.base_r, 0.0..=1.0).text("Red"))
                .changed()
            {
                self.processing_needed = true;
            }
            if ui
                .add(egui::Slider::new(&mut self.base_g, 0.0..=1.0).text("Green"))
                .changed()
            {
                self.processing_needed = true;
            }
            if ui
                .add(egui::Slider::new(&mut self.base_b, 0.0..=1.0).text("Blue"))
                .changed()
            {
                self.processing_needed = true;
            }
        });

        ui.collapsing("Base Offsets", |ui| {
            ui.label("Additional color correction offsets:");
            if ui
                .add(egui::Slider::new(&mut self.base_offset_r, -0.2..=0.2).text("Red"))
                .changed()
            {
                self.processing_needed = true;
            }
            if ui
                .add(egui::Slider::new(&mut self.base_offset_g, -0.2..=0.2).text("Green"))
                .changed()
            {
                self.processing_needed = true;
            }
            if ui
                .add(egui::Slider::new(&mut self.base_offset_b, -0.2..=0.2).text("Blue"))
                .changed()
            {
                self.processing_needed = true;
            }
        });

        ui.collapsing("White Balance", |ui| {
            ui.label("Pick neutral gray/white area for color balance:");

            // White balance eyedropper button
            let wb_text = if self.white_balance_mode {
                "⚖️ White Balance Active (click image)"
            } else {
                "⚖️ Pick Neutral Area"
            };
            if ui.button(wb_text).clicked() {
                self.white_balance_mode = !self.white_balance_mode;
            }

            ui.label("(Pick an area that should appear white/gray)");
            ui.separator();

            ui.label("White balance multipliers:");
            if ui
                .add(egui::Slider::new(&mut self.white_balance_r, 0.1..=3.0).text("Red"))
                .changed()
            {
                self.processing_needed = true;
            }
            if ui
                .add(egui::Slider::new(&mut self.white_balance_g, 0.1..=3.0).text("Green"))
                .changed()
            {
                self.processing_needed = true;
            }
            if ui
                .add(egui::Slider::new(&mut self.white_balance_b, 0.1..=3.0).text("Blue"))
                .changed()
            {
                self.processing_needed = true;
            }

            if ui.button("Reset White Balance").clicked() {
                self.white_balance_r = 1.0;
                self.white_balance_g = 1.0;
                self.white_balance_b = 1.0;
                self.processing_needed = true;
            }
        });

        ui.collapsing("Exposure", |ui| {
            ui.label("Brightness adjustment:");
            if ui
                .add(
                    egui::Slider::new(&mut self.exposure_compensation, 0.1..=3.0)
                        .text("Exposure")
                        .logarithmic(true),
                )
                .changed()
            {
                self.processing_needed = true;
            }
            if ui.button("Reset to 1.0").clicked() {
                self.exposure_compensation = 1.0;
                self.processing_needed = true;
            }
        });

        ui.collapsing("Tone Curve", |ui| {
            if ui
                .checkbox(&mut self.skip_tone_curve, "Skip tone curve")
                .changed()
            {
                self.processing_needed = true;
            }
            if !self.skip_tone_curve {
                ui.label("Tone curve strength (0 = linear, 1 = max):");
                if ui
                    .add(
                        egui::Slider::new(&mut self.tone_curve_strength, 0.0..=1.0)
                            .text("Strength"),
                    )
                    .changed()
                {
                    self.processing_needed = true;
                }
            }
        });

        ui.collapsing("Color Matrix", |ui| {
            if ui
                .checkbox(&mut self.skip_color_matrix, "Skip color matrix")
                .changed()
            {
                self.processing_needed = true;
            }

            if !self.skip_color_matrix && self.show_color_matrix {
                ui.label("3x3 color correction matrix:");
                ui.separator();
                for row in 0..3 {
                    ui.horizontal(|ui| {
                        for col in 0..3 {
                            if ui
                                .add(
                                    egui::DragValue::new(&mut self.color_matrix[row][col])
                                        .speed(0.01)
                                        .range(-2.0..=2.0),
                                )
                                .changed()
                            {
                                self.processing_needed = true;
                            }
                        }
                    });
                }
                if ui.button("Reset to Identity").clicked() {
                    self.color_matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
                    self.processing_needed = true;
                }
            }
        });

        ui.separator();

        // Show both original and preview resolutions
        if let Some(ref loaded) = self.loaded_image {
            ui.label(format!("Original: {}x{}", loaded.width, loaded.height));
        }
        if let Some(ref preview) = self.preview_image {
            ui.label(format!(
                "Preview: {}x{} (for fast adjustment)",
                preview.width, preview.height
            ));
        }

        if let Some(ref path) = self.loaded_path {
            ui.label(format!(
                "File: {}",
                path.file_name().unwrap().to_string_lossy()
            ));
        }
    }
}

/// Downsample image for fast preview
/// Target: max dimension of 1024px (adjustable)
fn downsample_image(image: &DecodedImage, max_dimension: u32) -> DecodedImage {
    let (width, height) = (image.width, image.height);

    // Check if downsampling is needed
    if width <= max_dimension && height <= max_dimension {
        return image.clone();
    }

    // Calculate new dimensions maintaining aspect ratio
    let scale = if width > height {
        max_dimension as f32 / width as f32
    } else {
        max_dimension as f32 / height as f32
    };

    let new_width = (width as f32 * scale).round() as u32;
    let new_height = (height as f32 * scale).round() as u32;

    // Convert f32 linear RGB to image crate format
    let mut img_buffer = image::RgbImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 3) as usize;
            let r = (image.data[idx] * 65535.0).clamp(0.0, 65535.0) as u16;
            let g = (image.data[idx + 1] * 65535.0).clamp(0.0, 65535.0) as u16;
            let b = (image.data[idx + 2] * 65535.0).clamp(0.0, 65535.0) as u16;

            // Convert to u8 for intermediate processing
            img_buffer.put_pixel(
                x,
                y,
                image::Rgb([(r >> 8) as u8, (g >> 8) as u8, (b >> 8) as u8]),
            );
        }
    }

    // Resize using Triangle (bilinear) filter - good balance of quality and speed
    let resized = image::imageops::resize(
        &img_buffer,
        new_width,
        new_height,
        image::imageops::FilterType::Triangle,
    );

    // Convert back to f32 linear RGB
    let mut data = Vec::with_capacity((new_width * new_height * 3) as usize);
    for pixel in resized.pixels() {
        data.push(pixel[0] as f32 / 255.0);
        data.push(pixel[1] as f32 / 255.0);
        data.push(pixel[2] as f32 / 255.0);
    }

    DecodedImage {
        width: new_width,
        height: new_height,
        data,
        channels: 3,
        black_level: image.black_level,
        white_level: image.white_level,
        color_matrix: image.color_matrix,
    }
}

/// Convert linear RGB value to sRGB with gamma correction
fn linear_to_srgb(linear: f32) -> u8 {
    let linear = linear.clamp(0.0, 1.0);
    let srgb = if linear <= 0.0031308 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    };
    (srgb * 255.0).round().clamp(0.0, 255.0) as u8
}
