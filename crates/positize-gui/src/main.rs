//! Positize GUI Application
//!
//! Qt-based GUI for film negative to positive conversion using cxx-qt.
//!
//! NOTE: Full cxx-qt integration is planned for M2. This is a scaffold showing
//! the intended architecture.

mod app_state;
mod preview;
mod roi_tool;
mod ui;
mod viewer;

use positize_core::{BaseEstimation, ConvertOptions, FilmPreset, ScanProfile};

fn main() {
    println!("Positize GUI - M2 Implementation Pending");
    println!("This will be a cxx-qt based application with:");
    println!("  - Image import and preview");
    println!("  - Interactive ROI selection for base estimation");
    println!("  - Film preset and scan profile management");
    println!("  - Real-time preview with adjustments");
    println!("  - Batch processing queue");
    println!("  - Export to TIFF16/DNG with progress tracking");

    // Placeholder to show dependencies work
    let _ = ConvertOptions {
        input_paths: vec![],
        output_dir: std::path::PathBuf::from("."),
        output_format: positize_core::models::OutputFormat::Tiff16,
        working_colorspace: "linear-rec2020".to_string(),
        bit_depth_policy: positize_core::models::BitDepthPolicy::MatchInput,
        film_preset: None,
        scan_profile: None,
        base_estimation: None,
        num_threads: None,
        skip_tone_curve: false,
        skip_color_matrix: false,
        exposure_compensation: 1.0,
        debug: false,
    };

    // TODO: In M2, initialize QApplication and run event loop
    // let app = QApplication::new();
    // let main_window = MainWindow::new();
    // main_window.show();
    // app.exec();
}
