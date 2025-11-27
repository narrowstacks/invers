//! Invers Web - Browser-based Film Negative Converter
//!
//! This crate provides a WebAssembly-based GUI for converting film negatives
//! to positives. All processing happens locally in the browser.

mod embedded_presets;
mod processing;
mod state;

use sycamore::prelude::*;
use wasm_bindgen::prelude::*;

/// Initialize the web application
#[wasm_bindgen(start)]
pub fn main() {
    // Initialize panic hook for better error messages
    console_error_panic_hook::set_once();

    // Initialize logging
    console_log::init_with_level(log::Level::Info).expect("Failed to initialize logger");

    log::info!("Invers Web starting...");

    // Mount the Sycamore application
    sycamore::render(App);

    log::info!("Invers Web initialized");
}

/// Main application component
#[component]
fn App() -> View {
    let state = state::AppState::new();
    let has_image = create_memo(move || state.has_image());

    view! {
        div(class="app") {
            header(class="app-header") {
                div {
                    h1 { "Invers" }
                    span(class="subtitle") { "Film Negative Converter" }
                }
                span(style="color: var(--text-secondary); font-size: 0.875rem") {
                    "All processing runs locally in your browser"
                }
            }

            main(class="main-content") {
                div(class="preview-panel") {
                    (if has_image.get() {
                        view! { p { "Image loaded - preview coming soon" } }
                    } else {
                        view! {
                            div(class="drop-zone") {
                                div(class="drop-zone-icon") { "📁" }
                                p(class="drop-zone-text") {
                                    strong { "Drop image here" }
                                    " or click to browse"
                                }
                                p(class="drop-zone-text") {
                                    "Supports TIFF (8/16/32-bit) and PNG (8/16-bit)"
                                }
                            }
                        }
                    })
                }

                div(class="controls-panel") {
                    div(class="control-section") {
                        div(class="control-section-header") {
                            span(class="control-section-title") { "Status" }
                        }
                        div(class="control-section-content") {
                            p { "Invers Web is loading..." }
                            p { "Full UI components coming soon." }
                        }
                    }
                }
            }

            div(class="status-bar") {
                span(class="status-message") {
                    "Ready"
                }
                span(style="color: var(--text-secondary)") {
                    "Invers Web v0.1.0"
                }
            }
        }
    }
}
