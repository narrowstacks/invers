//! Main application component
//!
//! The root component that assembles all UI pieces together.

use crate::components::{ControlsPanel, ExportSection, FileUpload, PreviewCanvas, StatusBar};
use crate::state::AppState;
use sycamore::prelude::*;

/// Main application component
#[component]
pub fn App<G: Html>() -> View<G> {
    // Create application state
    let state = AppState::new();

    // Check if we have an image loaded
    let has_image = {
        let state = state.clone();
        move || state.original_image.get().is_some()
    };

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
                    (if has_image() {
                        view! {
                            PreviewCanvas(state=&state)
                        }
                    } else {
                        view! {
                            FileUpload(state=&state)
                        }
                    })
                }

                div(class="controls-panel") {
                    ControlsPanel(state=&state)
                    ExportSection(state=&state)
                }
            }

            StatusBar(state=&state)
        }
    }
}
