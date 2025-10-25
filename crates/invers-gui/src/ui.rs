//! UI components and widgets
//!
//! Qt/QML UI components for the Positize GUI.
//!
//! In M2, this will contain:
//! - cxx-qt bridge definitions
//! - QML UI component implementations
//! - Signal/slot connections

use crate::app_state::AppState;

/// Main window controller
///
/// TODO: In M2, this will be a cxx-qt QObject
pub struct MainWindow {
    state: AppState,
}

impl MainWindow {
    pub fn new() -> Self {
        Self {
            state: AppState::default(),
        }
    }

    pub fn state(&self) -> &AppState {
        &self.state
    }

    pub fn state_mut(&mut self) -> &mut AppState {
        &mut self.state
    }
}

/// Import panel for file selection
pub struct ImportPanel;

impl ImportPanel {
    // TODO: Implement file picker
    // TODO: Implement drag-and-drop
}

/// Preset manager panel
pub struct PresetManager;

impl PresetManager {
    // TODO: Load/save presets
    // TODO: UI for selecting film/scan profiles
}

/// Export settings panel
pub struct ExportPanel;

impl ExportPanel {
    // TODO: Format selection
    // TODO: Colorspace selection
    // TODO: Output directory picker
}

/// Batch processing queue UI
pub struct BatchQueuePanel;

impl BatchQueuePanel {
    // TODO: Display batch items with status
    // TODO: Progress bars
    // TODO: Start/stop/clear controls
}
