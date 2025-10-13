# Positize GUI

Qt-based graphical interface for Positize using cxx-qt.

## Status

**M2 Implementation Pending** - This is a scaffold showing the intended architecture.

## Planned Architecture

### Rust Components

- `app_state.rs` - Central application state management
- `ui.rs` - Main window and panel controllers
- `viewer.rs` - Image viewer with zoom/pan
- `roi_tool.rs` - ROI selection tool
- `preview.rs` - Real-time preview rendering

### QML Components

- `qml/main.qml` - Main window layout
- `qml/components/ImageViewer.qml` - Image display widget
- `qml/components/RoiOverlay.qml` - ROI selection overlay

## M2 Implementation Steps

1. Add cxx-qt dependencies to `Cargo.toml`
2. Create cxx-qt bridge in `src/ui.rs` using `#[cxx_qt::bridge]` macro
3. Implement QObject wrappers for Rust state
4. Connect QML UI to Rust backend via signals/slots
5. Integrate with positize-core processing pipeline
6. Implement preview renderer with background updates
7. Add file dialog integration
8. Implement batch processing UI with progress tracking

## Features

### Import Panel
- File picker for single/multiple images
- Drag-and-drop support
- Recent files list

### Viewer
- Zoom/pan with mouse wheel and drag
- Histogram overlay (per-channel and RGB)
- Pixel probe showing RGB values at cursor
- Real-time preview at reduced resolution

### ROI Tool
- Interactive rectangular selection
- Visual overlay with semi-transparent mask
- Numeric readout of median values and statistics
- Draggable corner handles for fine-tuning

### Preset Manager
- Load/save film presets
- Choose scan profiles
- Create custom presets
- Preview preset effects

### Batch Queue
- Add multiple files to queue
- Apply same settings to all
- Progress bars for each item
- Parallel processing with configurable threads

### Export Settings
- Format selection (TIFF16, Linear DNG)
- Colorspace selection
- Output directory picker
- Bit depth policy

## Dependencies (M2)

```toml
cxx-qt = "0.7"
cxx-qt-lib = "0.7"
cxx = "1.0"
qml = "0.2"
```

## Build Requirements (M2)

- Qt 6.x development libraries
- CMake for cxx-qt build
- C++ compiler
