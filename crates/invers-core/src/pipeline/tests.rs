//! Tests for the processing pipeline
//!
//! Integration and unit tests for pipeline components.

use super::*;

// ========================================================================
// ProcessedImage Tests
// ========================================================================

#[test]
fn test_processed_image_creation() {
    let processed = ProcessedImage {
        width: 100,
        height: 200,
        data: vec![0.5; 60000], // 100 * 200 * 3
        channels: 3,
        export_as_grayscale: false,
    };

    assert_eq!(processed.width, 100);
    assert_eq!(processed.height, 200);
    assert_eq!(processed.data.len(), 60000);
    assert!(!processed.export_as_grayscale);
}
