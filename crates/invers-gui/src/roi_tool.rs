//! ROI (Region of Interest) selection tool
//!
//! Interactive tool for selecting a rectangular region for film base estimation.

/// ROI selection tool
pub struct RoiTool {
    /// Whether the tool is active
    active: bool,

    /// Selection start point
    start: Option<(u32, u32)>,

    /// Current selection rectangle (x, y, width, height)
    selection: Option<(u32, u32, u32, u32)>,
}

impl RoiTool {
    pub fn new() -> Self {
        Self {
            active: false,
            start: None,
            selection: None,
        }
    }

    /// Activate the ROI tool
    pub fn activate(&mut self) {
        self.active = true;
        self.clear_selection();
    }

    /// Deactivate the ROI tool
    pub fn deactivate(&mut self) {
        self.active = false;
    }

    /// Start a new selection
    pub fn start_selection(&mut self, x: u32, y: u32) {
        if self.active {
            self.start = Some((x, y));
            self.selection = Some((x, y, 0, 0));
        }
    }

    /// Update selection as mouse moves
    pub fn update_selection(&mut self, x: u32, y: u32) {
        if let Some((start_x, start_y)) = self.start {
            let width = x.abs_diff(start_x);
            let height = y.abs_diff(start_y);
            let min_x = start_x.min(x);
            let min_y = start_y.min(y);
            self.selection = Some((min_x, min_y, width, height));
        }
    }

    /// Finish selection
    pub fn finish_selection(&mut self) {
        self.start = None;
    }

    /// Get the current selection
    pub fn selection(&self) -> Option<(u32, u32, u32, u32)> {
        self.selection
    }

    /// Clear the selection
    pub fn clear_selection(&mut self) {
        self.start = None;
        self.selection = None;
    }

    /// Check if tool is active
    pub fn is_active(&self) -> bool {
        self.active
    }
}

/// ROI statistics display
pub struct RoiStats {
    /// Median RGB values in ROI
    pub medians: Option<[f32; 3]>,

    /// Standard deviation per channel
    pub std_devs: Option<[f32; 3]>,

    /// Number of pixels in ROI
    pub pixel_count: Option<u32>,
}

impl RoiStats {
    pub fn new() -> Self {
        Self {
            medians: None,
            std_devs: None,
            pixel_count: None,
        }
    }

    // TODO: Compute stats from image data and ROI
    // TODO: Display numeric readout in UI
}
