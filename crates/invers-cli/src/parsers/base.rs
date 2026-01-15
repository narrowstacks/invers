//! Base parsing functions for RGB values and ROI coordinates.

/// Parse base RGB values in format "R,G,B"
///
/// # Arguments
/// * `base_str` - A string in format "R,G,B" with values 0.0-1.0
///
/// # Returns
/// An array of [R, G, B] as f32 values
pub fn parse_base_rgb(base_str: &str) -> Result<[f32; 3], String> {
    let parts: Vec<&str> = base_str.split(',').collect();
    if parts.len() != 3 {
        return Err(format!(
            "Base must be in format R,G,B (e.g., 0.48,0.50,0.30), got: {}",
            base_str
        ));
    }

    let r = parts[0]
        .trim()
        .parse::<f32>()
        .map_err(|_| format!("Invalid red value: {}", parts[0]))?;
    let g = parts[1]
        .trim()
        .parse::<f32>()
        .map_err(|_| format!("Invalid green value: {}", parts[1]))?;
    let b = parts[2]
        .trim()
        .parse::<f32>()
        .map_err(|_| format!("Invalid blue value: {}", parts[2]))?;

    // Validate range
    for (val, name) in [(r, "Red"), (g, "Green"), (b, "Blue")] {
        if val <= 0.0 || val > 1.0 {
            return Err(format!(
                "{} value {} must be in range (0.0, 1.0]",
                name, val
            ));
        }
    }

    Ok([r, g, b])
}

/// Parse ROI string in format "x,y,width,height"
///
/// # Arguments
/// * `roi_str` - A string in format "x,y,width,height"
///
/// # Returns
/// A tuple of (x, y, width, height) as u32 values
pub fn parse_roi(roi_str: &str) -> Result<(u32, u32, u32, u32), String> {
    let parts: Vec<&str> = roi_str.split(',').collect();
    if parts.len() != 4 {
        return Err(format!(
            "ROI must be in format x,y,width,height, got: {}",
            roi_str
        ));
    }

    let x = parts[0]
        .trim()
        .parse::<u32>()
        .map_err(|_| format!("Invalid x coordinate: {}", parts[0]))?;
    let y = parts[1]
        .trim()
        .parse::<u32>()
        .map_err(|_| format!("Invalid y coordinate: {}", parts[1]))?;
    let width = parts[2]
        .trim()
        .parse::<u32>()
        .map_err(|_| format!("Invalid width: {}", parts[2]))?;
    let height = parts[3]
        .trim()
        .parse::<u32>()
        .map_err(|_| format!("Invalid height: {}", parts[3]))?;

    Ok((x, y, width, height))
}
