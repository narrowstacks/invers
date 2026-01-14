// Colorspace conversion shaders (RGB↔HSL) and HSL adjustments.

struct DummyParams {
    pixel_count: u32,
    _padding: vec3<u32>,
}

struct HslAdjustParams {
    // 8 color ranges: R(0), O(1), Y(2), G(3), A(4), B(5), P(6), M(7)
    // Using vec4 pairs for 16-byte alignment
    hue_adj_0: vec4<f32>,  // R, O, Y, G
    hue_adj_1: vec4<f32>,  // A, B, P, M
    sat_adj_0: vec4<f32>,  // R, O, Y, G
    sat_adj_1: vec4<f32>,  // A, B, P, M
    lum_adj_0: vec4<f32>,  // R, O, Y, G
    lum_adj_1: vec4<f32>,  // A, B, P, M
    pixel_count: u32,
    _padding: vec3<u32>,
}

@group(0) @binding(0) var<storage, read_write> pixels: array<f32>;
@group(0) @binding(1) var<uniform> params: DummyParams;

// Workgroup size for all shaders
const WORKGROUP_SIZE: u32 = 256u;

// Calculate linear pixel index from 2D dispatch grid
// Supports images larger than 65535 workgroups by using 2D dispatch
fn get_pixel_index(id: vec3<u32>, num_workgroups: vec3<u32>) -> u32 {
    return id.y * num_workgroups.x * WORKGROUP_SIZE + id.x;
}

// Color range centers in degrees
const COLOR_CENTERS: array<f32, 8> = array<f32, 8>(
    0.0,    // Red
    30.0,   // Orange
    60.0,   // Yellow
    120.0,  // Green
    180.0,  // Aqua
    240.0,  // Blue
    285.0,  // Purple
    315.0   // Magenta
);

// Helper function for HSL to RGB conversion
// Optimized using select() to avoid branching
fn hue_to_rgb(p: f32, q: f32, t_in: f32) -> f32 {
    // Wrap t to [0, 1) range using select() instead of branches
    var t = t_in;
    t = select(t, t + 1.0, t < 0.0);
    t = select(t, t - 1.0, t > 1.0);

    // Calculate all possible results
    let rising = p + (q - p) * 6.0 * t;           // t < 1/6: rising edge
    let peak = q;                                  // 1/6 <= t < 1/2: peak
    let falling = p + (q - p) * (2.0 / 3.0 - t) * 6.0; // 1/2 <= t < 2/3: falling edge
    let base = p;                                  // t >= 2/3: base

    // Use nested select for branchless selection
    // Order matters: check conditions from most restrictive to least
    let result = select(
        select(
            select(base, falling, t < 2.0 / 3.0),
            peak,
            t < 0.5
        ),
        rising,
        t < 1.0 / 6.0
    );

    return result;
}

// RGB to HSL conversion - optimized with select() to reduce branching
fn rgb_to_hsl_impl(r: f32, g: f32, b: f32) -> vec3<f32> {
    let max_val = max(max(r, g), b);
    let min_val = min(min(r, g), b);
    let delta = max_val - min_val;

    // Lightness
    let l = (max_val + min_val) / 2.0;

    // Saturation - compute both branches and select
    let s_low = delta / max(max_val + min_val, 0.0001);
    let s_high = delta / max(2.0 - max_val - min_val, 0.0001);
    let s = select(s_high, s_low, l < 0.5);

    // Hue calculation - compute all branches
    // When max is R: h = (g - b) / delta, add 6 if g < b
    let h_r_base = (g - b) / max(delta, 0.0001);
    let h_r = select(h_r_base, h_r_base + 6.0, g < b);
    // When max is G: h = (b - r) / delta + 2
    let h_g = (b - r) / max(delta, 0.0001) + 2.0;
    // When max is B: h = (r - g) / delta + 4
    let h_b = (r - g) / max(delta, 0.0001) + 4.0;

    // Select based on which component is max (use epsilon comparison for float equality)
    let eps = 0.0001;
    let is_r_max = abs(max_val - r) < eps;
    let is_g_max = abs(max_val - g) < eps;

    // Chained select: check R first, then G, default to B
    let h_raw = select(select(h_b, h_g, is_g_max), h_r, is_r_max);
    let h = h_raw * 60.0; // Convert to degrees

    // Zero out h and s when delta is negligible (grayscale)
    let has_color = delta > 0.0001;
    return vec3<f32>(
        select(0.0, h, has_color),
        select(0.0, s, has_color),
        l
    );
}

// HSL to RGB conversion - optimized with select()
fn hsl_to_rgb_impl(h: f32, s: f32, l: f32) -> vec3<f32> {
    // Compute both q branches and select
    let q_low = l * (1.0 + s);
    let q_high = l + s - l * s;
    let q = select(q_high, q_low, l < 0.5);
    let p = 2.0 * l - q;

    let h_norm = h / 360.0;

    let r = hue_to_rgb(p, q, h_norm + 1.0 / 3.0);
    let g = hue_to_rgb(p, q, h_norm);
    let b = hue_to_rgb(p, q, h_norm - 1.0 / 3.0);

    // Return grayscale if saturation is negligible
    return select(vec3<f32>(r, g, b), vec3<f32>(l, l, l), s < 0.0001);
}

// Convert RGB to HSL (in-place)
@compute @workgroup_size(256)
fn rgb_to_hsl(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let pixel_count = params.pixel_count;
    let pixel_idx = get_pixel_index(id, num_workgroups);

    if (pixel_idx >= pixel_count) {
        return;
    }

    let idx = pixel_idx * 3u;
    let r = pixels[idx];
    let g = pixels[idx + 1u];
    let b = pixels[idx + 2u];

    let hsl = rgb_to_hsl_impl(r, g, b);

    // Store H as 0-360, S and L as 0-1
    pixels[idx] = hsl.x;
    pixels[idx + 1u] = hsl.y;
    pixels[idx + 2u] = hsl.z;
}

// Convert HSL to RGB (in-place)
@compute @workgroup_size(256)
fn hsl_to_rgb(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let pixel_count = params.pixel_count;
    let pixel_idx = get_pixel_index(id, num_workgroups);

    if (pixel_idx >= pixel_count) {
        return;
    }

    let idx = pixel_idx * 3u;
    let h = pixels[idx];
    let s = pixels[idx + 1u];
    let l = pixels[idx + 2u];

    let rgb = hsl_to_rgb_impl(h, s, l);

    pixels[idx] = clamp(rgb.x, 0.0, 1.0);
    pixels[idx + 1u] = clamp(rgb.y, 0.0, 1.0);
    pixels[idx + 2u] = clamp(rgb.z, 0.0, 1.0);
}

// Separate binding for HSL adjust params
@group(0) @binding(1) var<uniform> hsl_params: HslAdjustParams;

// Helper function to get hue adjustment by index
fn get_hue_adj(idx: u32) -> f32 {
    if (idx < 4u) {
        return hsl_params.hue_adj_0[idx];
    } else {
        return hsl_params.hue_adj_1[idx - 4u];
    }
}

// Helper function to get saturation adjustment by index
fn get_sat_adj(idx: u32) -> f32 {
    if (idx < 4u) {
        return hsl_params.sat_adj_0[idx];
    } else {
        return hsl_params.sat_adj_1[idx - 4u];
    }
}

// Helper function to get luminance adjustment by index
fn get_lum_adj(idx: u32) -> f32 {
    if (idx < 4u) {
        return hsl_params.lum_adj_0[idx];
    } else {
        return hsl_params.lum_adj_1[idx - 4u];
    }
}

// Calculate angular distance between two hue values (branchless version)
fn hue_distance(h1: f32, h2: f32) -> f32 {
    let diff = abs(h1 - h2);
    return select(diff, 360.0 - diff, diff > 180.0);
}

// Sector boundaries: midpoints between adjacent color centers
// R(0), O(30), Y(60), G(120), A(180), B(240), P(285), M(315)
// Boundaries: 15, 45, 90, 150, 210, 262.5, 300, 337.5
const SECTOR_BOUNDS: array<f32, 8> = array<f32, 8>(
    15.0,    // R-O boundary (midpoint of 0-30)
    45.0,    // O-Y boundary (midpoint of 30-60)
    90.0,    // Y-G boundary (midpoint of 60-120)
    150.0,   // G-A boundary (midpoint of 120-180)
    210.0,   // A-B boundary (midpoint of 180-240)
    262.5,   // B-P boundary (midpoint of 240-285)
    300.0,   // P-M boundary (midpoint of 285-315)
    337.5    // M-R boundary (midpoint of 315-360)
);

// Find primary and secondary color ranges and blend factor
// O(1) direct calculation based on hue angle sectors
fn get_color_weights(hue: f32) -> vec3<f32> {
    // Normalize hue to [0, 360)
    var h = hue;
    h = select(h, h + 360.0, h < 0.0);
    h = select(h, h - 360.0, h >= 360.0);

    // Determine primary sector using binary-style search with select()
    // This avoids loops entirely by using arithmetic comparisons
    //
    // Sector mapping:
    // [337.5, 360) or [0, 15) -> 0 (Red)
    // [15, 45)                -> 1 (Orange)
    // [45, 90)                -> 2 (Yellow)
    // [90, 150)               -> 3 (Green)
    // [150, 210)              -> 4 (Aqua)
    // [210, 262.5)            -> 5 (Blue)
    // [262.5, 300)            -> 6 (Purple)
    // [300, 337.5)            -> 7 (Magenta)

    // Count how many boundaries the hue exceeds to determine sector
    // Red wraps around, so we handle it specially
    var sector: u32 = 0u;
    sector = select(sector, 1u, h >= SECTOR_BOUNDS[0]);
    sector = select(sector, 2u, h >= SECTOR_BOUNDS[1]);
    sector = select(sector, 3u, h >= SECTOR_BOUNDS[2]);
    sector = select(sector, 4u, h >= SECTOR_BOUNDS[3]);
    sector = select(sector, 5u, h >= SECTOR_BOUNDS[4]);
    sector = select(sector, 6u, h >= SECTOR_BOUNDS[5]);
    sector = select(sector, 7u, h >= SECTOR_BOUNDS[6]);
    sector = select(sector, 0u, h >= SECTOR_BOUNDS[7]); // Wrap to Red

    let primary_idx = sector;
    let primary_center = COLOR_CENTERS[primary_idx];

    // Calculate signed distance from primary center (handling wrap-around)
    var signed_dist = h - primary_center;
    // Handle wrap-around for Red sector
    if (primary_idx == 0u) {
        signed_dist = select(signed_dist, signed_dist - 360.0, h > 180.0);
    }

    // Secondary is the adjacent sector in the direction of the hue
    // If hue is CW from center (positive), secondary is next sector
    // If hue is CCW from center (negative), secondary is previous sector
    let secondary_idx = select(
        (primary_idx + 7u) % 8u,  // Previous sector (CCW)
        (primary_idx + 1u) % 8u,  // Next sector (CW)
        signed_dist >= 0.0
    );

    // Calculate blend factor based on position between the two centers
    let secondary_center = COLOR_CENTERS[secondary_idx];
    let dist_to_primary = hue_distance(h, primary_center);
    let dist_to_secondary = hue_distance(h, secondary_center);
    let total_dist = dist_to_primary + dist_to_secondary;

    // Blend is the proportion of influence from secondary color
    // When at primary center: blend = 0
    // At midpoint: blend = 0.5
    let blend = select(0.0, clamp(dist_to_primary / total_dist, 0.0, 0.5), total_dist > 0.001);

    return vec3<f32>(f32(primary_idx), f32(secondary_idx), blend);
}

// Apply 8-color HSL adjustments (Camera Raw style)
@compute @workgroup_size(256)
fn apply_hsl_adjustments(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let pixel_count = hsl_params.pixel_count;
    let pixel_idx = get_pixel_index(id, num_workgroups);

    if (pixel_idx >= pixel_count) {
        return;
    }

    let idx = pixel_idx * 3u;
    let r = pixels[idx];
    let g = pixels[idx + 1u];
    let b = pixels[idx + 2u];

    // Convert to HSL
    let hsl = rgb_to_hsl_impl(r, g, b);
    var h = hsl.x;
    var s = hsl.y;
    var l = hsl.z;

    // Get color range weights
    let weights = get_color_weights(h);
    let primary = u32(weights.x);
    let secondary = u32(weights.y);
    let blend = weights.z;

    // Calculate weighted adjustments
    let hue_adj = get_hue_adj(primary) * (1.0 - blend) +
                  get_hue_adj(secondary) * blend;
    let sat_adj = get_sat_adj(primary) * (1.0 - blend) +
                  get_sat_adj(secondary) * blend;
    let lum_adj = get_lum_adj(primary) * (1.0 - blend) +
                  get_lum_adj(secondary) * blend;

    // Apply adjustments
    // Hue: ±100 maps to ±30 degrees
    h = h + hue_adj * 0.3;
    h = select(h, h + 360.0, h < 0.0);
    h = select(h, h - 360.0, h >= 360.0);

    // Saturation: ±100 as multiplicative factor
    s = s * (1.0 + sat_adj / 100.0);
    s = clamp(s, 0.0, 1.0);

    // Luminance: ±100 as additive on 0-1 scale (scaled by /200)
    l = l + lum_adj / 200.0;
    l = clamp(l, 0.0, 1.0);

    // Convert back to RGB
    let rgb = hsl_to_rgb_impl(h, s, l);

    pixels[idx] = clamp(rgb.x, 0.0, 1.0);
    pixels[idx + 1u] = clamp(rgb.y, 0.0, 1.0);
    pixels[idx + 2u] = clamp(rgb.z, 0.0, 1.0);
}
