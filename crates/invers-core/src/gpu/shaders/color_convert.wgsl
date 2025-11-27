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
fn hue_to_rgb(p: f32, q: f32, t_in: f32) -> f32 {
    var t = t_in;
    if (t < 0.0) { t = t + 1.0; }
    if (t > 1.0) { t = t - 1.0; }

    if (t < 1.0 / 6.0) {
        return p + (q - p) * 6.0 * t;
    }
    if (t < 0.5) {
        return q;
    }
    if (t < 2.0 / 3.0) {
        return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
    }
    return p;
}

// RGB to HSL conversion
fn rgb_to_hsl_impl(r: f32, g: f32, b: f32) -> vec3<f32> {
    let max_val = max(max(r, g), b);
    let min_val = min(min(r, g), b);
    let delta = max_val - min_val;

    // Lightness
    let l = (max_val + min_val) / 2.0;

    var h: f32 = 0.0;
    var s: f32 = 0.0;

    if (delta > 0.0001) {
        // Saturation
        if (l < 0.5) {
            s = delta / (max_val + min_val);
        } else {
            s = delta / (2.0 - max_val - min_val);
        }

        // Hue
        if (max_val == r) {
            h = (g - b) / delta;
            if (g < b) {
                h = h + 6.0;
            }
        } else if (max_val == g) {
            h = (b - r) / delta + 2.0;
        } else {
            h = (r - g) / delta + 4.0;
        }

        h = h * 60.0; // Convert to degrees
    }

    return vec3<f32>(h, s, l);
}

// HSL to RGB conversion
fn hsl_to_rgb_impl(h: f32, s: f32, l: f32) -> vec3<f32> {
    if (s < 0.0001) {
        return vec3<f32>(l, l, l);
    }

    var q: f32;
    if (l < 0.5) {
        q = l * (1.0 + s);
    } else {
        q = l + s - l * s;
    }
    let p = 2.0 * l - q;

    let h_norm = h / 360.0;

    let r = hue_to_rgb(p, q, h_norm + 1.0 / 3.0);
    let g = hue_to_rgb(p, q, h_norm);
    let b = hue_to_rgb(p, q, h_norm - 1.0 / 3.0);

    return vec3<f32>(r, g, b);
}

// Convert RGB to HSL (in-place)
@compute @workgroup_size(256)
fn rgb_to_hsl(@builtin(global_invocation_id) id: vec3<u32>) {
    let pixel_count = params.pixel_count;
    let pixel_idx = id.x;

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
fn hsl_to_rgb(@builtin(global_invocation_id) id: vec3<u32>) {
    let pixel_count = params.pixel_count;
    let pixel_idx = id.x;

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

// Calculate angular distance between two hue values
fn hue_distance(h1: f32, h2: f32) -> f32 {
    var diff = abs(h1 - h2);
    if (diff > 180.0) {
        diff = 360.0 - diff;
    }
    return diff;
}

// Find primary and secondary color ranges and blend factor
fn get_color_weights(hue: f32) -> vec3<f32> {
    var primary_idx: u32 = 0u;
    var min_dist: f32 = 360.0;

    // Find nearest color center
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let dist = hue_distance(hue, COLOR_CENTERS[i]);
        if (dist < min_dist) {
            min_dist = dist;
            primary_idx = i;
        }
    }

    // Find second nearest
    var secondary_idx: u32 = (primary_idx + 1u) % 8u;
    var second_min_dist: f32 = 360.0;

    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        if (i != primary_idx) {
            let dist = hue_distance(hue, COLOR_CENTERS[i]);
            if (dist < second_min_dist) {
                second_min_dist = dist;
                secondary_idx = i;
            }
        }
    }

    // Calculate blend factor
    let total_dist = min_dist + second_min_dist;
    var blend: f32 = 0.0;
    if (total_dist > 0.001) {
        blend = clamp(min_dist / total_dist, 0.0, 0.5);
    }

    return vec3<f32>(f32(primary_idx), f32(secondary_idx), blend);
}

// Apply 8-color HSL adjustments (Camera Raw style)
@compute @workgroup_size(256)
fn apply_hsl_adjustments(@builtin(global_invocation_id) id: vec3<u32>) {
    let pixel_count = hsl_params.pixel_count;
    let pixel_idx = id.x;

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
    if (h < 0.0) { h = h + 360.0; }
    if (h >= 360.0) { h = h - 360.0; }

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
