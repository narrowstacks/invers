// Inversion shaders for film negative to positive conversion.
// Four modes: linear, logarithmic, divide-blend, mask-aware.

// Compile-time constants for efficient log10/pow10 operations.
// Using natural log: log10(x) = ln(x) / ln(10) = ln(x) * (1/ln(10))
// Using exp for pow: 10^x = e^(x * ln(10))
const LOG10_RECIP: f32 = 0.4342944819;  // 1/ln(10), multiply by this instead of dividing by ln(10)
const LN10: f32 = 2.302585093;          // ln(10), for converting pow(10, x) to exp(x * LN10)

struct InversionParams {
    base_r: f32,
    base_g: f32,
    base_b: f32,
    green_floor: f32,
    blue_floor: f32,
    bw_headroom: f32,  // Headroom for B&W mode (e.g., 0.05 = 5%)
    pixel_count: u32,
    _padding: u32,
}

@group(0) @binding(0) var<storage, read_write> pixels: array<f32>;
@group(0) @binding(1) var<uniform> params: InversionParams;

// Workgroup size for all shaders
const WORKGROUP_SIZE: u32 = 256u;

// Calculate linear pixel index from 2D dispatch grid
// Supports images larger than 65535 workgroups by using 2D dispatch
fn get_pixel_index(id: vec3<u32>, num_workgroups: vec3<u32>) -> u32 {
    return id.y * num_workgroups.x * WORKGROUP_SIZE + id.x;
}

// Linear inversion: positive = (base - negative) / base
// Note: No clamping here - matches CPU behavior. Final clamping happens in utility shader.
@compute @workgroup_size(256)
fn invert_linear(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let pixel_idx = get_pixel_index(id, num_workgroups);
    if (pixel_idx >= params.pixel_count) {
        return;
    }

    let idx = pixel_idx * 3u;
    let r = pixels[idx];
    let g = pixels[idx + 1u];
    let b = pixels[idx + 2u];

    // Linear inversion: (base - negative) / base
    // Allow values outside [0,1] to preserve dynamic range for subsequent operations
    let inv_r = (params.base_r - r) / params.base_r;
    let inv_g = (params.base_g - g) / params.base_g;
    let inv_b = (params.base_b - b) / params.base_b;

    pixels[idx] = inv_r;
    pixels[idx + 1u] = inv_g;
    pixels[idx + 2u] = inv_b;
}

// Logarithmic (density-based) inversion: positive = 10^(log10(base) - log10(negative))
@compute @workgroup_size(256)
fn invert_log(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let pixel_idx = get_pixel_index(id, num_workgroups);
    if (pixel_idx >= params.pixel_count) {
        return;
    }

    let idx = pixel_idx * 3u;
    let r = max(pixels[idx], 0.0001);
    let g = max(pixels[idx + 1u], 0.0001);
    let b = max(pixels[idx + 2u], 0.0001);

    // Pre-compute log10 of base using efficient multiplication by reciprocal
    // log10(x) = ln(x) * (1/ln(10)) = ln(x) * LOG10_RECIP
    let log_base_r = log(max(params.base_r, 0.0001)) * LOG10_RECIP;
    let log_base_g = log(max(params.base_g, 0.0001)) * LOG10_RECIP;
    let log_base_b = log(max(params.base_b, 0.0001)) * LOG10_RECIP;

    // Log inversion: 10^(log_base - log_pixel)
    // Using log10(x) = ln(x) * LOG10_RECIP
    let log_r = log(r) * LOG10_RECIP;
    let log_g = log(g) * LOG10_RECIP;
    let log_b = log(b) * LOG10_RECIP;

    // 10^x = e^(x * ln(10)) = exp(x * LN10)
    let inv_r = exp((log_base_r - log_r) * LN10);
    let inv_g = exp((log_base_g - log_g) * LN10);
    let inv_b = exp((log_base_b - log_b) * LN10);

    pixels[idx] = clamp(inv_r, 0.0, 10.0);
    pixels[idx + 1u] = clamp(inv_g, 0.0, 10.0);
    pixels[idx + 2u] = clamp(inv_b, 0.0, 10.0);
}

// Divide-blend inversion: mimics Photoshop's divide blend workflow
// 1. Divide by base: divided = pixel / base
// 2. Apply gamma: gamma_result = divided^(1/2.2)
// 3. Invert: result = 1.0 - gamma_result
@compute @workgroup_size(256)
fn invert_divide(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let pixel_idx = get_pixel_index(id, num_workgroups);
    if (pixel_idx >= params.pixel_count) {
        return;
    }

    let idx = pixel_idx * 3u;
    let r = pixels[idx];
    let g = pixels[idx + 1u];
    let b = pixels[idx + 2u];

    let gamma = 1.0 / 2.2;

    // Divide by base and clamp
    let div_r = clamp(r / max(params.base_r, 0.0001), 0.0, 10.0);
    let div_g = clamp(g / max(params.base_g, 0.0001), 0.0, 10.0);
    let div_b = clamp(b / max(params.base_b, 0.0001), 0.0, 10.0);

    // Apply gamma
    let gamma_r = pow(div_r, gamma);
    let gamma_g = pow(div_g, gamma);
    let gamma_b = pow(div_b, gamma);

    // Invert
    pixels[idx] = clamp(1.0 - gamma_r, 0.0, 1.0);
    pixels[idx + 1u] = clamp(1.0 - gamma_g, 0.0, 1.0);
    pixels[idx + 2u] = clamp(1.0 - gamma_b, 0.0, 1.0);
}

// MaskAware inversion: handles orange mask in color negative film
// 1. Standard inversion: 1.0 - (pixel / base)
// 2. Shadow floor correction for green and blue channels
@compute @workgroup_size(256)
fn invert_mask_aware(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let pixel_idx = get_pixel_index(id, num_workgroups);
    if (pixel_idx >= params.pixel_count) {
        return;
    }

    let idx = pixel_idx * 3u;
    let r = pixels[idx];
    let g = pixels[idx + 1u];
    let b = pixels[idx + 2u];

    // Standard inversion: 1.0 - (pixel / base)
    let inv_r = 1.0 - (r / max(params.base_r, 0.0001));
    let raw_inv_g = 1.0 - (g / max(params.base_g, 0.0001));
    let raw_inv_b = 1.0 - (b / max(params.base_b, 0.0001));

    // Apply shadow floor correction for green and blue
    // Removes blue cast from naive mask inversion
    // Using select() for non-divergent execution on GPU
    let corrected_g = (raw_inv_g - params.green_floor) / (1.0 - params.green_floor);
    let corrected_b = (raw_inv_b - params.blue_floor) / (1.0 - params.blue_floor);
    let inv_g = select(raw_inv_g, corrected_g, params.green_floor > 0.0);
    let inv_b = select(raw_inv_b, corrected_b, params.blue_floor > 0.0);

    pixels[idx] = clamp(inv_r, 0.0, 1.0);
    pixels[idx + 1u] = clamp(inv_g, 0.0, 1.0);
    pixels[idx + 2u] = clamp(inv_b, 0.0, 1.0);
}

// Black & White inversion: simple inversion with black point near film base
// Optimized for grayscale/monochrome images
// 1. Simple inversion: base - pixel
// 2. Scale so that (base - headroom) maps to 0 (black)
// 3. This preserves shadow detail by not clipping the film base completely
@compute @workgroup_size(256)
fn invert_bw(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let pixel_idx = get_pixel_index(id, num_workgroups);
    if (pixel_idx >= params.pixel_count) {
        return;
    }

    let idx = pixel_idx * 3u;
    let r = pixels[idx];
    let g = pixels[idx + 1u];
    let b = pixels[idx + 2u];

    // Use average base for B&W (all channels should be similar)
    let base = (params.base_r + params.base_g + params.base_b) / 3.0;

    // Effective black point: base minus headroom
    // headroom is a fraction of base (e.g., 0.05 means 5% headroom)
    let black_point = base * (1.0 - params.bw_headroom);

    // Simple inversion: base - pixel, then scale
    // When pixel = base, result = 0 (but we want some headroom)
    // When pixel = 0, result = base (white)
    // Scale factor: 1.0 / black_point to map black_point -> 1.0
    let scale = 1.0 / max(black_point, 0.0001);

    let inv_r = (base - r) * scale;
    let inv_g = (base - g) * scale;
    let inv_b = (base - b) * scale;

    pixels[idx] = clamp(inv_r, 0.0, 1.0);
    pixels[idx + 1u] = clamp(inv_g, 0.0, 1.0);
    pixels[idx + 2u] = clamp(inv_b, 0.0, 1.0);
}
