// Inversion shaders for film negative to positive conversion.
// Four modes: linear, logarithmic, divide-blend, mask-aware.

struct InversionParams {
    base_r: f32,
    base_g: f32,
    base_b: f32,
    green_floor: f32,
    blue_floor: f32,
    pixel_count: u32,
    _padding: vec2<u32>,
}

@group(0) @binding(0) var<storage, read_write> pixels: array<f32>;
@group(0) @binding(1) var<uniform> params: InversionParams;

// Linear inversion: positive = (base - negative) / base
// Note: No clamping here - matches CPU behavior. Final clamping happens in utility shader.
@compute @workgroup_size(256)
fn invert_linear(@builtin(global_invocation_id) id: vec3<u32>) {
    let pixel_idx = id.x;
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
fn invert_log(@builtin(global_invocation_id) id: vec3<u32>) {
    let pixel_idx = id.x;
    if (pixel_idx >= params.pixel_count) {
        return;
    }

    let idx = pixel_idx * 3u;
    let r = max(pixels[idx], 0.0001);
    let g = max(pixels[idx + 1u], 0.0001);
    let b = max(pixels[idx + 2u], 0.0001);

    // Pre-compute log of base
    let log_base_r = log(max(params.base_r, 0.0001)) / log(10.0);
    let log_base_g = log(max(params.base_g, 0.0001)) / log(10.0);
    let log_base_b = log(max(params.base_b, 0.0001)) / log(10.0);

    // Log inversion: 10^(log_base - log_pixel)
    let log_r = log(r) / log(10.0);
    let log_g = log(g) / log(10.0);
    let log_b = log(b) / log(10.0);

    let inv_r = pow(10.0, log_base_r - log_r);
    let inv_g = pow(10.0, log_base_g - log_g);
    let inv_b = pow(10.0, log_base_b - log_b);

    pixels[idx] = clamp(inv_r, 0.0, 10.0);
    pixels[idx + 1u] = clamp(inv_g, 0.0, 10.0);
    pixels[idx + 2u] = clamp(inv_b, 0.0, 10.0);
}

// Divide-blend inversion: mimics Photoshop's divide blend workflow
// 1. Divide by base: divided = pixel / base
// 2. Apply gamma: gamma_result = divided^(1/2.2)
// 3. Invert: result = 1.0 - gamma_result
@compute @workgroup_size(256)
fn invert_divide(@builtin(global_invocation_id) id: vec3<u32>) {
    let pixel_idx = id.x;
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
fn invert_mask_aware(@builtin(global_invocation_id) id: vec3<u32>) {
    let pixel_idx = id.x;
    if (pixel_idx >= params.pixel_count) {
        return;
    }

    let idx = pixel_idx * 3u;
    let r = pixels[idx];
    let g = pixels[idx + 1u];
    let b = pixels[idx + 2u];

    // Standard inversion: 1.0 - (pixel / base)
    var inv_r = 1.0 - (r / max(params.base_r, 0.0001));
    var inv_g = 1.0 - (g / max(params.base_g, 0.0001));
    var inv_b = 1.0 - (b / max(params.base_b, 0.0001));

    // Apply shadow floor correction for green and blue
    // Removes blue cast from naive mask inversion
    if (params.green_floor > 0.0) {
        inv_g = (inv_g - params.green_floor) / (1.0 - params.green_floor);
    }
    if (params.blue_floor > 0.0) {
        inv_b = (inv_b - params.blue_floor) / (1.0 - params.blue_floor);
    }

    pixels[idx] = clamp(inv_r, 0.0, 1.0);
    pixels[idx + 1u] = clamp(inv_g, 0.0, 1.0);
    pixels[idx + 2u] = clamp(inv_b, 0.0, 1.0);
}
