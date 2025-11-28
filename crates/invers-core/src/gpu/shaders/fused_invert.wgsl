// Fused inversion shader: combines inversion + shadow lift + highlight compression
// into a single GPU dispatch to minimize memory bandwidth and dispatch overhead.

struct FusedInvertParams {
    // Inversion parameters
    base_r: f32,
    base_g: f32,
    base_b: f32,
    green_floor: f32,
    blue_floor: f32,
    bw_headroom: f32,

    // Shadow lift parameters
    shadow_lift: f32,

    // Highlight compression parameters
    highlight_threshold: f32,
    highlight_compression: f32,

    // Control flags (packed as bits in u32)
    // bit 0-2: inversion mode (0=linear, 1=log, 2=divide, 3=mask_aware, 4=bw)
    // bit 3: apply shadow lift
    // bit 4: apply highlight compression
    flags: u32,

    pixel_count: u32,
    _padding: u32,
}

@group(0) @binding(0) var<storage, read_write> pixels: array<f32>;
@group(0) @binding(1) var<uniform> params: FusedInvertParams;

const WORKGROUP_SIZE: u32 = 256u;

const MODE_LINEAR: u32 = 0u;
const MODE_LOG: u32 = 1u;
const MODE_DIVIDE: u32 = 2u;
const MODE_MASK_AWARE: u32 = 3u;
const MODE_BW: u32 = 4u;

const FLAG_SHADOW_LIFT: u32 = 8u;      // bit 3
const FLAG_HIGHLIGHT_COMPRESS: u32 = 16u; // bit 4

fn get_pixel_index(id: vec3<u32>, num_workgroups: vec3<u32>) -> u32 {
    return id.y * num_workgroups.x * WORKGROUP_SIZE + id.x;
}

fn get_inversion_mode() -> u32 {
    return params.flags & 7u; // bits 0-2
}

// Inversion functions (inlined for performance)
fn invert_linear(r: f32, g: f32, b: f32) -> vec3<f32> {
    return vec3<f32>(
        (params.base_r - r) / params.base_r,
        (params.base_g - g) / params.base_g,
        (params.base_b - b) / params.base_b
    );
}

fn invert_log(r: f32, g: f32, b: f32) -> vec3<f32> {
    let safe_r = max(r, 0.0001);
    let safe_g = max(g, 0.0001);
    let safe_b = max(b, 0.0001);

    // Use log10 directly via ln(x)/ln(10)
    let ln10 = 2.302585093;
    let log_base_r = log(max(params.base_r, 0.0001)) / ln10;
    let log_base_g = log(max(params.base_g, 0.0001)) / ln10;
    let log_base_b = log(max(params.base_b, 0.0001)) / ln10;

    let log_r = log(safe_r) / ln10;
    let log_g = log(safe_g) / ln10;
    let log_b = log(safe_b) / ln10;

    return vec3<f32>(
        clamp(pow(10.0, log_base_r - log_r), 0.0, 10.0),
        clamp(pow(10.0, log_base_g - log_g), 0.0, 10.0),
        clamp(pow(10.0, log_base_b - log_b), 0.0, 10.0)
    );
}

fn invert_divide(r: f32, g: f32, b: f32) -> vec3<f32> {
    let gamma = 1.0 / 2.2;

    let div_r = clamp(r / max(params.base_r, 0.0001), 0.0, 10.0);
    let div_g = clamp(g / max(params.base_g, 0.0001), 0.0, 10.0);
    let div_b = clamp(b / max(params.base_b, 0.0001), 0.0, 10.0);

    return vec3<f32>(
        clamp(1.0 - pow(div_r, gamma), 0.0, 1.0),
        clamp(1.0 - pow(div_g, gamma), 0.0, 1.0),
        clamp(1.0 - pow(div_b, gamma), 0.0, 1.0)
    );
}

fn invert_mask_aware(r: f32, g: f32, b: f32) -> vec3<f32> {
    var inv_r = 1.0 - (r / max(params.base_r, 0.0001));
    var inv_g = 1.0 - (g / max(params.base_g, 0.0001));
    var inv_b = 1.0 - (b / max(params.base_b, 0.0001));

    if (params.green_floor > 0.0) {
        inv_g = (inv_g - params.green_floor) / (1.0 - params.green_floor);
    }
    if (params.blue_floor > 0.0) {
        inv_b = (inv_b - params.blue_floor) / (1.0 - params.blue_floor);
    }

    return vec3<f32>(
        clamp(inv_r, 0.0, 1.0),
        clamp(inv_g, 0.0, 1.0),
        clamp(inv_b, 0.0, 1.0)
    );
}

fn invert_bw(r: f32, g: f32, b: f32) -> vec3<f32> {
    let base = (params.base_r + params.base_g + params.base_b) / 3.0;
    let black_point = base * (1.0 - params.bw_headroom);
    let scale = 1.0 / max(black_point, 0.0001);

    return vec3<f32>(
        clamp((base - r) * scale, 0.0, 1.0),
        clamp((base - g) * scale, 0.0, 1.0),
        clamp((base - b) * scale, 0.0, 1.0)
    );
}

fn apply_shadow_lift(color: vec3<f32>) -> vec3<f32> {
    return clamp(color + vec3<f32>(params.shadow_lift), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn apply_highlight_compress(color: vec3<f32>) -> vec3<f32> {
    var result = color;
    let threshold = params.highlight_threshold;
    let compression = params.highlight_compression;

    if (result.x > threshold) {
        result.x = threshold + (result.x - threshold) * compression;
    }
    if (result.y > threshold) {
        result.y = threshold + (result.y - threshold) * compression;
    }
    if (result.z > threshold) {
        result.z = threshold + (result.z - threshold) * compression;
    }

    return result;
}

@compute @workgroup_size(256)
fn fused_invert_main(
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

    // Step 1: Inversion (based on mode)
    var color: vec3<f32>;
    let mode = get_inversion_mode();

    switch (mode) {
        case MODE_LINEAR: {
            color = invert_linear(r, g, b);
        }
        case MODE_LOG: {
            color = invert_log(r, g, b);
        }
        case MODE_DIVIDE: {
            color = invert_divide(r, g, b);
        }
        case MODE_MASK_AWARE: {
            color = invert_mask_aware(r, g, b);
        }
        case MODE_BW: {
            color = invert_bw(r, g, b);
        }
        default: {
            color = invert_linear(r, g, b);
        }
    }

    // Step 2: Shadow lift (if enabled)
    if ((params.flags & FLAG_SHADOW_LIFT) != 0u) {
        color = apply_shadow_lift(color);
    }

    // Step 3: Highlight compression (if enabled)
    if ((params.flags & FLAG_HIGHLIGHT_COMPRESS) != 0u) {
        color = apply_highlight_compress(color);
    }

    pixels[idx] = color.x;
    pixels[idx + 1u] = color.y;
    pixels[idx + 2u] = color.z;
}
