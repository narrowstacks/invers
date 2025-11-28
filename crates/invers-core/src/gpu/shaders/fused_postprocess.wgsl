// Fused post-processing shader: combines gains + color matrix + tone curve + clamp
// into a single GPU dispatch to minimize memory bandwidth and dispatch overhead.

struct FusedPostprocessParams {
    // Gains (auto-levels)
    gain_r: f32,
    gain_g: f32,
    gain_b: f32,
    offset_r: f32,
    offset_g: f32,
    offset_b: f32,

    // Color matrix (row-major 3x3)
    m00: f32, m01: f32, m02: f32,
    m10: f32, m11: f32, m12: f32,
    m20: f32, m21: f32, m22: f32,

    // Tone curve
    tone_strength: f32,
    toe_strength: f32,
    toe_length: f32,
    shoulder_strength: f32,
    shoulder_start: f32,

    // Exposure
    exposure_multiplier: f32,

    // Control flags
    // bit 0: apply gains
    // bit 1: apply color matrix
    // bit 2: apply tone curve (scurve)
    // bit 3: apply tone curve (asymmetric)
    // bit 4: apply clamp
    // bit 5: apply exposure
    flags: u32,

    pixel_count: u32,
}

@group(0) @binding(0) var<storage, read_write> pixels: array<f32>;
@group(0) @binding(1) var<uniform> params: FusedPostprocessParams;

const WORKGROUP_SIZE: u32 = 256u;

const FLAG_GAINS: u32 = 1u;           // bit 0
const FLAG_COLOR_MATRIX: u32 = 2u;    // bit 1
const FLAG_TONE_SCURVE: u32 = 4u;     // bit 2
const FLAG_TONE_ASYMMETRIC: u32 = 8u; // bit 3
const FLAG_CLAMP: u32 = 16u;          // bit 4
const FLAG_EXPOSURE: u32 = 32u;       // bit 5

const WORKING_RANGE_FLOOR: f32 = 1.0 / 65535.0;
const WORKING_RANGE_CEILING: f32 = 1.0 - WORKING_RANGE_FLOOR;

fn get_pixel_index(id: vec3<u32>, num_workgroups: vec3<u32>) -> u32 {
    return id.y * num_workgroups.x * WORKGROUP_SIZE + id.x;
}

fn apply_gains(color: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        clamp((color.x - params.offset_r) * params.gain_r, 0.0, 1.0),
        clamp((color.y - params.offset_g) * params.gain_g, 0.0, 1.0),
        clamp((color.z - params.offset_b) * params.gain_b, 0.0, 1.0)
    );
}

fn apply_color_matrix(color: vec3<f32>) -> vec3<f32> {
    let new_r = params.m00 * color.x + params.m01 * color.y + params.m02 * color.z;
    let new_g = params.m10 * color.x + params.m11 * color.y + params.m12 * color.z;
    let new_b = params.m20 * color.x + params.m21 * color.y + params.m22 * color.z;

    return vec3<f32>(
        clamp(new_r, WORKING_RANGE_FLOOR, WORKING_RANGE_CEILING),
        clamp(new_g, WORKING_RANGE_FLOOR, WORKING_RANGE_CEILING),
        clamp(new_b, WORKING_RANGE_FLOOR, WORKING_RANGE_CEILING)
    );
}

fn smoothstep_value(t: f32) -> f32 {
    return t * t * (3.0 - 2.0 * t);
}

fn apply_scurve_point(x: f32, strength: f32) -> f32 {
    if (strength < 0.01) {
        return x;
    }

    var adjusted: f32;
    if (x < 0.5) {
        let t = x * 2.0;
        adjusted = smoothstep_value(t) * 0.5;
    } else {
        let t = (x - 0.5) * 2.0;
        adjusted = 0.5 + smoothstep_value(t) * 0.5;
    }

    let s_value = (adjusted - 0.5) * (1.0 + strength * 0.5) + 0.5;
    return clamp(x * (1.0 - strength) + s_value * strength, 0.0, 1.0);
}

fn apply_tone_scurve(color: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        apply_scurve_point(color.x, params.tone_strength),
        apply_scurve_point(color.y, params.tone_strength),
        apply_scurve_point(color.z, params.tone_strength)
    );
}

fn apply_asymmetric_point(x: f32) -> f32 {
    var result: f32;

    if (x < params.toe_length) {
        let gamma = 1.0 / (1.0 + params.toe_strength * 1.5);
        let t = x / params.toe_length;
        result = params.toe_length * pow(t, gamma);
    } else if (x > params.shoulder_start) {
        let gamma = 1.0 + params.shoulder_strength * 2.0;
        let range = 1.0 - params.shoulder_start;
        let t = (x - params.shoulder_start) / range;
        result = params.shoulder_start + range * (1.0 - pow(1.0 - t, gamma));
    } else {
        result = x;
    }

    return clamp(result, 0.0, 1.0);
}

fn apply_tone_asymmetric(color: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        apply_asymmetric_point(color.x),
        apply_asymmetric_point(color.y),
        apply_asymmetric_point(color.z)
    );
}

fn apply_exposure(color: vec3<f32>) -> vec3<f32> {
    return clamp(color * params.exposure_multiplier, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn apply_clamp(color: vec3<f32>) -> vec3<f32> {
    return clamp(color, vec3<f32>(WORKING_RANGE_FLOOR), vec3<f32>(WORKING_RANGE_CEILING));
}

@compute @workgroup_size(256)
fn fused_postprocess_main(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let pixel_idx = get_pixel_index(id, num_workgroups);
    if (pixel_idx >= params.pixel_count) {
        return;
    }

    let idx = pixel_idx * 3u;
    var color = vec3<f32>(pixels[idx], pixels[idx + 1u], pixels[idx + 2u]);

    // Step 1: Apply gains (auto-levels)
    if ((params.flags & FLAG_GAINS) != 0u) {
        color = apply_gains(color);
    }

    // Step 2: Apply exposure
    if ((params.flags & FLAG_EXPOSURE) != 0u) {
        color = apply_exposure(color);
    }

    // Step 3: Apply color matrix
    if ((params.flags & FLAG_COLOR_MATRIX) != 0u) {
        color = apply_color_matrix(color);
    }

    // Step 4: Apply tone curve (either scurve or asymmetric, not both)
    if ((params.flags & FLAG_TONE_ASYMMETRIC) != 0u) {
        color = apply_tone_asymmetric(color);
    } else if ((params.flags & FLAG_TONE_SCURVE) != 0u) {
        color = apply_tone_scurve(color);
    }

    // Step 5: Final clamp
    if ((params.flags & FLAG_CLAMP) != 0u) {
        color = apply_clamp(color);
    }

    pixels[idx] = color.x;
    pixels[idx + 1u] = color.y;
    pixels[idx + 2u] = color.z;
}
