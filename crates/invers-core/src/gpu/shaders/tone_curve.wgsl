// Tone curve shaders for contrast adjustment.
// Two modes: S-curve (symmetric) and asymmetric (film-like with toe/shoulder).

// Working range constants
const WORKING_RANGE_FLOOR: f32 = 1.0 / 65535.0;
const WORKING_RANGE_CEILING: f32 = 1.0 - WORKING_RANGE_FLOOR;

// Soft-clip knee size (5% - matches CPU implementation)
const SOFT_CLIP_KNEE: f32 = 0.05;

struct ToneCurveParams {
    strength: f32,
    toe_strength: f32,
    toe_length: f32,
    shoulder_strength: f32,
    shoulder_start: f32,
    pixel_count: u32,
    _padding: vec2<u32>,
}

@group(0) @binding(0) var<storage, read_write> pixels: array<f32>;
@group(0) @binding(1) var<uniform> params: ToneCurveParams;

// Workgroup size for all shaders
const WORKGROUP_SIZE: u32 = 256u;

// Calculate linear pixel index from 2D dispatch grid
// Supports images larger than 65535 workgroups by using 2D dispatch
fn get_pixel_index(id: vec3<u32>, num_workgroups: vec3<u32>) -> u32 {
    return id.y * num_workgroups.x * WORKGROUP_SIZE + id.x;
}

// Soft-clip a value to [0, 1] range with smooth knee
fn soft_clip_highlight(value: f32, knee: f32) -> f32 {
    if (value <= 0.0) {
        return WORKING_RANGE_FLOOR;
    } else if (value <= 1.0 - knee) {
        return value;
    } else if (value >= 1.0 + knee) {
        return WORKING_RANGE_CEILING;
    } else {
        let knee_start = 1.0 - knee;
        let normalized = (value - knee_start) / (2.0 * knee);
        let compressed = 1.0 - exp(-normalized * 2.0);
        return knee_start + knee * compressed;
    }
}

// Smoothstep function: t² × (3 - 2t)
fn smoothstep_value(t: f32) -> f32 {
    return t * t * (3.0 - 2.0 * t);
}

// Apply S-curve to a single value - optimized with select()
fn apply_scurve_point(x: f32, strength: f32) -> f32 {
    // Pre-compute both shadow and highlight paths to avoid divergent branching
    // Shadow region: t = x * 2, result = smoothstep(t) * 0.5
    let t_shadow = x * 2.0;
    let adjusted_shadow = smoothstep_value(t_shadow) * 0.5;

    // Highlight region: t = (x - 0.5) * 2, result = 0.5 + smoothstep(t) * 0.5
    let t_highlight = (x - 0.5) * 2.0;
    let adjusted_highlight = 0.5 + smoothstep_value(t_highlight) * 0.5;

    // Select based on position (branchless)
    let adjusted = select(adjusted_highlight, adjusted_shadow, x < 0.5);

    // Apply contrast around midpoint
    let s_value = (adjusted - 0.5) * (1.0 + strength * 0.5) + 0.5;

    // Blend between original and adjusted
    let result = x * (1.0 - strength) + s_value * strength;

    // Return original if strength is negligible, otherwise return adjusted
    // Clamp to working range (no soft-clip to preserve highlight detail)
    return select(
        clamp(result, WORKING_RANGE_FLOOR, WORKING_RANGE_CEILING),
        x,
        strength < 0.01
    );
}

// S-curve tone mapping (symmetric around midpoint)
@compute @workgroup_size(256)
fn apply_scurve(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let pixel_idx = get_pixel_index(id, num_workgroups);
    if (pixel_idx >= params.pixel_count) {
        return;
    }

    let idx = pixel_idx * 3u;

    pixels[idx] = apply_scurve_point(pixels[idx], params.strength);
    pixels[idx + 1u] = apply_scurve_point(pixels[idx + 1u], params.strength);
    pixels[idx + 2u] = apply_scurve_point(pixels[idx + 2u], params.strength);
}

// Apply asymmetric curve to a single value - optimized with select()
fn apply_asymmetric_point(
    x: f32,
    toe_strength: f32,
    toe_length: f32,
    shoulder_strength: f32,
    shoulder_start: f32
) -> f32 {
    // Pre-compute all three regions to avoid divergent branching

    // Toe region (shadow lift)
    let toe_gamma = 1.0 / (1.0 + toe_strength * 1.5);
    let toe_t = x / max(toe_length, 0.0001);
    let toe_result = toe_length * pow(toe_t, toe_gamma);

    // Shoulder region (highlight compression)
    let shoulder_gamma = 1.0 + shoulder_strength * 2.0;
    let shoulder_range = 1.0 - shoulder_start;
    let shoulder_t = (x - shoulder_start) / max(shoulder_range, 0.0001);
    let shoulder_result = shoulder_start + shoulder_range * (1.0 - pow(1.0 - shoulder_t, shoulder_gamma));

    // Select result based on region (branchless)
    let result = select(
        select(x, shoulder_result, x > shoulder_start),
        toe_result,
        x < toe_length
    );

    // Clamp to working range (no soft-clip to preserve highlight detail)
    return clamp(result, WORKING_RANGE_FLOOR, WORKING_RANGE_CEILING);
}

// Asymmetric tone curve with separate toe (shadow) and shoulder (highlight) control
@compute @workgroup_size(256)
fn apply_asymmetric(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let pixel_idx = get_pixel_index(id, num_workgroups);
    if (pixel_idx >= params.pixel_count) {
        return;
    }

    let idx = pixel_idx * 3u;

    pixels[idx] = apply_asymmetric_point(
        pixels[idx],
        params.toe_strength,
        params.toe_length,
        params.shoulder_strength,
        params.shoulder_start
    );
    pixels[idx + 1u] = apply_asymmetric_point(
        pixels[idx + 1u],
        params.toe_strength,
        params.toe_length,
        params.shoulder_strength,
        params.shoulder_start
    );
    pixels[idx + 2u] = apply_asymmetric_point(
        pixels[idx + 2u],
        params.toe_strength,
        params.toe_length,
        params.shoulder_strength,
        params.shoulder_start
    );
}
