// Tone curve shaders for contrast adjustment.
// Two modes: S-curve (symmetric) and asymmetric (film-like with toe/shoulder).

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

// Smoothstep function: t² × (3 - 2t)
fn smoothstep_value(t: f32) -> f32 {
    return t * t * (3.0 - 2.0 * t);
}

// Apply S-curve to a single value
fn apply_scurve_point(x: f32, strength: f32) -> f32 {
    if (strength < 0.01) {
        return x;
    }

    var adjusted: f32;
    if (x < 0.5) {
        // Shadow region
        let t = x * 2.0;
        let smoothed = smoothstep_value(t);
        adjusted = smoothed * 0.5;
    } else {
        // Highlight region
        let t = (x - 0.5) * 2.0;
        let smoothed = smoothstep_value(t);
        adjusted = 0.5 + smoothed * 0.5;
    }

    // Apply contrast around midpoint
    let s_value = (adjusted - 0.5) * (1.0 + strength * 0.5) + 0.5;

    // Blend between original and adjusted
    let result = x * (1.0 - strength) + s_value * strength;

    return clamp(result, 0.0, 1.0);
}

// S-curve tone mapping (symmetric around midpoint)
@compute @workgroup_size(256)
fn apply_scurve(@builtin(global_invocation_id) id: vec3<u32>) {
    let pixel_idx = id.x;
    if (pixel_idx >= params.pixel_count) {
        return;
    }

    let idx = pixel_idx * 3u;

    pixels[idx] = apply_scurve_point(pixels[idx], params.strength);
    pixels[idx + 1u] = apply_scurve_point(pixels[idx + 1u], params.strength);
    pixels[idx + 2u] = apply_scurve_point(pixels[idx + 2u], params.strength);
}

// Apply asymmetric curve to a single value
fn apply_asymmetric_point(
    x: f32,
    toe_strength: f32,
    toe_length: f32,
    shoulder_strength: f32,
    shoulder_start: f32
) -> f32 {
    var result: f32;

    if (x < toe_length) {
        // Toe region (shadow lift)
        let gamma = 1.0 / (1.0 + toe_strength * 1.5);
        let t = x / toe_length;
        result = toe_length * pow(t, gamma);
    } else if (x > shoulder_start) {
        // Shoulder region (highlight compression)
        let gamma = 1.0 + shoulder_strength * 2.0;
        let range = 1.0 - shoulder_start;
        let t = (x - shoulder_start) / range;
        result = shoulder_start + range * (1.0 - pow(1.0 - t, gamma));
    } else {
        // Mid region (linear passthrough)
        result = x;
    }

    return clamp(result, 0.0, 1.0);
}

// Asymmetric tone curve with separate toe (shadow) and shoulder (highlight) control
@compute @workgroup_size(256)
fn apply_asymmetric(@builtin(global_invocation_id) id: vec3<u32>) {
    let pixel_idx = id.x;
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
