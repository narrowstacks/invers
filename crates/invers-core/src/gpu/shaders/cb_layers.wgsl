// Curves-based (CB) pipeline adjustment layers shader.

struct CbLayerParams {
    wb_offsets: vec4<f32>,
    wb_gamma: vec4<f32>,
    color_offsets: vec4<f32>,
    tonal_0: vec4<f32>, // exposure_factor, brightness_gamma, contrast, highlights
    tonal_1: vec4<f32>, // shadows, blacks, whites, shadow_range
    tonal_2: vec4<f32>, // highlight_range, unused...
    shadow_colors: vec4<f32>,
    highlight_colors: vec4<f32>,
    flags: vec4<u32>, // wb_method, layer_order, pixel_count, apply_flags
}

@group(0) @binding(0) var<storage, read_write> pixels: array<f32>;
@group(0) @binding(1) var<uniform> params: CbLayerParams;

const WORKGROUP_SIZE: u32 = 256u;
const SHADOW_THRESHOLD: f32 = 0.1;
const HIGHLIGHT_THRESHOLD: f32 = 0.8;

fn get_pixel_index(id: vec3<u32>, num_workgroups: vec3<u32>) -> u32 {
    return id.y * num_workgroups.x * WORKGROUP_SIZE + id.x;
}

fn clamp01(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

fn atanh(x: f32) -> f32 {
    return 0.5 * log((1.0 + x) / (1.0 - x));
}

fn sigmoid_core(contrast: f32, midpoint: f32, x: f32) -> f32 {
    return tanh(0.5 * contrast * (x - midpoint));
}

fn apply_sigmoid(contrast: f32, midpoint: f32, x: f32, range: f32) -> f32 {
    let r = select(1.0, range, range != 0.0);
    let sig_0 = sigmoid_core(contrast, midpoint, 0.0);
    let sig_1 = sigmoid_core(contrast, midpoint, 1.0);
    let sig_x = sigmoid_core(contrast, midpoint, x / r);
    return ((sig_x - sig_0) / (sig_1 - sig_0)) * r;
}

fn apply_inverse_sigmoid(contrast: f32, midpoint: f32, x: f32, range: f32) -> f32 {
    let r = select(1.0, range, range != 0.0);
    let sig_0 = sigmoid_core(contrast, midpoint, 0.0);
    let sig_1 = sigmoid_core(contrast, midpoint, 1.0);
    var arg = (sig_1 - sig_0) * x / r + sig_0;
    arg = clamp(arg, -0.9999, 0.9999);
    return (midpoint + 2.0 / contrast * atanh(arg)) * r;
}

fn apply_wb_value(value: f32, offset: f32, gamma: f32, method: u32) -> f32 {
    let v = value;
    if (method == 0u) { // LinearFixed
        if (v > 0.0 && v < 1.0) {
            if (v > HIGHLIGHT_THRESHOLD) {
                let blend = 1.0 - HIGHLIGHT_THRESHOLD;
                return clamp01(
                    offset + (blend * v - offset * v + offset * HIGHLIGHT_THRESHOLD) / blend
                );
            } else if (v < SHADOW_THRESHOLD) {
                return clamp01(v + offset * v / SHADOW_THRESHOLD);
            }
            return clamp01(v + offset);
        }
        return clamp01(v);
    } else if (method == 1u) { // LinearDynamic
        return clamp01(v + offset);
    } else if (method == 2u) { // ShadowWeighted
        return clamp01(pow(v, 1.0 / gamma));
    } else if (method == 3u) { // HighlightWeighted
        return clamp01(1.0 - pow(1.0 - v, gamma));
    }
    return clamp01((pow(v, 1.0 / gamma) + 1.0 - pow(1.0 - v, gamma)) / 2.0);
}

fn apply_gamma(value: f32, gamma: f32) -> f32 {
    return pow(value, gamma);
}

fn apply_exposure(value: f32, exposure: f32) -> f32 {
    if (exposure < 1.0) {
        return 1.0 - pow(1.0 - value, 1.0 / exposure);
    } else if (exposure > 1.0) {
        return value * (1.0 - (1.0 - pow(2.0, -exposure)) * 0.4);
    }
    return value;
}

fn apply_contrast(value: f32, contrast: f32) -> f32 {
    let midpoint = 0.5;
    let scale = 0.2;
    if (contrast >= 1.0) {
        let c = 1.0 + contrast * scale;
        return apply_sigmoid(c, midpoint, value, 1.0);
    } else if (contrast <= -1.0) {
        let c = 1.0 + abs(contrast) * scale * 0.5;
        return apply_inverse_sigmoid(c, midpoint, value, 1.0);
    }
    return value;
}

fn apply_highlights(value: f32, highlights: f32) -> f32 {
    let midpoint = 0.75;
    let range = 0.9;
    let scale = 0.1;
    let strength = 0.5 + abs(highlights) * scale;
    if (highlights >= 1.0) {
        if (value > 1.0 - range) {
            return 1.0 - apply_sigmoid(strength, midpoint, 1.0 - value, range);
        }
    } else if (highlights <= -1.0) {
        if (value > 1.0 - range) {
            return 1.0 - apply_inverse_sigmoid(strength, midpoint, 1.0 - value, range);
        }
    }
    return value;
}

fn apply_shadows(value: f32, shadows: f32) -> f32 {
    let midpoint = 0.75;
    let range = 0.9;
    let strength = 0.5 + abs(shadows) * 0.1;
    if (shadows > 0.0) {
        if (value < range) {
            return apply_inverse_sigmoid(strength, midpoint, value, range);
        }
    } else if (shadows < 0.0) {
        if (value < range) {
            return apply_sigmoid(strength, midpoint, value, range);
        }
    }
    return value;
}

fn apply_blacks(value: f32, blacks: f32, shadow_range: f32) -> f32 {
    let decay = (-shadow_range + 1.0) * 0.33 + 5.0;
    let midpoint = 0.75;
    let range = 0.5;
    if (blacks >= 1.0) {
        let lift = blacks / 255.0;
        if (value < 0.9) {
            return lift * exp(-value * decay) + value;
        }
    } else if (blacks <= -1.0) {
        let strength = 0.5 + abs(blacks) * 0.1;
        if (value < range) {
            return apply_sigmoid(strength, midpoint, value, range);
        }
    }
    return value;
}

fn apply_whites(value: f32, whites: f32, highlight_range: f32) -> f32 {
    let decay = (-highlight_range + 1.0) * 0.33 + 5.0;
    let midpoint = 0.75;
    let range = 0.5;
    if (whites <= -1.0) {
        let lift = -whites / 255.0;
        if (value > 0.1) {
            return 1.0 - lift * exp(-(1.0 - value) * decay) - (1.0 - value);
        }
    } else if (whites >= 1.0) {
        let strength = 0.5 + abs(whites) * 0.1;
        if (value > 1.0 - range) {
            return 1.0 - apply_sigmoid(strength, midpoint, 1.0 - value, range);
        }
    }
    return value;
}

fn apply_color_gamma(value: f32, color_offset: f32) -> f32 {
    let blend_range = 0.2;
    let offset = (1.0 - color_offset) / 4.0;
    let adjusted = value - offset;
    if (value > 0.0 && value < 1.0) {
        if (adjusted >= 1.0) {
            return 1.0;
        } else if (value > 1.0 - blend_range) {
            return value - offset * (1.0 - value) / blend_range;
        } else if (adjusted <= 0.0) {
            return 0.0;
        }
        return adjusted;
    }
    return value;
}

fn apply_shadow_tone(value: f32, shadow_color: f32, shadow_range: f32) -> f32 {
    let range = 0.9 - (10.0 - shadow_range) * 0.0444;
    let midpoint = 0.75;
    let scale = 0.125;
    let color_shift = -shadow_color;
    if (color_shift == 0.0) {
        return value;
    }
    let strength = 0.75 + abs(color_shift) * (1.0 + (10.0 - shadow_range) / 18.0) * scale;
    if (value < range) {
        if (color_shift > 0.0) {
            return apply_inverse_sigmoid(strength, midpoint, value, range);
        }
        return apply_sigmoid(strength, midpoint, value, range);
    }
    return value;
}

fn apply_highlight_tone(value: f32, highlight_color: f32, highlight_range: f32) -> f32 {
    let range = 0.9 - (10.0 - highlight_range) * 0.0444;
    let midpoint = 0.75;
    let scale = 0.125;
    let color_shift = -highlight_color;
    if (color_shift == 0.0) {
        return value;
    }
    let strength = 0.75 + abs(color_shift) * (1.0 + (10.0 - highlight_range) / 18.0) * scale;
    if (value > 1.0 - range) {
        if (color_shift > 0.0) {
            return 1.0 - apply_sigmoid(strength, midpoint, 1.0 - value, range);
        }
        return 1.0 - apply_inverse_sigmoid(strength, midpoint, 1.0 - value, range);
    }
    return value;
}

// Helper: Apply all tonal adjustments (exposure, brightness, contrast, highlights, shadows, blacks, whites)
fn apply_tonal_adjustments(
    r: ptr<function, f32>,
    g: ptr<function, f32>,
    b: ptr<function, f32>,
    exposure_factor: f32,
    brightness_gamma: f32,
    contrast: f32,
    highlights: f32,
    shadows: f32,
    blacks: f32,
    whites: f32,
    shadow_range: f32,
    highlight_range: f32
) {
    if (exposure_factor != 1.0) {
        *r = apply_exposure(*r, exposure_factor);
        *g = apply_exposure(*g, exposure_factor);
        *b = apply_exposure(*b, exposure_factor);
    }
    if (brightness_gamma != 1.0) {
        *r = apply_gamma(*r, brightness_gamma);
        *g = apply_gamma(*g, brightness_gamma);
        *b = apply_gamma(*b, brightness_gamma);
    }
    if (contrast >= 1.0 || contrast <= -1.0) {
        *r = apply_contrast(*r, contrast);
        *g = apply_contrast(*g, contrast);
        *b = apply_contrast(*b, contrast);
    }
    if (highlights >= 1.0 || highlights <= -1.0) {
        *r = apply_highlights(*r, highlights);
        *g = apply_highlights(*g, highlights);
        *b = apply_highlights(*b, highlights);
    }
    if (shadows != 0.0) {
        *r = apply_shadows(*r, shadows);
        *g = apply_shadows(*g, shadows);
        *b = apply_shadows(*b, shadows);
    }
    if (blacks >= 1.0 || blacks <= -1.0) {
        *r = apply_blacks(*r, blacks, shadow_range);
        *g = apply_blacks(*g, blacks, shadow_range);
        *b = apply_blacks(*b, blacks, shadow_range);
    }
    if (whites >= 1.0 || whites <= -1.0) {
        *r = apply_whites(*r, whites, highlight_range);
        *g = apply_whites(*g, whites, highlight_range);
        *b = apply_whites(*b, whites, highlight_range);
    }
}

// Helper: Apply white balance and color gamma adjustments
fn apply_wb_and_color(
    r: ptr<function, f32>,
    g: ptr<function, f32>,
    b: ptr<function, f32>,
    wb_offsets: vec3<f32>,
    wb_gamma: vec3<f32>,
    color_offsets: vec3<f32>,
    wb_method: u32,
    apply_wb: bool
) {
    if (apply_wb) {
        *r = apply_wb_value(*r, wb_offsets.x, wb_gamma.x, wb_method);
        *g = apply_wb_value(*g, wb_offsets.y, wb_gamma.y, wb_method);
        *b = apply_wb_value(*b, wb_offsets.z, wb_gamma.z, wb_method);
    }

    *r = apply_color_gamma(*r, color_offsets.x);
    *g = apply_color_gamma(*g, color_offsets.y);
    *b = apply_color_gamma(*b, color_offsets.z);
}

@compute @workgroup_size(256)
fn apply_cb_layers(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let pixel_idx = get_pixel_index(id, num_workgroups);
    if (pixel_idx >= params.flags.z) {
        return;
    }

    let idx = pixel_idx * 3u;
    var r = pixels[idx];
    var g = pixels[idx + 1u];
    var b = pixels[idx + 2u];

    let exposure_factor = params.tonal_0.x;
    let brightness_gamma = params.tonal_0.y;
    let contrast = params.tonal_0.z;
    let highlights = params.tonal_0.w;
    let shadows = params.tonal_1.x;
    let blacks = params.tonal_1.y;
    let whites = params.tonal_1.z;
    let shadow_range = params.tonal_1.w;
    let highlight_range = params.tonal_2.x;

    let wb_offsets = vec3<f32>(params.wb_offsets.x, params.wb_offsets.y, params.wb_offsets.z);
    let wb_gamma = vec3<f32>(params.wb_gamma.x, params.wb_gamma.y, params.wb_gamma.z);
    let color_offsets = vec3<f32>(params.color_offsets.x, params.color_offsets.y, params.color_offsets.z);

    let wb_method = params.flags.x;
    let layer_order = params.flags.y;
    let apply_flags = params.flags.w;
    let apply_wb = (apply_flags & 1u) != 0u;
    let do_shadow_tone = (apply_flags & 2u) != 0u;
    let do_highlight_tone = (apply_flags & 4u) != 0u;

    if (layer_order == 0u) {
        // Order 0: WB + color gamma first, then tonal adjustments
        apply_wb_and_color(&r, &g, &b, wb_offsets, wb_gamma, color_offsets, wb_method, apply_wb);
        apply_tonal_adjustments(&r, &g, &b, exposure_factor, brightness_gamma, contrast, highlights, shadows, blacks, whites, shadow_range, highlight_range);
    } else {
        // Order 1: Tonal adjustments first, then WB + color gamma
        apply_tonal_adjustments(&r, &g, &b, exposure_factor, brightness_gamma, contrast, highlights, shadows, blacks, whites, shadow_range, highlight_range);
        apply_wb_and_color(&r, &g, &b, wb_offsets, wb_gamma, color_offsets, wb_method, apply_wb);
    }

    if (do_shadow_tone) {
        r = apply_shadow_tone(r, params.shadow_colors.x, shadow_range);
        g = apply_shadow_tone(g, params.shadow_colors.y, shadow_range);
        b = apply_shadow_tone(b, params.shadow_colors.z, shadow_range);
    }

    if (do_highlight_tone) {
        r = apply_highlight_tone(r, params.highlight_colors.x, highlight_range);
        g = apply_highlight_tone(g, params.highlight_colors.y, highlight_range);
        b = apply_highlight_tone(b, params.highlight_colors.z, highlight_range);
    }

    pixels[idx] = clamp01(r);
    pixels[idx + 1u] = clamp01(g);
    pixels[idx + 2u] = clamp01(b);
}
