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

// Apply white balance - optimized with reduced branching
fn apply_wb_value(value: f32, offset: f32, gamma: f32, method: u32) -> f32 {
    let v = value;

    // Pre-compute all possible results to reduce divergent branches
    // Method 0: LinearFixed with region-based blending
    let blend = 1.0 - HIGHLIGHT_THRESHOLD;
    let linear_fixed_highlight = offset + (blend * v - offset * v + offset * HIGHLIGHT_THRESHOLD) / blend;
    let linear_fixed_shadow = v + offset * v / SHADOW_THRESHOLD;
    let linear_fixed_mid = v + offset;

    // Select LinearFixed result based on value region (branchless for inner selection)
    let in_valid_range = v > 0.0 && v < 1.0;
    let is_highlight = v > HIGHLIGHT_THRESHOLD;
    let is_shadow = v < SHADOW_THRESHOLD;
    let linear_fixed_result = select(
        select(linear_fixed_mid, linear_fixed_shadow, is_shadow),
        linear_fixed_highlight,
        is_highlight
    );
    let method0_result = select(v, linear_fixed_result, in_valid_range);

    // Method 1: LinearDynamic
    let method1_result = v + offset;

    // Method 2: ShadowWeighted gamma
    let method2_result = pow(v, 1.0 / gamma);

    // Method 3: HighlightWeighted gamma
    let method3_result = 1.0 - pow(1.0 - v, gamma);

    // Method 4: Balanced (default)
    let method4_result = (method2_result + method3_result) / 2.0;

    // Select result based on method using chained select (reduces divergence)
    let result = select(
        select(
            select(
                select(method4_result, method3_result, method == 3u),
                method2_result, method == 2u
            ),
            method1_result, method == 1u
        ),
        method0_result, method == 0u
    );

    return clamp01(result);
}

fn apply_gamma(value: f32, gamma: f32) -> f32 {
    return pow(value, gamma);
}

// Apply exposure - optimized with select() to reduce branching
fn apply_exposure(value: f32, exposure: f32) -> f32 {
    // Pre-compute both results
    let under_exposed = 1.0 - pow(1.0 - value, 1.0 / exposure);
    let over_exposed = value * (1.0 - (1.0 - pow(2.0, -exposure)) * 0.4);

    // Select based on exposure value (reduces divergent branching)
    return select(
        select(value, over_exposed, exposure > 1.0),
        under_exposed,
        exposure < 1.0
    );
}

// Apply contrast - optimized with select()
fn apply_contrast(value: f32, contrast: f32) -> f32 {
    let midpoint = 0.5;
    let scale = 0.2;
    let c_pos = 1.0 + contrast * scale;
    let c_neg = 1.0 + abs(contrast) * scale * 0.5;

    let sigmoid_result = apply_sigmoid(c_pos, midpoint, value, 1.0);
    let inv_sigmoid_result = apply_inverse_sigmoid(c_neg, midpoint, value, 1.0);

    return select(
        select(value, inv_sigmoid_result, contrast <= -1.0),
        sigmoid_result,
        contrast >= 1.0
    );
}

// Apply highlights - optimized with select()
fn apply_highlights(value: f32, highlights: f32) -> f32 {
    let midpoint = 0.75;
    let range = 0.9;
    let scale = 0.1;
    let strength = 0.5 + abs(highlights) * scale;
    let inv_value = 1.0 - value;
    let in_range = value > 1.0 - range;

    let pos_result = 1.0 - apply_sigmoid(strength, midpoint, inv_value, range);
    let neg_result = 1.0 - apply_inverse_sigmoid(strength, midpoint, inv_value, range);

    // Apply only if in range and highlights active
    let adjusted = select(
        select(value, neg_result, highlights <= -1.0),
        pos_result,
        highlights >= 1.0
    );
    return select(value, adjusted, in_range);
}

// Apply shadows - optimized with select()
fn apply_shadows(value: f32, shadows: f32) -> f32 {
    let midpoint = 0.75;
    let range = 0.9;
    let strength = 0.5 + abs(shadows) * 0.1;
    let in_range = value < range;

    let pos_result = apply_inverse_sigmoid(strength, midpoint, value, range);
    let neg_result = apply_sigmoid(strength, midpoint, value, range);

    let adjusted = select(
        select(value, neg_result, shadows < 0.0),
        pos_result,
        shadows > 0.0
    );
    return select(value, adjusted, in_range);
}

// Apply blacks - optimized with select()
fn apply_blacks(value: f32, blacks: f32, shadow_range: f32) -> f32 {
    let decay = (-shadow_range + 1.0) * 0.33 + 5.0;
    let midpoint = 0.75;
    let range = 0.5;
    let lift = blacks / 255.0;
    let strength = 0.5 + abs(blacks) * 0.1;

    let pos_result = lift * exp(-value * decay) + value;
    let neg_result = apply_sigmoid(strength, midpoint, value, range);

    let pos_active = blacks >= 1.0 && value < 0.9;
    let neg_active = blacks <= -1.0 && value < range;

    return select(select(value, neg_result, neg_active), pos_result, pos_active);
}

// Apply whites - optimized with select()
fn apply_whites(value: f32, whites: f32, highlight_range: f32) -> f32 {
    let decay = (-highlight_range + 1.0) * 0.33 + 5.0;
    let midpoint = 0.75;
    let range = 0.5;
    let lift = -whites / 255.0;
    let strength = 0.5 + abs(whites) * 0.1;
    let inv_value = 1.0 - value;

    let neg_result = 1.0 - lift * exp(-inv_value * decay) - inv_value;
    let pos_result = 1.0 - apply_sigmoid(strength, midpoint, inv_value, range);

    let neg_active = whites <= -1.0 && value > 0.1;
    let pos_active = whites >= 1.0 && value > 1.0 - range;

    return select(select(value, pos_result, pos_active), neg_result, neg_active);
}

// Apply color gamma - optimized with select()
fn apply_color_gamma(value: f32, color_offset: f32) -> f32 {
    let blend_range = 0.2;
    let offset = (1.0 - color_offset) / 4.0;
    let adjusted = value - offset;
    let in_valid_range = value > 0.0 && value < 1.0;
    let in_blend_zone = value > 1.0 - blend_range;

    let blend_result = value - offset * (1.0 - value) / blend_range;

    // Nested select for region-based result
    let region_result = select(
        select(adjusted, 0.0, adjusted <= 0.0),
        select(adjusted, blend_result, in_blend_zone),
        adjusted < 1.0
    );
    let clamped_result = select(region_result, 1.0, adjusted >= 1.0);

    return select(value, clamped_result, in_valid_range);
}

// Apply shadow tone - optimized with select()
fn apply_shadow_tone(value: f32, shadow_color: f32, shadow_range: f32) -> f32 {
    let range = 0.9 - (10.0 - shadow_range) * 0.0444;
    let midpoint = 0.75;
    let scale = 0.125;
    let color_shift = -shadow_color;
    let strength = 0.75 + abs(color_shift) * (1.0 + (10.0 - shadow_range) / 18.0) * scale;
    let in_range = value < range;

    let pos_result = apply_inverse_sigmoid(strength, midpoint, value, range);
    let neg_result = apply_sigmoid(strength, midpoint, value, range);

    let adjusted = select(neg_result, pos_result, color_shift > 0.0);
    let has_shift = color_shift != 0.0;

    return select(value, select(value, adjusted, in_range), has_shift);
}

// Apply highlight tone - optimized with select()
fn apply_highlight_tone(value: f32, highlight_color: f32, highlight_range: f32) -> f32 {
    let range = 0.9 - (10.0 - highlight_range) * 0.0444;
    let midpoint = 0.75;
    let scale = 0.125;
    let color_shift = -highlight_color;
    let strength = 0.75 + abs(color_shift) * (1.0 + (10.0 - highlight_range) / 18.0) * scale;
    let inv_value = 1.0 - value;
    let in_range = value > 1.0 - range;

    let pos_result = 1.0 - apply_sigmoid(strength, midpoint, inv_value, range);
    let neg_result = 1.0 - apply_inverse_sigmoid(strength, midpoint, inv_value, range);

    let adjusted = select(neg_result, pos_result, color_shift > 0.0);
    let has_shift = color_shift != 0.0;

    return select(value, select(value, adjusted, in_range), has_shift);
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
