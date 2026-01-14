// Curves-based (CB) pipeline inversion shader.

struct CbInversionParams {
    white_r: f32,
    black_r: f32,
    white_g: f32,
    black_g: f32,
    white_b: f32,
    black_b: f32,
    is_negative: u32,
    pixel_count: u32,
}

@group(0) @binding(0) var<storage, read_write> pixels: array<f32>;
@group(0) @binding(1) var<uniform> params: CbInversionParams;

const WORKGROUP_SIZE: u32 = 256u;

fn get_pixel_index(id: vec3<u32>, num_workgroups: vec3<u32>) -> u32 {
    return id.y * num_workgroups.x * WORKGROUP_SIZE + id.x;
}

@compute @workgroup_size(256)
fn invert_cb(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let pixel_idx = get_pixel_index(id, num_workgroups);
    if (pixel_idx >= params.pixel_count) {
        return;
    }

    let idx = pixel_idx * 3u;
    let r_range = max(params.white_r - params.black_r, 1.0);
    let g_range = max(params.white_g - params.black_g, 1.0);
    let b_range = max(params.white_b - params.black_b, 1.0);

    let r_255 = pixels[idx] * 255.0;
    let g_255 = pixels[idx + 1u] * 255.0;
    let b_255 = pixels[idx + 2u] * 255.0;

    let r_norm = clamp((r_255 - params.black_r) / r_range, 0.0, 1.0);
    let g_norm = clamp((g_255 - params.black_g) / g_range, 0.0, 1.0);
    let b_norm = clamp((b_255 - params.black_b) / b_range, 0.0, 1.0);

    if (params.is_negative != 0u) {
        pixels[idx] = 1.0 - r_norm;
        pixels[idx + 1u] = 1.0 - g_norm;
        pixels[idx + 2u] = 1.0 - b_norm;
    } else {
        pixels[idx] = r_norm;
        pixels[idx + 1u] = g_norm;
        pixels[idx + 2u] = b_norm;
    }
}
