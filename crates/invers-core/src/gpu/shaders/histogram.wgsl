// Histogram shaders using atomic operations.
// Accumulates per-channel histograms with 65536 buckets.

const NUM_BUCKETS: u32 = 65536u;

@group(0) @binding(0) var<storage, read> pixels: array<f32>;
@group(0) @binding(1) var<storage, read_write> histogram_r: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> histogram_g: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> histogram_b: array<atomic<u32>>;

// Convert float value to histogram bucket index
fn value_to_bucket(value: f32) -> u32 {
    let clamped = clamp(value, 0.0, 1.0);
    return min(u32(clamped * f32(NUM_BUCKETS - 1u)), NUM_BUCKETS - 1u);
}

// Accumulate histogram using global atomics
// Each thread processes one pixel
@compute @workgroup_size(256)
fn accumulate_histogram(@builtin(global_invocation_id) id: vec3<u32>) {
    let pixel_count = arrayLength(&pixels) / 3u;
    let pixel_idx = id.x;

    if (pixel_idx >= pixel_count) {
        return;
    }

    let idx = pixel_idx * 3u;
    let r = pixels[idx];
    let g = pixels[idx + 1u];
    let b = pixels[idx + 2u];

    // Get bucket indices
    let bucket_r = value_to_bucket(r);
    let bucket_g = value_to_bucket(g);
    let bucket_b = value_to_bucket(b);

    // Atomic increment
    atomicAdd(&histogram_r[bucket_r], 1u);
    atomicAdd(&histogram_g[bucket_g], 1u);
    atomicAdd(&histogram_b[bucket_b], 1u);
}

// Clear histogram buffers
// Uses the same bindings as accumulate but overwrites instead of incrementing
// This entry point uses a different bind group layout (3 storage buffers, no pixels)
@compute @workgroup_size(256)
fn clear_histogram(@builtin(global_invocation_id) id: vec3<u32>) {
    let bucket_idx = id.x;

    if (bucket_idx >= NUM_BUCKETS) {
        return;
    }

    // Use atomicExchange to set values to 0 (atomicStore not universally supported)
    // This effectively clears the histogram bucket
    _ = atomicExchange(&histogram_r[bucket_idx], 0u);
    _ = atomicExchange(&histogram_g[bucket_idx], 0u);
    _ = atomicExchange(&histogram_b[bucket_idx], 0u);
}
