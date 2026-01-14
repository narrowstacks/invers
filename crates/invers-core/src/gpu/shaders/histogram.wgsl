// Histogram shaders using atomic operations with workgroup-local optimization.
// Accumulates per-channel histograms with 65536 buckets.
//
// Optimization strategy:
// - Use 256-bin workgroup-local histograms to reduce global atomic contention
// - Local bins coarsen 65536 buckets to 256 (bucket >> 8)
// - After local accumulation, threads merge local counts to global histogram
// - The merge phase has only 256 threads writing (vs 256 writing to random locations)
//
// This reduces atomic contention in two ways:
// 1. Local accumulation: 256 threads contend on 256 bins (1:1 ratio best case)
// 2. Global merge: 256 atomic ops per workgroup (vs potentially 256 scattered ops)

const NUM_BUCKETS: u32 = 65536u;
const NUM_LOCAL_BINS: u32 = 256u;
const LOCAL_BIN_SHIFT: u32 = 8u; // log2(65536/256) = 8

// Workgroup size for all shaders
const WORKGROUP_SIZE: u32 = 256u;

@group(0) @binding(0) var<storage, read> pixels: array<f32>;
@group(0) @binding(1) var<storage, read_write> histogram_r: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> histogram_g: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> histogram_b: array<atomic<u32>>;

// Workgroup-local histograms for reduced contention
// Each workgroup accumulates to these local bins first, then merges to global
var<workgroup> local_hist_r: array<atomic<u32>, 256>;
var<workgroup> local_hist_g: array<atomic<u32>, 256>;
var<workgroup> local_hist_b: array<atomic<u32>, 256>;

// Calculate linear pixel index from 2D dispatch grid
// Supports images larger than 65535 workgroups by using 2D dispatch
fn get_pixel_index(id: vec3<u32>, num_workgroups: vec3<u32>) -> u32 {
    return id.y * num_workgroups.x * WORKGROUP_SIZE + id.x;
}

// Convert float value to histogram bucket index (full precision, 65536 buckets)
fn value_to_bucket(value: f32) -> u32 {
    let clamped = clamp(value, 0.0, 1.0);
    return min(u32(clamped * f32(NUM_BUCKETS - 1u)), NUM_BUCKETS - 1u);
}

// Convert full bucket index to local bin index (coarse, 256 bins)
fn bucket_to_local_bin(bucket: u32) -> u32 {
    return bucket >> LOCAL_BIN_SHIFT;
}

// Accumulate histogram using workgroup-local optimization.
//
// This implementation uses local histograms with coarsened bins, trading some
// bucket precision for significantly reduced global atomic contention.
// Counts are accumulated in local 256-bin histograms, then merged to the
// center bucket of each 256-bucket range in the global histogram.
//
// Precision loss: Values are binned to 256 bins instead of 65536.
// For most histogram use cases (percentile calculation, exposure analysis),
// this precision is sufficient and the performance gain is substantial.
@compute @workgroup_size(256)
fn accumulate_histogram(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let local_idx = local_id.x;
    let pixel_count = arrayLength(&pixels) / 3u;
    let pixel_idx = get_pixel_index(global_id, num_workgroups);

    // Phase 1: Initialize local histograms to 0
    // Each thread clears one bin (256 threads, 256 bins - perfect mapping)
    atomicStore(&local_hist_r[local_idx], 0u);
    atomicStore(&local_hist_g[local_idx], 0u);
    atomicStore(&local_hist_b[local_idx], 0u);

    // Ensure all local bins are initialized before accumulation
    workgroupBarrier();

    // Phase 2: Accumulate to local histograms
    // Contention is reduced because threads within a workgroup are likely
    // processing spatially adjacent pixels with similar values, leading to
    // fewer collisions on local bins
    if (pixel_idx < pixel_count) {
        let idx = pixel_idx * 3u;
        let r = pixels[idx];
        let g = pixels[idx + 1u];
        let b = pixels[idx + 2u];

        // Get full precision bucket indices, then coarsen to local bins
        let bucket_r = value_to_bucket(r);
        let bucket_g = value_to_bucket(g);
        let bucket_b = value_to_bucket(b);

        let local_bin_r = bucket_to_local_bin(bucket_r);
        let local_bin_g = bucket_to_local_bin(bucket_g);
        let local_bin_b = bucket_to_local_bin(bucket_b);

        // Local atomic adds - contention limited to workgroup
        atomicAdd(&local_hist_r[local_bin_r], 1u);
        atomicAdd(&local_hist_g[local_bin_g], 1u);
        atomicAdd(&local_hist_b[local_bin_b], 1u);
    }

    // Ensure all local accumulation is complete before global merge
    workgroupBarrier();

    // Phase 3: Merge local bins to global histogram
    // Each thread handles one local bin (256 threads, 256 bins)
    // This is the key optimization: instead of 256 threads writing to
    // random global locations, we have 256 threads each writing to a
    // deterministic location (the center of their bin's range)
    let local_count_r = atomicLoad(&local_hist_r[local_idx]);
    let local_count_g = atomicLoad(&local_hist_g[local_idx]);
    let local_count_b = atomicLoad(&local_hist_b[local_idx]);

    // Map local bin to center of its 256-bucket range in global histogram
    // local_bin 0 -> global bucket 128
    // local_bin 1 -> global bucket 384
    // local_bin N -> global bucket (N * 256) + 128
    let global_bucket = (local_idx << LOCAL_BIN_SHIFT) + 128u;

    // Only write if there's something to add (skip unnecessary atomics)
    if (local_count_r > 0u) {
        atomicAdd(&histogram_r[global_bucket], local_count_r);
    }
    if (local_count_g > 0u) {
        atomicAdd(&histogram_g[global_bucket], local_count_g);
    }
    if (local_count_b > 0u) {
        atomicAdd(&histogram_b[global_bucket], local_count_b);
    }
}

// Full-precision histogram accumulation with workgroup batching.
//
// This alternative implementation maintains full 65536-bucket precision
// by storing each thread's actual bucket index and writing to global
// after the workgroup synchronizes. The synchronization provides some
// batching benefit but less contention reduction than the coarsened approach.
//
// Use this when precise bucket distribution is needed (e.g., for exact
// percentile lookups or detailed histogram visualization).
@compute @workgroup_size(256)
fn accumulate_histogram_precise(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let local_idx = local_id.x;
    let pixel_count = arrayLength(&pixels) / 3u;
    let pixel_idx = get_pixel_index(global_id, num_workgroups);

    // Phase 1: Initialize local histograms (used for collision tracking)
    atomicStore(&local_hist_r[local_idx], 0u);
    atomicStore(&local_hist_g[local_idx], 0u);
    atomicStore(&local_hist_b[local_idx], 0u);

    workgroupBarrier();

    // Phase 2: Track local bin collisions and store exact buckets
    var bucket_r: u32 = 0u;
    var bucket_g: u32 = 0u;
    var bucket_b: u32 = 0u;
    var has_pixel: bool = false;

    if (pixel_idx < pixel_count) {
        has_pixel = true;
        let idx = pixel_idx * 3u;
        let r = pixels[idx];
        let g = pixels[idx + 1u];
        let b = pixels[idx + 2u];

        bucket_r = value_to_bucket(r);
        bucket_g = value_to_bucket(g);
        bucket_b = value_to_bucket(b);

        // Track how many threads hit each local bin (for analysis)
        let local_bin_r = bucket_to_local_bin(bucket_r);
        let local_bin_g = bucket_to_local_bin(bucket_g);
        let local_bin_b = bucket_to_local_bin(bucket_b);

        atomicAdd(&local_hist_r[local_bin_r], 1u);
        atomicAdd(&local_hist_g[local_bin_g], 1u);
        atomicAdd(&local_hist_b[local_bin_b], 1u);
    }

    // Synchronize before global writes to batch operations
    workgroupBarrier();

    // Phase 3: Write to global with full precision
    // Batching through barrier provides some contention reduction
    if (has_pixel) {
        atomicAdd(&histogram_r[bucket_r], 1u);
        atomicAdd(&histogram_g[bucket_g], 1u);
        atomicAdd(&histogram_b[bucket_b], 1u);
    }
}

// Clear histogram buffers
// Uses the same bindings as accumulate but overwrites instead of incrementing
@compute @workgroup_size(256)
fn clear_histogram(@builtin(global_invocation_id) id: vec3<u32>) {
    let bucket_idx = id.x;

    if (bucket_idx >= NUM_BUCKETS) {
        return;
    }

    // Use atomicExchange to set values to 0 (atomicStore not available on storage buffers)
    _ = atomicExchange(&histogram_r[bucket_idx], 0u);
    _ = atomicExchange(&histogram_g[bucket_idx], 0u);
    _ = atomicExchange(&histogram_b[bucket_idx], 0u);
}
