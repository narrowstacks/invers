//! Parallelization helpers for image processing operations
//!
//! This module provides generic functions to abstract the common pattern of
//! conditionally executing parallel or sequential code based on data size.
//! These helpers reduce code duplication across the auto_adjust modules.

use rayon::prelude::*;

use super::PARALLEL_THRESHOLD;

/// Parallel fold/reduce over chunks with automatic threshold-based dispatch.
///
/// This function abstracts the common pattern:
/// ```ignore
/// if num_elements >= PARALLEL_THRESHOLD {
///     data.par_chunks_exact(chunk_size)
///         .fold(|| init(), |acc, chunk| fold_fn(acc, chunk))
///         .reduce(|| init(), |a, b| reduce_fn(a, b))
/// } else {
///     // sequential version
/// }
/// ```
///
/// # Type Parameters
/// * `T` - The element type in the slice
/// * `A` - The accumulator type (must be Send for parallel execution)
/// * `I` - The init function type
/// * `F` - The fold function type
/// * `R` - The reduce function type
///
/// # Arguments
/// * `data` - The slice to process
/// * `chunk_size` - Size of each chunk (e.g., 3 for RGB pixels)
/// * `init` - Function that creates a new accumulator
/// * `fold_fn` - Function that folds a chunk into the accumulator
/// * `reduce_fn` - Function that combines two accumulators
///
/// # Returns
/// The final reduced accumulator value
///
/// # Example
/// ```ignore
/// let (r_sum, g_sum, b_sum) = parallel_fold_reduce(
///     &data,
///     3,
///     || (0.0f64, 0.0f64, 0.0f64),
///     |acc, pixel| (acc.0 + pixel[0] as f64, acc.1 + pixel[1] as f64, acc.2 + pixel[2] as f64),
///     |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
/// );
/// ```
pub fn parallel_fold_reduce<T, A, I, F, R>(
    data: &[T],
    chunk_size: usize,
    init: I,
    fold_fn: F,
    reduce_fn: R,
) -> A
where
    T: Sync,
    A: Send + Clone,
    I: Fn() -> A + Sync,
    F: Fn(A, &[T]) -> A + Sync,
    R: Fn(A, A) -> A + Sync,
{
    let num_elements = data.len() / chunk_size;

    if num_elements >= PARALLEL_THRESHOLD {
        data.par_chunks_exact(chunk_size)
            .fold(&init, &fold_fn)
            .reduce(&init, &reduce_fn)
    } else {
        let mut acc = init();
        for chunk in data.chunks_exact(chunk_size) {
            acc = fold_fn(acc, chunk);
        }
        acc
    }
}

/// Parallel for-each over immutable chunks with automatic threshold-based dispatch.
///
/// This function abstracts the common pattern:
/// ```ignore
/// if num_elements >= PARALLEL_THRESHOLD {
///     data.par_chunks_exact(chunk_size).for_each(|chunk| { ... });
/// } else {
///     for chunk in data.chunks_exact(chunk_size) { ... }
/// }
/// ```
///
/// # Arguments
/// * `data` - The slice to process
/// * `chunk_size` - Size of each chunk (e.g., 3 for RGB pixels)
/// * `f` - Function to apply to each chunk
///
/// # Example
/// ```ignore
/// parallel_for_each_chunk(&data, 3, |pixel| {
///     println!("R={}, G={}, B={}", pixel[0], pixel[1], pixel[2]);
/// });
/// ```
#[allow(dead_code)]
pub fn parallel_for_each_chunk<T, F>(data: &[T], chunk_size: usize, f: F)
where
    T: Sync,
    F: Fn(&[T]) + Sync,
{
    let num_elements = data.len() / chunk_size;

    if num_elements >= PARALLEL_THRESHOLD {
        data.par_chunks_exact(chunk_size).for_each(&f);
    } else {
        for chunk in data.chunks_exact(chunk_size) {
            f(chunk);
        }
    }
}

/// Parallel for-each over mutable chunks with automatic threshold-based dispatch.
///
/// This function abstracts the common pattern:
/// ```ignore
/// if num_elements >= PARALLEL_THRESHOLD {
///     data.par_chunks_exact_mut(chunk_size).for_each(|chunk| { ... });
/// } else {
///     for chunk in data.chunks_exact_mut(chunk_size) { ... }
/// }
/// ```
///
/// # Arguments
/// * `data` - The mutable slice to process
/// * `chunk_size` - Size of each chunk (e.g., 3 for RGB pixels)
/// * `f` - Function to apply to each chunk (can mutate the chunk)
///
/// # Example
/// ```ignore
/// parallel_for_each_chunk_mut(&mut data, 3, |pixel| {
///     pixel[0] *= r_gain;
///     pixel[1] *= g_gain;
///     pixel[2] *= b_gain;
/// });
/// ```
pub fn parallel_for_each_chunk_mut<T, F>(data: &mut [T], chunk_size: usize, f: F)
where
    T: Send + Sync,
    F: Fn(&mut [T]) + Sync,
{
    let num_elements = data.len() / chunk_size;

    if num_elements >= PARALLEL_THRESHOLD {
        data.par_chunks_exact_mut(chunk_size).for_each(&f);
    } else {
        for chunk in data.chunks_exact_mut(chunk_size) {
            f(chunk);
        }
    }
}

/// Parallel iteration over mutable elements with automatic threshold-based dispatch.
///
/// This function abstracts the common pattern:
/// ```ignore
/// if num_elements >= PARALLEL_THRESHOLD {
///     data.par_iter_mut().for_each(|value| { ... });
/// } else {
///     for value in data.iter_mut() { ... }
/// }
/// ```
///
/// # Arguments
/// * `data` - The mutable slice to process
/// * `threshold` - Minimum number of elements for parallel execution
/// * `f` - Function to apply to each element
///
/// # Example
/// ```ignore
/// parallel_for_each_mut(&mut data, 90_000, |value| {
///     *value *= scale;
/// });
/// ```
#[allow(dead_code)]
pub fn parallel_for_each_mut<T, F>(data: &mut [T], threshold: usize, f: F)
where
    T: Send + Sync,
    F: Fn(&mut T) + Sync,
{
    if data.len() >= threshold {
        data.par_iter_mut().for_each(&f);
    } else {
        for value in data.iter_mut() {
            f(value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_fold_reduce_small() {
        // Small dataset - should use sequential path
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let (r_sum, g_sum, b_sum) = parallel_fold_reduce(
            &data,
            3,
            || (0.0f64, 0.0f64, 0.0f64),
            |acc, pixel| {
                (
                    acc.0 + pixel[0] as f64,
                    acc.1 + pixel[1] as f64,
                    acc.2 + pixel[2] as f64,
                )
            },
            |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
        );

        assert!((r_sum - 5.0).abs() < 0.001); // 1 + 4
        assert!((g_sum - 7.0).abs() < 0.001); // 2 + 5
        assert!((b_sum - 9.0).abs() < 0.001); // 3 + 6
    }

    #[test]
    fn test_parallel_fold_reduce_large() {
        // Large dataset - should use parallel path
        let num_pixels = PARALLEL_THRESHOLD + 1000;
        let mut data: Vec<f32> = Vec::with_capacity(num_pixels * 3);
        for i in 0..num_pixels {
            let v = (i as f32) / (num_pixels as f32);
            data.push(v); // R
            data.push(v * 0.5); // G
            data.push(v * 0.25); // B
        }

        let (r_sum, g_sum, b_sum) = parallel_fold_reduce(
            &data,
            3,
            || (0.0f64, 0.0f64, 0.0f64),
            |acc, pixel| {
                (
                    acc.0 + pixel[0] as f64,
                    acc.1 + pixel[1] as f64,
                    acc.2 + pixel[2] as f64,
                )
            },
            |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
        );

        // Verify ratios are correct (g_sum should be ~0.5 * r_sum, b_sum ~0.25 * r_sum)
        assert!(
            (g_sum / r_sum - 0.5).abs() < 0.01,
            "g_sum/r_sum = {}",
            g_sum / r_sum
        );
        assert!(
            (b_sum / r_sum - 0.25).abs() < 0.01,
            "b_sum/r_sum = {}",
            b_sum / r_sum
        );
    }

    #[test]
    fn test_parallel_for_each_chunk_mut_small() {
        let mut data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let gains = [2.0f32, 0.5, 1.5];

        parallel_for_each_chunk_mut(&mut data, 3, |pixel| {
            pixel[0] *= gains[0];
            pixel[1] *= gains[1];
            pixel[2] *= gains[2];
        });

        assert!((data[0] - 2.0).abs() < 0.001);
        assert!((data[1] - 1.0).abs() < 0.001);
        assert!((data[2] - 4.5).abs() < 0.001);
        assert!((data[3] - 8.0).abs() < 0.001);
        assert!((data[4] - 2.5).abs() < 0.001);
        assert!((data[5] - 9.0).abs() < 0.001);
    }

    #[test]
    fn test_parallel_for_each_chunk_mut_large() {
        // Large dataset - should use parallel path
        let num_pixels = PARALLEL_THRESHOLD + 1000;
        let mut data: Vec<f32> = vec![1.0; num_pixels * 3];
        let scale = 2.0f32;

        parallel_for_each_chunk_mut(&mut data, 3, |pixel| {
            pixel[0] *= scale;
            pixel[1] *= scale;
            pixel[2] *= scale;
        });

        // All values should be 2.0
        assert!(data.iter().all(|&v| (v - 2.0).abs() < 0.001));
    }

    #[test]
    fn test_parallel_for_each_mut() {
        let mut data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let scale = 0.5f32;

        parallel_for_each_mut(&mut data, 10, |value| {
            *value *= scale;
        });

        assert!((data[0] - 0.5).abs() < 0.001);
        assert!((data[1] - 1.0).abs() < 0.001);
        assert!((data[2] - 1.5).abs() < 0.001);
        assert!((data[3] - 2.0).abs() < 0.001);
    }
}
