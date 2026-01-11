//! Benchmarks for invers-core pipeline operations
//!
//! Run with: cargo bench -p invers-core

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use invers_core::auto_adjust::{auto_color, auto_levels, auto_levels_no_clip, compress_highlights};
use invers_core::models::ToneCurveParams;
use invers_core::pipeline::{apply_color_matrix, apply_tone_curve};
use std::collections::HashMap;

/// Generate synthetic test image data (positive image)
fn generate_test_positive(width: u32, height: u32) -> Vec<f32> {
    let pixel_count = (width * height) as usize;
    let mut data = Vec::with_capacity(pixel_count * 3);

    for i in 0..pixel_count {
        let x = (i % width as usize) as f32 / width as f32;
        let y = (i / width as usize) as f32 / height as f32;

        // Positive values between 0.1 and 0.9
        data.push(0.1 + 0.8 * x);
        data.push(0.1 + 0.8 * y);
        data.push(0.1 + 0.8 * (x + y) / 2.0);
    }

    data
}

/// Benchmark tone curve application
fn bench_tone_curve(c: &mut Criterion) {
    let mut group = c.benchmark_group("tone_curves");

    let params = ToneCurveParams {
        curve_type: "asymmetric".to_string(),
        strength: 0.5,
        toe_strength: 0.3,
        shoulder_strength: 0.7,
        toe_length: 0.2,
        shoulder_start: 0.8,
        params: HashMap::new(),
    };

    for size in [256, 512, 1024, 2048].iter() {
        let width = *size;
        let height = *size;
        let pixel_count = (width * height) as u64;

        group.throughput(Throughput::Elements(pixel_count));

        group.bench_with_input(
            BenchmarkId::new("apply_tone_curve", format!("{}x{}", width, height)),
            &(width, height),
            |b, &(w, h)| {
                let mut data = generate_test_positive(w, h);
                b.iter(|| {
                    apply_tone_curve(black_box(&mut data), black_box(&params));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark color matrix application
fn bench_color_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("color_matrix");

    // Typical color correction matrix
    let matrix = [[1.2, -0.1, -0.1], [-0.1, 1.2, -0.1], [-0.1, -0.1, 1.2]];

    for size in [256, 512, 1024, 2048].iter() {
        let width = *size;
        let height = *size;
        let pixel_count = (width * height) as u64;

        group.throughput(Throughput::Elements(pixel_count));

        group.bench_with_input(
            BenchmarkId::new("apply", format!("{}x{}", width, height)),
            &(width, height),
            |b, &(w, h)| {
                let mut data = generate_test_positive(w, h);
                b.iter(|| {
                    apply_color_matrix(black_box(&mut data), black_box(&matrix), 3);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark auto-levels histogram stretching
fn bench_auto_levels(c: &mut Criterion) {
    let mut group = c.benchmark_group("auto_adjust");

    for size in [256, 512, 1024].iter() {
        let width = *size;
        let height = *size;
        let pixel_count = (width * height) as u64;

        group.throughput(Throughput::Elements(pixel_count));

        group.bench_with_input(
            BenchmarkId::new("auto_levels", format!("{}x{}", width, height)),
            &(width, height),
            |b, &(w, h)| {
                let mut data = generate_test_positive(w, h);
                b.iter(|| {
                    let _ = auto_levels(black_box(&mut data), 3, black_box(0.1));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("auto_levels_no_clip", format!("{}x{}", width, height)),
            &(width, height),
            |b, &(w, h)| {
                let mut data = generate_test_positive(w, h);
                b.iter(|| {
                    let _ = auto_levels_no_clip(black_box(&mut data), 3, black_box(0.1));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("auto_color", format!("{}x{}", width, height)),
            &(width, height),
            |b, &(w, h)| {
                let mut data = generate_test_positive(w, h);
                b.iter(|| {
                    let _ = auto_color(black_box(&mut data), 3, 1.0, 0.5, 2.0, 0.15);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark highlight compression
fn bench_highlight_compression(c: &mut Criterion) {
    let mut group = c.benchmark_group("highlights");

    for size in [512, 1024, 2048].iter() {
        let width = *size;
        let height = *size;
        let pixel_count = (width * height) as u64;

        group.throughput(Throughput::Elements(pixel_count));

        group.bench_with_input(
            BenchmarkId::new("compress", format!("{}x{}", width, height)),
            &(width, height),
            |b, &(w, h)| {
                let mut data = generate_test_positive(w, h);
                b.iter(|| {
                    compress_highlights(black_box(&mut data), 0.9, 0.5);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark combined pipeline operations (simulated workflow)
fn bench_pipeline_workflow(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_pipeline");

    let matrix = [
        [1.1, -0.05, -0.05],
        [-0.05, 1.1, -0.05],
        [-0.05, -0.05, 1.1],
    ];
    let tone_params = ToneCurveParams {
        curve_type: "asymmetric".to_string(),
        strength: 0.4,
        toe_strength: 0.3,
        shoulder_strength: 0.7,
        toe_length: 0.2,
        shoulder_start: 0.8,
        params: HashMap::new(),
    };

    for size in [512, 1024].iter() {
        let width = *size;
        let height = *size;
        let pixel_count = (width * height) as u64;

        group.throughput(Throughput::Elements(pixel_count));

        group.bench_with_input(
            BenchmarkId::new("adjust_matrix_tone", format!("{}x{}", width, height)),
            &(width, height),
            |b, &(w, h)| {
                b.iter(|| {
                    let mut data = generate_test_positive(w, h);

                    // Auto-levels
                    let _ = auto_levels(&mut data, 3, 0.1);

                    // Color matrix
                    apply_color_matrix(&mut data, &matrix, 3);

                    // Tone curve
                    apply_tone_curve(&mut data, &tone_params);

                    black_box(data)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_tone_curve,
    bench_color_matrix,
    bench_auto_levels,
    bench_highlight_compression,
    bench_pipeline_workflow,
);

criterion_main!(benches);
