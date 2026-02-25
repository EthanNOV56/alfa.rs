//! Benchmark time series operations

use alpha_expr::timeseries::TimeSeries;
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

fn create_large_series(size: usize) -> TimeSeries {
    let data: Vec<f64> = (0..size).map(|i| (i as f64).sin() * 100.0).collect();
    TimeSeries::new(data)
}

fn bench_timeseries_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("timeseries_creation");
    
    for &size in &[100, 1000, 10000, 100000] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let data: Vec<f64> = (0..size).map(|i| (i as f64).sin() * 100.0).collect();
                let ts = TimeSeries::new(black_box(data));
                black_box(ts);
            });
        });
    }
    
    group.finish();
}

fn bench_lag_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("lag_operations");
    
    for &size in &[1000, 10000] {
        let ts = create_large_series(size);
        
        for &periods in &[1, 5, 20, 100] {
            group.bench_with_input(
                BenchmarkId::new(format!("lag_{}_periods", periods), size),
                &size,
                |b, _| {
                    b.iter(|| {
                        let result = ts.lag(black_box(periods));
                        black_box(result);
                    });
                }
            );
        }
    }
    
    group.finish();
}

fn bench_moving_average(c: &mut Criterion) {
    let mut group = c.benchmark_group("moving_average");
    
    for &size in &[1000, 10000] {
        let ts = create_large_series(size);
        
        for &window in &[10, 20, 50, 100, 200] {
            group.bench_with_input(
                BenchmarkId::new(format!("ma_window_{}", window), size),
                &size,
                |b, _| {
                    b.iter(|| {
                        let result = ts.moving_average(black_box(window));
                        black_box(result);
                    });
                }
            );
        }
    }
    
    group.finish();
}

fn bench_exponential_moving_average(c: &mut Criterion) {
    let mut group = c.benchmark_group("exponential_moving_average");
    
    for &size in &[1000, 10000] {
        let ts = create_large_series(size);
        
        for &span in &[10, 20, 30, 50, 100] {
            group.bench_with_input(
                BenchmarkId::new(format!("ema_span_{}", span), size),
                &size,
                |b, _| {
                    b.iter(|| {
                        let result = ts.exponential_moving_average(black_box(span));
                        black_box(result);
                    });
                }
            );
        }
    }
    
    group.finish();
}

fn bench_rolling_std(c: &mut Criterion) {
    let mut group = c.benchmark_group("rolling_std");
    
    for &size in &[1000, 10000] {
        let ts = create_large_series(size);
        
        for &window in &[10, 20, 50, 100] {
            group.bench_with_input(
                BenchmarkId::new(format!("std_window_{}", window), size),
                &size,
                |b, _| {
                    b.iter(|| {
                        let result = ts.rolling_std(black_box(window));
                        black_box(result);
                    });
                }
            );
        }
    }
    
    group.finish();
}

fn bench_correlation(c: &mut Criterion) {
    let mut group = c.benchmark_group("correlation");
    
    for &size in &[1000, 5000] {
        let ts1 = create_large_series(size);
        let ts2 = TimeSeries::new(
            (0..size).map(|i| (i as f64).cos() * 80.0 + 20.0).collect()
        );
        
        for &window in &[20, 50, 100, 200] {
            group.bench_with_input(
                BenchmarkId::new(format!("corr_window_{}", window), size),
                &size,
                |b, _| {
                    b.iter(|| {
                        let result = ts1.correlation(&ts2, black_box(window));
                        black_box(result);
                    });
                }
            );
        }
    }
    
    group.finish();
}

fn bench_alpha_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("alpha_computation");
    
    for &size in &[1000, 5000] {
        let returns = TimeSeries::new(
            (0..size).map(|i| ((i as f64) * 0.01).sin() * 0.05).collect()
        );
        let market_returns = TimeSeries::new(
            (0..size).map(|i| ((i as f64) * 0.008).cos() * 0.04).collect()
        );
        
        group.bench_with_input(BenchmarkId::new("sharpe_ratio", size), &size, |b, _| {
            b.iter(|| {
                let result = returns.sharpe_ratio(0.02, black_box(20));
                black_box(result);
            });
        });
        
        group.bench_with_input(BenchmarkId::new("beta", size), &size, |b, _| {
            b.iter(|| {
                let result = returns.beta(&market_returns, black_box(20));
                black_box(result);
            });
        });
        
        group.bench_with_input(BenchmarkId::new("max_drawdown", size), &size, |b, _| {
            b.iter(|| {
                let result = returns.max_drawdown(black_box(50));
                black_box(result);
            });
        });
    }
    
    group.finish();
}

fn bench_batch_operations(c: &mut Criterion) {
    let ts = create_large_series(10000);
    
    c.bench_function("batch_window_operations", |b| {
        b.iter(|| {
            let ma = ts.moving_average(black_box(20));
            let ema = ts.exponential_moving_average(black_box(30));
            let std = ts.rolling_std(black_box(20));
            let _ = (ma, ema, std);
        });
    });
    
    c.bench_function("pct_change_and_diff", |b| {
        b.iter(|| {
            let pct = ts.pct_change(black_box(1));
            let diff = ts.diff(black_box(1));
            let _ = (pct, diff);
        });
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .sample_size(30)
        .warm_up_time(std::time::Duration::from_secs(1))
        .measurement_time(std::time::Duration::from_secs(2));
    targets = bench_timeseries_creation,
              bench_lag_operations,
              bench_moving_average,
              bench_exponential_moving_average,
              bench_rolling_std,
              bench_correlation,
              bench_alpha_computation,
              bench_batch_operations
);

criterion_main!(benches);