//! Parameter testing and optimization infrastructure
//!
//! Provides tools for testing different parameter combinations to optimize
//! conversion results against reference images.

mod from_impls;
mod grid_search;
mod runners;
mod scoring;
mod types;

// Re-export all public types
pub use types::{ParameterGrid, ParameterTest, PreloadedTestContext, TestResult};

// Re-export runner functions
pub use runners::{run_parameter_test, run_parameter_test_preloaded};

// Re-export grid search functions
pub use grid_search::{
    print_test_result, run_adaptive_grid_search, run_parameter_grid_search,
    run_parameter_grid_search_parallel,
};
