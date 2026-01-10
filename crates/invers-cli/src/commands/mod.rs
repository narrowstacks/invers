//! Command implementations for the invers CLI.

mod analyze;
mod batch;
mod convert;
mod init;
mod preset;

#[cfg(debug_assertions)]
mod debug;

// Re-export all command functions
pub use analyze::cmd_analyze;
pub use batch::cmd_batch;
pub use convert::cmd_convert;
pub use init::cmd_init;
pub use preset::{cmd_preset_create, cmd_preset_list, cmd_preset_show};

#[cfg(debug_assertions)]
pub use debug::{cmd_diagnose, cmd_test_params};
