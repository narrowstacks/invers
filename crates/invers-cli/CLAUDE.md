# invers-cli

Command-line interface for the Invers film converter.

## Commands

- `convert` - Convert a single negative to positive
- `batch` - Process multiple images with shared settings
- `analyze` - Analyze image to estimate film base color
- `init` - Initialize user config directory (~/.invers/)
- `completions` - Generate shell completions

Debug-only (debug builds): `diagnose`, `test-params`

## Structure

- `main.rs` - CLI argument parsing (clap derive)
- `lib.rs` - Shared utilities, option builders, parsing helpers
- `commands/` - Command implementations

## Key Functions

- `process_single_image()` (`lib.rs:181`) - Core processing orchestration
- `build_convert_options_full_with_gpu()` (`lib.rs:774`) - ConvertOptions builder
- `parse_*` functions - CLI argument parsing helpers

## Adding a Command

1. Add subcommand variant to `Commands` enum in `main.rs`
2. Create handler function in `commands/`
3. Wire up in `main()` match statement
