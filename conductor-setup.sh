#!/bin/bash
set -e

echo "🔧 Setting up Invers workspace..."

# Check if Rust/Cargo is installed
if ! command -v cargo &> /dev/null; then
    echo "❌ Error: Cargo is not installed. Please install Rust from https://rustup.rs/"
    exit 1
fi

# Display Rust version
echo "✓ Cargo found: $(cargo --version)"

# Build the project (this will download dependencies and compile)
echo "📦 Building Invers workspace..."
cargo build

# Run a quick check to verify everything compiles
echo "🔍 Running cargo check..."
cargo check

echo "✅ Workspace setup complete!"
echo ""
echo "You can now:"
echo "  - Build release version: cargo build --release"
echo "  - Run CLI: cargo run -p invers-cli -- --help"
echo "  - Run tests: cargo test"
echo "  - Format code: cargo fmt"
echo "  - Lint code: cargo clippy"
