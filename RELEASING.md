# Releasing invers

This document describes how to create a new release and publish it to Homebrew.

## Prerequisites

1. Ensure all changes are merged to `main`
2. Update version in `Cargo.toml` workspace section
3. Ensure CI passes on `main`

## Creating a Release

### 1. Update Version

Edit `Cargo.toml` at the workspace root:

```toml
[workspace.package]
version = "X.Y.Z"  # Update this
```

Commit the version bump:

```bash
git add Cargo.toml
git commit -m "chore: bump version to X.Y.Z"
git push origin main
```

### 2. Create and Push Tag

```bash
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin vX.Y.Z
```

This triggers the release workflow which:
- Builds binaries for macOS (Intel + Apple Silicon) and Linux (x86_64)
- Creates a GitHub release with the binaries
- Generates SHA256 checksums for each binary

### 3. Wait for Release Workflow

Monitor the [Actions tab](https://github.com/narrowstacks/invers/actions) to ensure the release workflow completes successfully.

## Publishing to Homebrew

### Initial Setup (One-time)

1. Create a new repository: `narrowstacks/homebrew-invers`
2. Copy `Formula/invers.rb` to the new repo

### Updating the Formula

After the GitHub release is created:

```bash
# Update the formula with correct SHA256 hashes
./scripts/update-formula.sh vX.Y.Z

# Review the changes
cat Formula/invers.rb
```

Then copy the updated formula to your homebrew tap:

```bash
cp Formula/invers.rb /path/to/homebrew-invers/Formula/
cd /path/to/homebrew-invers
git add Formula/invers.rb
git commit -m "Update invers to X.Y.Z"
git push
```

### Installing via Homebrew

Users can install invers with:

```bash
# Add the tap (first time only)
brew tap narrowstacks/invers

# Install
brew install invers

# Or install directly
brew install narrowstacks/invers/invers
```

## Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR** (X.0.0): Breaking changes
- **MINOR** (0.X.0): New features, backwards compatible
- **PATCH** (0.0.X): Bug fixes, backwards compatible

Pre-release versions use suffixes: `v0.1.0-alpha`, `v0.1.0-beta`, `v0.1.0-rc1`

## Troubleshooting

### Release workflow failed

1. Check the [Actions tab](https://github.com/narrowstacks/invers/actions) for error logs
2. Common issues:
   - Compilation errors on specific platforms
   - Missing dependencies
3. Fix the issue, delete the tag, and try again:
   ```bash
   git tag -d vX.Y.Z
   git push origin :refs/tags/vX.Y.Z
   ```

### Homebrew formula not working

Test the formula locally:

```bash
brew install --build-from-source Formula/invers.rb
```

Check for:
- Incorrect SHA256 hashes
- Missing dependencies
- Binary name mismatch
