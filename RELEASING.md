# Releasing invers

This document describes how to create a new release and publish it to Homebrew.

## Prerequisites

1. Ensure all changes are merged to `main`
2. Update version in `Cargo.toml` workspace section
3. Ensure CI passes on `main`
4. Have the homebrew tap repo cloned at `~/GitHub/homebrew-invers`

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

### 4. Update Homebrew Formula

Once the release is published:

```bash
./scripts/update-formula.sh vX.Y.Z
```

This script will:
1. Fetch SHA256 checksums from the GitHub release
2. Update the formula in the homebrew-invers tap
3. Commit and push the changes

If your tap repo is in a different location, set the environment variable:

```bash
TAP_REPO_PATH=/path/to/homebrew-invers ./scripts/update-formula.sh vX.Y.Z
```

## Installing via Homebrew

Users can install invers with:

```bash
# Add the tap (first time only)
brew tap narrowstacks/invers

# Install
brew install invers

# Or install directly without adding tap
brew install narrowstacks/invers/invers
```

## Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR** (X.0.0): Breaking changes
- **MINOR** (0.X.0): New features, backwards compatible
- **PATCH** (0.0.X): Bug fixes, backwards compatible

Pre-release versions use suffixes: `v0.1.0-alpha`, `v0.1.0-beta`, `v0.1.0-rc1`

## Repository Structure

- **narrowstacks/invers** - Main project repository
- **narrowstacks/homebrew-invers** - Homebrew tap (contains only the formula)

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

### Formula update script fails

- Ensure the release exists and all artifacts were uploaded
- Check that the tap repo is cloned: `ls ~/GitHub/homebrew-invers`
- Verify you have push access to the tap repo

### Users report installation issues

1. Verify the formula: `brew audit --strict narrowstacks/invers/invers`
2. Test installation: `brew install --verbose narrowstacks/invers/invers`
3. Check SHA256 hashes match the release artifacts
