# invers-raw

RAW file decoding using LibRaw (via rsraw bindings).

Isolated from invers-core to avoid rebuilding LibRaw when pipeline changes.

## API

- `decode_raw(path)` - Decode RAW file to linear f32 RGB
- `is_raw_extension(ext)` - Check if extension is supported RAW format
- `RAW_EXTENSIONS` - List of supported extensions (cr2, nef, arw, dng, etc.)

## LibRaw Configuration

Set in `decode_raw()` (`lib.rs:58`):

- AHD demosaic (`user_qual = 3`) - best quality for film scanning
- No auto brightness (`no_auto_bright = 1`)
- Camera white balance if available

## Output

`DecodedRaw` struct with:

- Linear RGB data (f32, 0.0-1.0)
- Width, height, channels
- Optional black/white levels and color matrix
