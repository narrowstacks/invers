//! Data models for Invers
//!
//! Core data structures for film presets, scan profiles, and processing options.

mod base_estimation;
mod cb;
mod convert_options;
mod preset;
mod scan_profile;

// Re-export all public types to maintain the existing public API
pub use base_estimation::{BaseEstimation, BaseEstimationMethod, BaseSamplingMode};

pub use cb::{
    CbChannelOrigins, CbColorModel, CbColorModelParams, CbEnginePreset, CbEngineSettings,
    CbEnhancedProfile, CbFilmCharacter, CbHistogramAnalysis, CbLayerOrder, CbOptions, CbSourceType,
    CbToneProfile, CbToneProfileParams, CbWbMethod, CbWbPreset, CbWbTonality,
};

pub use convert_options::{
    AutoLevelsMode, AutoWbMode, BitDepthPolicy, ConvertOptions, DensityBalance,
    DensityBalanceSource, InversionMode, NeutralPointSample, OutputFormat, PipelineMode,
    ShadowLiftMode,
};

pub use preset::{FilmPreset, ToneCurveParams};

pub use scan_profile::{
    DemosaicHints, HslAdjustments, MaskProfile, ScanProfile, WhiteBalanceHints,
};
