//! Colormap / LUT definitions for image fusion
//!
//! Colormaps are used for color overlay mode (PET/CT, perfusion).
//! The actual rendering uses inline WGSL functions in fusion.wgsl,
//! but these CPU-side definitions are kept for testing and potential
//! texture-based LUT upload in the future.
#![allow(dead_code)]

/// A single colormap entry (RGBA, 0-255)
#[derive(Debug, Clone, Copy)]
pub struct ColormapEntry {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl ColormapEntry {
    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }
}

/// Colormap type for overlay rendering (matches fusion/mod.rs ColormapType)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColormapType {
    /// Grayscale (default, same as base windowing)
    #[default]
    Grayscale,
    /// Hot colormap (black → red → yellow → white) — PET standard
    Hot,
    /// Rainbow (blue → cyan → green → yellow → red)
    Rainbow,
    /// Cool-warm diverging colormap (blue → white → red)
    CoolWarm,
}

impl ColormapType {
    /// Display name
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Grayscale => "Grayscale",
            Self::Hot => "Hot",
            Self::Rainbow => "Rainbow",
            Self::CoolWarm => "Cool-Warm",
        }
    }

    /// Shader index for uniform
    pub fn shader_index(&self) -> u32 {
        match self {
            Self::Grayscale => 0,
            Self::Hot => 1,
            Self::Rainbow => 2,
            Self::CoolWarm => 3,
        }
    }
}

/// Generate a 256-entry LUT for a given colormap type.
/// Returns Vec of [r, g, b, a] u8 values (1024 bytes total).
pub fn generate_lut(colormap: ColormapType) -> Vec<u8> {
    let mut lut = Vec::with_capacity(256 * 4);
    for i in 0..256u32 {
        let t = i as f32 / 255.0; // normalized 0..1
        let entry = match colormap {
            ColormapType::Grayscale => ColormapEntry::new(i as u8, i as u8, i as u8, 255),
            ColormapType::Hot => hot_colormap(t),
            ColormapType::Rainbow => rainbow_colormap(t),
            ColormapType::CoolWarm => cool_warm_colormap(t),
        };
        lut.push(entry.r);
        lut.push(entry.g);
        lut.push(entry.b);
        lut.push(entry.a);
    }
    lut
}

/// Hot colormap: black → red → yellow → white
/// Standard for PET imaging
fn hot_colormap(t: f32) -> ColormapEntry {
    let r;
    let g;
    let b;

    if t < 0.333 {
        // Black → Red
        r = (t / 0.333 * 255.0) as u8;
        g = 0;
        b = 0;
    } else if t < 0.667 {
        // Red → Yellow
        r = 255;
        g = ((t - 0.333) / 0.333 * 255.0) as u8;
        b = 0;
    } else {
        // Yellow → White
        r = 255;
        g = 255;
        b = ((t - 0.667) / 0.333 * 255.0) as u8;
    }

    ColormapEntry::new(r, g, b, 255)
}

/// Rainbow colormap: blue → cyan → green → yellow → red
fn rainbow_colormap(t: f32) -> ColormapEntry {
    let r;
    let g;
    let b;

    if t < 0.25 {
        // Blue → Cyan
        let s = t / 0.25;
        r = 0;
        g = (s * 255.0) as u8;
        b = 255;
    } else if t < 0.5 {
        // Cyan → Green
        let s = (t - 0.25) / 0.25;
        r = 0;
        g = 255;
        b = ((1.0 - s) * 255.0) as u8;
    } else if t < 0.75 {
        // Green → Yellow
        let s = (t - 0.5) / 0.25;
        r = (s * 255.0) as u8;
        g = 255;
        b = 0;
    } else {
        // Yellow → Red
        let s = (t - 0.75) / 0.25;
        r = 255;
        g = ((1.0 - s) * 255.0) as u8;
        b = 0;
    }

    ColormapEntry::new(r, g, b, 255)
}

/// Cool-warm diverging colormap: blue → white → red
/// Useful for difference images
fn cool_warm_colormap(t: f32) -> ColormapEntry {
    let r;
    let g;
    let b;

    if t < 0.5 {
        // Blue → White
        let s = t * 2.0;
        r = (s * 255.0) as u8;
        g = (s * 255.0) as u8;
        b = 255;
    } else {
        // White → Red
        let s = (t - 0.5) * 2.0;
        r = 255;
        g = ((1.0 - s) * 255.0) as u8;
        b = ((1.0 - s) * 255.0) as u8;
    }

    ColormapEntry::new(r, g, b, 255)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lut_size() {
        for cm in [
            ColormapType::Hot,
            ColormapType::Rainbow,
            ColormapType::CoolWarm,
            ColormapType::Grayscale,
        ] {
            let lut = generate_lut(cm);
            assert_eq!(lut.len(), 256 * 4, "LUT size wrong for {:?}", cm);
        }
    }

    #[test]
    fn test_hot_endpoints() {
        let black = hot_colormap(0.0);
        assert_eq!(black.r, 0);
        assert_eq!(black.g, 0);
        assert_eq!(black.b, 0);

        let white = hot_colormap(1.0);
        assert_eq!(white.r, 255);
        assert_eq!(white.g, 255);
        // Close to 255 (may be 254 due to float precision)
        assert!(white.b >= 253);
    }

    #[test]
    fn test_rainbow_endpoints() {
        let blue = rainbow_colormap(0.0);
        assert_eq!(blue.r, 0);
        assert_eq!(blue.b, 255);

        let red = rainbow_colormap(1.0);
        assert_eq!(red.r, 255);
        assert_eq!(red.g, 0);
        assert_eq!(red.b, 0);
    }

    #[test]
    fn test_grayscale_lut() {
        let lut = generate_lut(ColormapType::Grayscale);
        // First entry should be black
        assert_eq!(lut[0], 0);
        assert_eq!(lut[1], 0);
        assert_eq!(lut[2], 0);
        // Last entry should be white
        assert_eq!(lut[255 * 4], 255);
        assert_eq!(lut[255 * 4 + 1], 255);
        assert_eq!(lut[255 * 4 + 2], 255);
    }
}
