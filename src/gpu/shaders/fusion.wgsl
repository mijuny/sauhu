// Fusion shader for blending two DICOM images
//
// Takes two u16 textures (base and overlay), applies windowing separately,
// then blends them according to the selected blend mode.
//
// Phase 1: Alpha blend
// Phase 2: Color overlay with inline colormaps (Hot, Rainbow, Cool-Warm)
// Phase 3: Checkerboard with adjustable tile size

struct FusionUniforms {
    // Blend parameters
    opacity: f32,           // Overlay opacity 0.0–1.0
    blend_mode: u32,        // 0=Alpha, 1=ColorOverlay, 2=Checkerboard

    // Base image windowing
    base_wc: f32,
    base_ww: f32,
    base_slope: f32,
    base_intercept: f32,
    base_pixel_rep: u32,

    // Overlay image windowing
    overlay_wc: f32,
    overlay_ww: f32,
    overlay_slope: f32,
    overlay_intercept: f32,
    overlay_pixel_rep: u32,

    // Transform (zoom/pan) — shared for both images (same geometry after coregistration)
    scale: vec2<f32>,
    offset: vec2<f32>,

    // Phase 2+3 extras
    colormap_index: u32,    // 0=Grayscale, 1=Hot, 2=Rainbow, 3=CoolWarm
    checker_size: f32,      // Checkerboard square size in pixels (Phase 3)
    threshold: f32,         // Overlay threshold 0.0–1.0 (below = transparent)

    _padding: f32,          // Alignment padding to 16-byte boundary
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@group(0) @binding(0)
var<uniform> params: FusionUniforms;

@group(0) @binding(1)
var base_texture: texture_2d<u32>;

@group(0) @binding(2)
var overlay_texture: texture_2d<u32>;

// Vertex shader: fullscreen quad (same as windowing.wgsl)
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, 1.0),
    );

    var uvs = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
    );

    var indices = array<u32, 6>(0u, 1u, 2u, 2u, 1u, 3u);
    let idx = indices[vertex_index];

    var out: VertexOutput;
    let pos = positions[idx] * params.scale + params.offset;
    out.position = vec4<f32>(pos, 0.0, 1.0);
    out.uv = uvs[idx];
    return out;
}

// Apply DICOM windowing to a raw pixel value
fn apply_windowing(raw: u32, pixel_rep: u32, slope: f32, intercept: f32, wc: f32, ww: f32) -> f32 {
    var raw_value: f32;
    if (pixel_rep == 1u && raw > 32767u) {
        raw_value = f32(i32(raw) - 65536);
    } else {
        raw_value = f32(raw);
    }
    let value = raw_value * slope + intercept;
    let min_val = wc - ww / 2.0;
    return clamp((value - min_val) / ww, 0.0, 1.0);
}

// ============================================================
// Inline colormaps — no texture LUT needed
// ============================================================

/// Hot colormap: black → red → yellow → white (PET standard)
fn colormap_hot(t: f32) -> vec3<f32> {
    var r: f32;
    var g: f32;
    var b: f32;

    if (t < 0.333) {
        r = t / 0.333;
        g = 0.0;
        b = 0.0;
    } else if (t < 0.667) {
        r = 1.0;
        g = (t - 0.333) / 0.334;
        b = 0.0;
    } else {
        r = 1.0;
        g = 1.0;
        b = (t - 0.667) / 0.333;
    }

    return vec3<f32>(r, g, clamp(b, 0.0, 1.0));
}

/// Rainbow colormap: blue → cyan → green → yellow → red
fn colormap_rainbow(t: f32) -> vec3<f32> {
    var r: f32;
    var g: f32;
    var b: f32;

    if (t < 0.25) {
        // Blue → Cyan
        let s = t / 0.25;
        r = 0.0;
        g = s;
        b = 1.0;
    } else if (t < 0.5) {
        // Cyan → Green
        let s = (t - 0.25) / 0.25;
        r = 0.0;
        g = 1.0;
        b = 1.0 - s;
    } else if (t < 0.75) {
        // Green → Yellow
        let s = (t - 0.5) / 0.25;
        r = s;
        g = 1.0;
        b = 0.0;
    } else {
        // Yellow → Red
        let s = (t - 0.75) / 0.25;
        r = 1.0;
        g = 1.0 - s;
        b = 0.0;
    }

    return vec3<f32>(r, g, b);
}

/// Cool-warm diverging colormap: blue → white → red (difference images)
fn colormap_cool_warm(t: f32) -> vec3<f32> {
    var r: f32;
    var g: f32;
    var b: f32;

    if (t < 0.5) {
        // Blue → White
        let s = t * 2.0;
        r = s;
        g = s;
        b = 1.0;
    } else {
        // White → Red
        let s = (t - 0.5) * 2.0;
        r = 1.0;
        g = 1.0 - s;
        b = 1.0 - s;
    }

    return vec3<f32>(r, g, b);
}

/// Apply selected colormap to a normalized intensity value
fn apply_colormap(t: f32, colormap_idx: u32) -> vec3<f32> {
    switch colormap_idx {
        // 0 = Grayscale
        case 0u: {
            return vec3<f32>(t, t, t);
        }
        // 1 = Hot
        case 1u: {
            return colormap_hot(t);
        }
        // 2 = Rainbow
        case 2u: {
            return colormap_rainbow(t);
        }
        // 3 = Cool-Warm
        case 3u: {
            return colormap_cool_warm(t);
        }
        default: {
            return vec3<f32>(t, t, t);
        }
    }
}

// Fragment shader: blend base and overlay
@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    // Sample base texture
    let base_dims = textureDimensions(base_texture);
    let base_coords = vec2<i32>(uv * vec2<f32>(base_dims));
    let base_raw = textureLoad(base_texture, base_coords, 0).r;
    let base_val = apply_windowing(
        base_raw,
        params.base_pixel_rep,
        params.base_slope,
        params.base_intercept,
        params.base_wc,
        params.base_ww
    );

    // Sample overlay texture
    let overlay_dims = textureDimensions(overlay_texture);
    let overlay_coords = vec2<i32>(uv * vec2<f32>(overlay_dims));
    let overlay_raw = textureLoad(overlay_texture, overlay_coords, 0).r;
    let overlay_val = apply_windowing(
        overlay_raw,
        params.overlay_pixel_rep,
        params.overlay_slope,
        params.overlay_intercept,
        params.overlay_wc,
        params.overlay_ww
    );

    // Base as grayscale
    let base_color = vec3<f32>(base_val, base_val, base_val);

    // Blend based on mode
    var final_color: vec3<f32>;

    switch params.blend_mode {
        // Alpha blend — grayscale mix
        case 0u: {
            let overlay_color = vec3<f32>(overlay_val, overlay_val, overlay_val);
            final_color = mix(base_color, overlay_color, params.opacity);
        }
        // Color Overlay — overlay with colormap + threshold
        case 1u: {
            // Apply threshold: if overlay intensity is below threshold, show only base
            if (overlay_val < params.threshold) {
                final_color = base_color;
            } else {
                // Map overlay intensity to colormap color
                let colored_overlay = apply_colormap(overlay_val, params.colormap_index);
                // Blend colormap overlay onto grayscale base
                final_color = mix(base_color, colored_overlay, params.opacity);
            }
        }
        // Checkerboard — alternating tiles from base and overlay
        case 2u: {
            // Pixel coordinates in the image
            let px = uv * vec2<f32>(base_dims);
            // Tile indices (floor division by checker_size)
            let tile_x = i32(floor(px.x / params.checker_size));
            let tile_y = i32(floor(px.y / params.checker_size));
            let checker = (tile_x + tile_y) % 2;

            let overlay_color = vec3<f32>(overlay_val, overlay_val, overlay_val);
            if (checker == 0) {
                final_color = base_color;
            } else {
                final_color = overlay_color;
            }
        }
        default: {
            final_color = base_color;
        }
    }

    return vec4<f32>(final_color, 1.0);
}
