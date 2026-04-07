// Window/Level shader for DICOM images
// Applies rescale and windowing transformation on GPU

struct Uniforms {
    window_center: f32,
    window_width: f32,
    rescale_slope: f32,
    rescale_intercept: f32,
    // Transform for positioning
    scale: vec2<f32>,
    offset: vec2<f32>,
    // Pixel representation (0 = unsigned, 1 = signed)
    pixel_representation: u32,
    _padding: u32,  // Alignment padding
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(0) @binding(1)
var dicom_texture: texture_2d<u32>;

// Vertex shader: fullscreen quad
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Generate fullscreen quad from vertex index (0-5 for 2 triangles)
    // Triangle 1: 0, 1, 2
    // Triangle 2: 2, 1, 3
    var positions = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),  // bottom-left
        vec2<f32>(1.0, -1.0),   // bottom-right
        vec2<f32>(-1.0, 1.0),   // top-left
        vec2<f32>(1.0, 1.0),    // top-right
    );

    var uvs = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 1.0),  // bottom-left (flip Y for image coords)
        vec2<f32>(1.0, 1.0),  // bottom-right
        vec2<f32>(0.0, 0.0),  // top-left
        vec2<f32>(1.0, 0.0),  // top-right
    );

    // Index order for 2 triangles: 0, 1, 2, 2, 1, 3
    var indices = array<u32, 6>(0u, 1u, 2u, 2u, 1u, 3u);
    let idx = indices[vertex_index];

    var out: VertexOutput;
    // Apply scale and offset for zoom/pan
    let pos = positions[idx] * uniforms.scale + uniforms.offset;
    out.position = vec4<f32>(pos, 0.0, 1.0);
    out.uv = uvs[idx];
    return out;
}

// Fragment shader: apply window/level
@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    // DEBUG MODE: uncomment one of these to test
    // return vec4<f32>(uv.x, uv.y, 0.0, 1.0);  // UV gradient test
    // return vec4<f32>(1.0, 0.0, 0.0, 1.0);     // Solid red test

    // Get texture dimensions
    let tex_dims = textureDimensions(dicom_texture);

    // Convert UV to pixel coordinates
    let pixel_coords = vec2<i32>(uv * vec2<f32>(tex_dims));

    // Sample raw pixel value (u16 stored as u32)
    let raw = textureLoad(dicom_texture, pixel_coords, 0).r;

    // Handle signed pixel representation
    // If pixel_representation is 1, interpret u16 as i16
    var raw_value: f32;
    if (uniforms.pixel_representation == 1u && raw > 32767u) {
        // Signed: values > 32767 are negative (two's complement)
        raw_value = f32(i32(raw) - 65536);
    } else {
        raw_value = f32(raw);
    }

    // Apply rescale to get actual value (HU for CT)
    let value = raw_value * uniforms.rescale_slope + uniforms.rescale_intercept;

    // Apply window/level
    let min_val = uniforms.window_center - uniforms.window_width / 2.0;
    let normalized = clamp((value - min_val) / uniforms.window_width, 0.0, 1.0);

    // Output grayscale
    return vec4<f32>(normalized, normalized, normalized, 1.0);
}
