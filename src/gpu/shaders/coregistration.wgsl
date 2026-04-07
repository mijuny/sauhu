// Coregistration compute shader
// Performs volume resampling and NCC metric calculation on GPU
//
// CRITICAL: Transform operates in PHYSICAL (mm) coordinates
// This ensures transform parameters are independent of resolution level

// Uniforms for transform and dimensions
struct TransformUniforms {
    // 4x4 inverse transform matrix (column-major) - operates in mm coordinates
    transform: mat4x4<f32>,
    // Target volume dimensions (width, height, depth, _padding)
    target_dims: vec4<u32>,
    // Source volume dimensions (width, height, depth, _padding)
    source_dims: vec4<u32>,
    // Target voxel spacing in mm (x, y, z, _padding)
    target_spacing: vec4<f32>,
    // Source voxel spacing in mm (x, y, z, _padding)
    source_spacing: vec4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: TransformUniforms;
@group(0) @binding(1) var<storage, read> source_volume: array<f32>;
@group(0) @binding(2) var<storage, read_write> resampled_volume: array<f32>;

// Trilinear interpolation sample from source volume (in source voxel coords)
fn trilinear_sample(pos: vec3<f32>) -> f32 {
    let w = f32(uniforms.source_dims.x);
    let h = f32(uniforms.source_dims.y);
    let d = f32(uniforms.source_dims.z);

    // Check bounds
    if (pos.x < 0.0 || pos.y < 0.0 || pos.z < 0.0 ||
        pos.x >= w || pos.y >= h || pos.z >= d) {
        return 0.0;
    }

    let x0 = u32(floor(pos.x));
    let y0 = u32(floor(pos.y));
    let z0 = u32(floor(pos.z));

    let x1 = min(x0 + 1u, uniforms.source_dims.x - 1u);
    let y1 = min(y0 + 1u, uniforms.source_dims.y - 1u);
    let z1 = min(z0 + 1u, uniforms.source_dims.z - 1u);

    let xf = pos.x - f32(x0);
    let yf = pos.y - f32(y0);
    let zf = pos.z - f32(z0);

    let stride_y = uniforms.source_dims.x;
    let stride_z = uniforms.source_dims.x * uniforms.source_dims.y;

    // Sample 8 corners
    let c000 = source_volume[x0 + y0 * stride_y + z0 * stride_z];
    let c100 = source_volume[x1 + y0 * stride_y + z0 * stride_z];
    let c010 = source_volume[x0 + y1 * stride_y + z0 * stride_z];
    let c110 = source_volume[x1 + y1 * stride_y + z0 * stride_z];
    let c001 = source_volume[x0 + y0 * stride_y + z1 * stride_z];
    let c101 = source_volume[x1 + y0 * stride_y + z1 * stride_z];
    let c011 = source_volume[x0 + y1 * stride_y + z1 * stride_z];
    let c111 = source_volume[x1 + y1 * stride_y + z1 * stride_z];

    // Trilinear interpolation
    let c00 = c000 * (1.0 - xf) + c100 * xf;
    let c10 = c010 * (1.0 - xf) + c110 * xf;
    let c01 = c001 * (1.0 - xf) + c101 * xf;
    let c11 = c011 * (1.0 - xf) + c111 * xf;

    let c0 = c00 * (1.0 - yf) + c10 * yf;
    let c1 = c01 * (1.0 - yf) + c11 * yf;

    return c0 * (1.0 - zf) + c1 * zf;
}

// Resample source volume with transform
// Using 4x4x4 = 64 invocations to fit within Intel Arc's 256 limit
//
// The transform operates in PHYSICAL (mm) coordinates:
// 1. Convert target voxel to physical coords using target_spacing
// 2. Apply inverse transform (in mm space)
// 3. Convert result to source voxel coords using source_spacing
@compute @workgroup_size(4, 4, 4)
fn resample_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;

    // Check bounds against target dimensions
    if (x >= uniforms.target_dims.x || y >= uniforms.target_dims.y || z >= uniforms.target_dims.z) {
        return;
    }

    // Convert target voxel center to physical coordinates (mm)
    let target_physical = vec4<f32>(
        (f32(x) + 0.5) * uniforms.target_spacing.x,
        (f32(y) + 0.5) * uniforms.target_spacing.y,
        (f32(z) + 0.5) * uniforms.target_spacing.z,
        1.0
    );

    // Apply inverse transform in physical space
    let source_physical = uniforms.transform * target_physical;

    // Convert physical coordinates to source voxel coordinates
    let source_voxel = vec3<f32>(
        source_physical.x / uniforms.source_spacing.x,
        source_physical.y / uniforms.source_spacing.y,
        source_physical.z / uniforms.source_spacing.z
    );

    // Sample with trilinear interpolation
    let value = trilinear_sample(source_voxel);

    // Write to resampled volume (using target dimensions for indexing)
    let idx = x + y * uniforms.target_dims.x + z * uniforms.target_dims.x * uniforms.target_dims.y;
    resampled_volume[idx] = value;
}

// ============== NCC METRIC CALCULATION ==============

struct NccUniforms {
    // Volume dimensions
    dims: vec4<u32>,
    // Number of voxels
    num_voxels: u32,
    // Explicit padding to avoid vec3 alignment issues
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

// Partial sums for NCC reduction
struct NccPartials {
    sum_target: f32,
    sum_source: f32,
    sum_target_sq: f32,
    sum_source_sq: f32,
    sum_product: f32,
    count: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> ncc_uniforms: NccUniforms;
@group(0) @binding(1) var<storage, read> target_volume: array<f32>;
@group(0) @binding(2) var<storage, read> source_resampled: array<f32>;
@group(0) @binding(3) var<storage, read_write> partials: array<NccPartials>;

var<workgroup> local_sum_target: array<f32, 256>;
var<workgroup> local_sum_source: array<f32, 256>;
var<workgroup> local_sum_target_sq: array<f32, 256>;
var<workgroup> local_sum_source_sq: array<f32, 256>;
var<workgroup> local_sum_product: array<f32, 256>;
var<workgroup> local_count: array<u32, 256>;

// Compute NCC partial sums with workgroup reduction
// Using 256 invocations to fit within Intel Arc's limit
@compute @workgroup_size(256)
fn ncc_reduce(@builtin(global_invocation_id) global_id: vec3<u32>,
              @builtin(local_invocation_id) local_id: vec3<u32>,
              @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let idx = global_id.x;
    let lid = local_id.x;

    // Initialize local values
    var t: f32 = 0.0;
    var s: f32 = 0.0;
    var count: u32 = 0u;

    if (idx < ncc_uniforms.num_voxels) {
        t = target_volume[idx];
        s = source_resampled[idx];
        // Only count non-zero voxels (where source was in bounds)
        if (s > 0.0 || t > 0.0) {
            count = 1u;
        }
    }

    // Store to local memory
    local_sum_target[lid] = t;
    local_sum_source[lid] = s;
    local_sum_target_sq[lid] = t * t;
    local_sum_source_sq[lid] = s * s;
    local_sum_product[lid] = t * s;
    local_count[lid] = count;

    workgroupBarrier();

    // Parallel reduction within workgroup
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (lid < stride) {
            local_sum_target[lid] += local_sum_target[lid + stride];
            local_sum_source[lid] += local_sum_source[lid + stride];
            local_sum_target_sq[lid] += local_sum_target_sq[lid + stride];
            local_sum_source_sq[lid] += local_sum_source_sq[lid + stride];
            local_sum_product[lid] += local_sum_product[lid + stride];
            local_count[lid] += local_count[lid + stride];
        }
        workgroupBarrier();
    }

    // First thread writes workgroup result
    if (lid == 0u) {
        let wg_idx = workgroup_id.x;
        partials[wg_idx].sum_target = local_sum_target[0];
        partials[wg_idx].sum_source = local_sum_source[0];
        partials[wg_idx].sum_target_sq = local_sum_target_sq[0];
        partials[wg_idx].sum_source_sq = local_sum_source_sq[0];
        partials[wg_idx].sum_product = local_sum_product[0];
        partials[wg_idx].count = local_count[0];
    }
}
