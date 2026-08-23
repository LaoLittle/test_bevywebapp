struct VertexOut {
    @builtin(position)
    position: vec4<f32>,
    @location(0)
    uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOut {
    var pos = array<vec2<f32>, 6>(
        vec2(-1.0, -1.0),
        vec2(1.0, -1.0),
        vec2(1.0, 1.0),
        vec2(-1.0, -1.0),
        vec2(1.0, 1.0),
        vec2(-1.0, 1.0),
    );

    var uv = array<vec2<f32>, 6>(
        vec2(0.0, 1.0),
        vec2(1.0, 1.0),
        vec2(1.0, 0.0),
        vec2(0.0, 1.0),
        vec2(1.0, 0.0),
        vec2(0.0, 0.0),
    );

    var out: VertexOut;
    out.position = vec4(pos[index], 0.0, 1.0);
    out.uv = uv[index];
    return out;
}

@group(0) @binding(0)
var y_tex: texture_2d<f32>;
@group(0) @binding(1)
var u_tex: texture_2d<f32>;
@group(0) @binding(2)
var v_tex: texture_2d<f32>;
@group(0) @binding(3)
var samp: sampler;

struct ColorMatrix {
    matrix: array<vec4<f32>, 3>,
    offset: vec4<f32>,
};

@group(0) @binding(4)
var<uniform> color_matrix: ColorMatrix;

@fragment
fn fs_main(input: VertexOut) -> @location(0) vec4<f32> {
    let y = textureSample(y_tex, samp, input.uv).r;
    let u = textureSample(u_tex, samp, input.uv).r - 0.5;
    let v = textureSample(v_tex, samp, input.uv).r - 0.5;
    let yuv = vec3<f32>(y, u, v);
    let r = dot(color_matrix.matrix[0].xyz, yuv) + color_matrix.offset.x;
    let g = dot(color_matrix.matrix[1].xyz, yuv) + color_matrix.offset.y;
    let b = dot(color_matrix.matrix[2].xyz, yuv) + color_matrix.offset.z;

    return vec4(
        clamp(r, 0.0, 1.0),
        clamp(g, 0.0, 1.0),
        clamp(b, 0.0, 1.0),
        1.0,
    );
}