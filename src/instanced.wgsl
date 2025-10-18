struct Uniforms { width: f32, height: f32, cell: f32, _pad: f32 };
@group(0) @binding(0) var<uniform> uni: Uniforms;

struct VsInVert { @location(0) pos: vec2<f32> };
struct VsInInst {
  @location(1) gx: u32,
  @location(2) gy: u32,
  @location(3) r: f32,
  @location(4) g: f32,
  @location(5) b: f32,
  @location(6) a: f32,
};

struct VSOut { @builtin(position) pos: vec4<f32>; @location(0) col: vec4<f32>; };

@vertex
fn vs(v: VsInVert, i: VsInInst) -> VSOut {
  let cell = uni.cell;
  let px = f32(i.gx) * cell;
  let py = f32(i.gy) * cell;
  let x = (px + v.pos.x * cell) / uni.width * 2.0 - 1.0;
  let y = (py + v.pos.y * cell) / uni.height * 2.0 - 1.0;
  var o: VSOut;
  o.pos = vec4<f32>(x, y, 0.0, 1.0);
  o.col = vec4<f32>(i.r, i.g, i.b, i.a);
  return o;
}

@fragment
fn fs(@location(0) col: vec4<f32>) -> @location(0) vec4<f32> {
  return col;
}
