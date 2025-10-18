struct Uniforms { width: f32, height: f32, cell: f32, _pad: f32 };
@group(0) @binding(0) var<uniform> uni: Uniforms;

struct VSOut { @builtin(position) pos: vec4<f32> };

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  var pos = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -3.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>( 3.0,  1.0),
  );
  var o: VSOut;
  o.pos = vec4<f32>(pos[vi], 0.0, 1.0);
  return o;
}

@fragment
fn fs(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
  // Checkerboard grid based on pixel coordinates
  let x = i32(p.x / uni.cell);
  let y = i32(p.y / uni.cell);
  let is_dark = ((x + y) & 1) == 0;
  let c_dark = vec3<f32>(0.14, 0.14, 0.20);
  let c_light = vec3<f32>(0.16, 0.16, 0.24);
  let col = select(c_light, c_dark, is_dark);
  return vec4<f32>(col, 1.0);
}
