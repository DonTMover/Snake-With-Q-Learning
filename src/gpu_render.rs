#![cfg(feature = "gpu-render")]

use wgpu::util::DeviceExt;
use winit::window::Window;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    width: f32,
    height: f32,
    cell: f32,
    _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Instance {
    pub gx: u32,
    pub gy: u32,
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

pub struct GpuRenderer {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    // Grid pipeline (full-screen pass with checkerboard)
    grid_pipeline: wgpu::RenderPipeline,
    // Cell pipeline (instanced quads)
    cell_pipeline: wgpu::RenderPipeline,
    uniform_buf: wgpu::Buffer,
    uniform_bg: wgpu::BindGroup,
    quad_vb: wgpu::Buffer,
    instance_buf: wgpu::Buffer,
    instance_capacity: usize,
}

impl GpuRenderer {
    pub async fn new(window: &Window, width: u32, height: u32) -> anyhow::Result<Self> {
        let instance = wgpu::Instance::default();
        let surface = unsafe { instance.create_surface(window) }?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| anyhow::anyhow!("No GPU adapter"))?;
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                },
                None,
            )
            .await?;

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode: caps.present_modes.get(0).copied().unwrap_or(wgpu::PresentMode::Fifo),
            alpha_mode: caps.alpha_modes.get(0).copied().unwrap_or(wgpu::CompositeAlphaMode::Auto),
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        // Uniforms
        let uniforms = Uniforms { width: width as f32, height: height as f32, cell: 20.0, _pad: 0.0 };
        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("uniform-layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let uniform_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("uniform-bg"),
            layout: &uniform_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() }],
        });

        // Shaders
        let grid_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("grid-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("grid.wgsl").into()),
        });
        let cell_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cell-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("instanced.wgsl").into()),
        });

        // Pipelines
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("grid-pl"),
            bind_group_layouts: &[&uniform_layout],
            push_constant_ranges: &[],
        });
        let grid_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("grid-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState { module: &grid_shader, entry_point: "vs", buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &grid_shader,
                entry_point: "fs",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let cell_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cell-pl"),
            bind_group_layouts: &[&uniform_layout],
            push_constant_ranges: &[],
        });
        let quad_vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("quad-vb"),
            contents: bytemuck::bytes_of(&[
                // pos (x,y) as a strip: (0,0)-(1,0)-(0,1)-(1,1)
                0.0f32, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0,
            ]),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let instance_capacity = 4096usize;
        let instance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instance-buf"),
            size: (instance_capacity * std::mem::size_of::<Instance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cell_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("cell-pipeline"),
            layout: Some(&cell_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &cell_shader,
                entry_point: "vs",
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<[f32; 2]>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x2],
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<Instance>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[
                            wgpu::VertexAttribute { shader_location: 1, offset: 0, format: wgpu::VertexFormat::Uint32 },
                            wgpu::VertexAttribute { shader_location: 2, offset: 4, format: wgpu::VertexFormat::Uint32 },
                            wgpu::VertexAttribute { shader_location: 3, offset: 8, format: wgpu::VertexFormat::Float32 },
                            wgpu::VertexAttribute { shader_location: 4, offset: 12, format: wgpu::VertexFormat::Float32 },
                            wgpu::VertexAttribute { shader_location: 5, offset: 16, format: wgpu::VertexFormat::Float32 },
                            wgpu::VertexAttribute { shader_location: 6, offset: 20, format: wgpu::VertexFormat::Float32 },
                        ],
                    },
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &cell_shader,
                entry_point: "fs",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleStrip, ..Default::default() },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        Ok(Self {
            surface,
            device,
            queue,
            config,
            grid_pipeline,
            cell_pipeline,
            uniform_buf,
            uniform_bg,
            quad_vb,
            instance_buf,
            instance_capacity,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 { return; }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        let uniforms = Uniforms { width: width as f32, height: height as f32, cell: 20.0, _pad: 0.0 };
        self.queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));
    }

    pub fn render(&mut self, instances: &[Instance]) -> anyhow::Result<()> {
        // Ensure capacity
        if instances.len() > self.instance_capacity {
            // Recreate buffer with larger capacity
            let new_cap = (instances.len().next_power_of_two()).max(self.instance_capacity * 2);
            self.instance_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("instance-buf"),
                size: (new_cap * std::mem::size_of::<Instance>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.instance_capacity = new_cap;
        }
        if !instances.is_empty() {
            self.queue.write_buffer(&self.instance_buf, 0, bytemuck::cast_slice(instances));
        }

        let frame = self
            .surface
            .get_current_texture()
            .map_err(|e| anyhow::anyhow!("surface acquire failed: {e}"))?;
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("encoder") });

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("grid+cells"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.04, g: 0.04, b: 0.06, a: 1.0 }), store: true },
                })],
                depth_stencil_attachment: None,
            });
            // Grid full-screen
            rpass.set_pipeline(&self.grid_pipeline);
            rpass.set_bind_group(0, &self.uniform_bg, &[]);
            rpass.draw(0..3, 0..1); // full-screen triangle

            // Cells
            if !instances.is_empty() {
                rpass.set_pipeline(&self.cell_pipeline);
                rpass.set_bind_group(0, &self.uniform_bg, &[]);
                rpass.set_vertex_buffer(0, self.quad_vb.slice(..));
                rpass.set_vertex_buffer(1, self.instance_buf.slice(..(instances.len() * std::mem::size_of::<Instance>()) as u64));
                rpass.draw(0..4, 0..(instances.len() as u32));
            }
        }
        self.queue.submit(Some(encoder.finish()));
        frame.present();
        Ok(())
    }
}
