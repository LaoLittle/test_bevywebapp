use std::collections::VecDeque;
use std::fs::File;
use std::io::BufReader;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, SyncSender, TryRecvError, sync_channel};
use std::thread;
use std::time::{Duration, Instant};

use oxideav_core::{Demuxer, NullCodecResolver};
use rav1d::{Decoder as Av1Decoder, PlanarImageComponent, Rav1dError};
use rodio::{DeviceSinkBuilder, MixerDeviceSink, Player, buffer::SamplesBuffer};
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

#[derive(Clone, Copy)]
pub struct ColorSpace {
    pub rgb_to_yuv: [[f32; 3]; 3],
    pub yuv_to_rgb: [[f32; 3]; 3],
}

impl ColorSpace {
    pub const BT601: Self = Self {
        rgb_to_yuv: [
            [0.299, 0.587, 0.114],
            [-0.168736, -0.331264, 0.5],
            [0.5, -0.418688, -0.081312],
        ],
        yuv_to_rgb: [
            [1.0, 0.0, 1.402],
            [1.0, -0.344136, -0.714136],
            [1.0, 1.772, 0.0],
        ],
    };

    pub const BT709: Self = Self {
        rgb_to_yuv: [
            [0.2126, 0.7152, 0.0722],
            [-0.114572, -0.385428, 0.5],
            [0.5, -0.454153, -0.045847],
        ],
        yuv_to_rgb: [
            [1.0, 0.0, 1.5748],
            [1.0, -0.187324, -0.468124],
            [1.0, 1.8556, 0.0],
        ],
    };

    pub const BT2020: Self = Self {
        rgb_to_yuv: [
            [0.2627, 0.678, 0.0593],
            [-0.13963, -0.36037, 0.5],
            [0.5, -0.459786, -0.040214],
        ],
        yuv_to_rgb: [
            [1.0, 0.0, 1.4746],
            [1.0, -0.16455, -0.57135],
            [1.0, 1.8814, 0.0],
        ],
    };

    pub fn custom(rgb_to_yuv: [[f32; 3]; 3], yuv_to_rgb: [[f32; 3]; 3]) -> Self {
        Self {
            rgb_to_yuv,
            yuv_to_rgb,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ShaderColorMatrix {
    matrix: [[f32; 4]; 3],
    offset: [f32; 4],
}

impl From<ColorSpace> for ShaderColorMatrix {
    fn from(color_space: ColorSpace) -> Self {
        Self {
            matrix: color_space
                .yuv_to_rgb
                .map(|row| [row[0], row[1], row[2], 0.0]),
            offset: [0.0, 0.0, 0.0, 0.0],
        }
    }
}

struct Yuv420Image {
    width: u32,
    height: u32,
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
    chroma_width: u32,
    chroma_height: u32,
}

struct VideoFrame {
    timestamp: Duration,
    image: Yuv420Image,
}

enum DecodeEvent {
    Frame(VideoFrame),
    Audio {
        channels: u16,
        sample_rate: u32,
        samples: Vec<f32>,
    },
    End,
    Error(String),
}

impl Yuv420Image {
    fn from_picture(picture: &rav1d::Picture) -> Self {
        let width = picture.width();
        let height = picture.height();
        let chroma_width = width.div_ceil(2);
        let chroma_height = height.div_ceil(2);
        let copy_plane = |component: PlanarImageComponent, plane_width: u32, plane_height: u32| {
            let source = picture.plane(component);
            let stride = picture.stride(component) as usize;
            let mut output = vec![0; (plane_width * plane_height) as usize];
            for row in 0..plane_height as usize {
                let source_start = row * stride;
                let output_start = row * plane_width as usize;
                output[output_start..output_start + plane_width as usize]
                    .copy_from_slice(&source[source_start..source_start + plane_width as usize]);
            }
            output
        };
        Self {
            width,
            height,
            y: copy_plane(PlanarImageComponent::Y, width, height),
            u: copy_plane(PlanarImageComponent::U, chroma_width, chroma_height),
            v: copy_plane(PlanarImageComponent::V, chroma_width, chroma_height),
            chroma_width,
            chroma_height,
        }
    }
}

fn send_picture(
    sender: &SyncSender<DecodeEvent>,
    picture: rav1d::Picture,
    time_base: oxideav_core::TimeBase,
    first_timestamp: &mut Option<Duration>,
) -> Result<(), String> {
    let timestamp = Duration::from_secs_f64(
        time_base
            .seconds_of(picture.timestamp().unwrap_or(0))
            .max(0.0),
    );
    let first = *first_timestamp.get_or_insert(timestamp);
    sender
        .send(DecodeEvent::Frame(VideoFrame {
            timestamp: timestamp.saturating_sub(first),
            image: Yuv420Image::from_picture(&picture),
        }))
        .map_err(|_| "playback window closed".to_string())
}

fn start_stream(
    path: &str,
) -> Result<(Receiver<DecodeEvent>, VideoFrame, VecDeque<DecodeEvent>), Box<dyn std::error::Error>>
{
    let input = BufReader::new(File::open(path)?);
    let mut demuxer = oxideav_mkv::demux::open_typed(Box::new(input), &NullCodecResolver)?;
    let streams = demuxer.streams().to_vec();
    let video_stream = streams
        .iter()
        .find(|stream| stream.params.codec_id.as_str() == "av1")
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "AV1 video stream not found",
            )
        })?;
    let audio_stream = streams
        .iter()
        .find(|stream| stream.params.codec_id.as_str() == "pcm_s16le");
    let video_index = video_stream.index;
    let video_time_base = video_stream.time_base;
    let audio_index = audio_stream.map(|stream| stream.index);
    let audio_channels = audio_stream
        .and_then(|stream| stream.params.channels)
        .unwrap_or(2);
    let audio_rate = audio_stream
        .and_then(|stream| stream.params.sample_rate)
        .unwrap_or(48_000);
    let (sender, receiver) = sync_channel(4);
    let decode_sender = sender.clone();
    thread::spawn(move || {
        let result = (|| -> Result<(), String> {
            let mut decoder = Av1Decoder::new().map_err(|error| error.to_string())?;
            let mut first_timestamp = None;
            loop {
                let packet = match demuxer.next_packet() {
                    Ok(packet) => packet,
                    Err(_) => break,
                };
                if packet.stream_index == video_index {
                    let timestamp = packet.pts.unwrap_or(0);
                    let mut send_result = decoder.send_data(
                        packet.data.into_boxed_slice(),
                        None,
                        Some(timestamp),
                        packet.duration,
                    );
                    loop {
                        while let Ok(picture) = decoder.get_picture() {
                            send_picture(
                                &decode_sender,
                                picture,
                                video_time_base,
                                &mut first_timestamp,
                            )?;
                        }
                        match send_result {
                            Ok(()) => break,
                            Err(Rav1dError::TryAgain) => send_result = decoder.send_pending_data(),
                            Err(error) => return Err(error.to_string()),
                        }
                    }
                } else if audio_index == Some(packet.stream_index) {
                    let samples = packet
                        .data
                        .chunks_exact(2)
                        .map(|sample| i16::from_le_bytes([sample[0], sample[1]]) as f32 / 32768.0)
                        .collect();
                    decode_sender
                        .send(DecodeEvent::Audio {
                            channels: audio_channels,
                            sample_rate: audio_rate,
                            samples,
                        })
                        .map_err(|_| "playback window closed".to_string())?;
                }
            }
            decoder.flush();
            while let Ok(picture) = decoder.get_picture() {
                send_picture(
                    &decode_sender,
                    picture,
                    video_time_base,
                    &mut first_timestamp,
                )?;
            }
            decode_sender
                .send(DecodeEvent::End)
                .map_err(|_| "playback window closed".to_string())
        })();
        if let Err(error) = result {
            let _ = decode_sender.send(DecodeEvent::Error(error));
        }
    });
    let mut pending_events = VecDeque::new();
    let first_frame = loop {
        match receiver.recv()? {
            DecodeEvent::Frame(frame) => break frame,
            event @ DecodeEvent::Audio { .. } => pending_events.push_back(event),
            DecodeEvent::End => return Err("output.mkv contains no decoded AV1 frames".into()),
            DecodeEvent::Error(error) => return Err(error.into()),
        }
    };
    Ok((receiver, first_frame, pending_events))
}

struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    y_texture: wgpu::Texture,
    u_texture: wgpu::Texture,
    v_texture: wgpu::Texture,
}

impl Renderer {
    async fn new(window: Arc<Window>, yuv: &Yuv420Image, color_space: ColorSpace) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let surface = instance
            .create_surface(window.clone())
            .expect("failed to create surface");
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await
            .expect("no suitable GPU adapter found");
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .expect("failed to create GPU device");
        let caps = surface.get_capabilities(&adapter);
        let size = window.inner_size();
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|format| !format.is_srgb())
            .unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: caps.present_modes[0],
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("yuv420 shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });
        let make_texture = |label: &str, width: u32, height: u32| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
        };
        let y_texture = make_texture("Y plane", yuv.width, yuv.height);
        let u_texture = make_texture("U plane", yuv.chroma_width, yuv.chroma_height);
        let v_texture = make_texture("V plane", yuv.chroma_width, yuv.chroma_height);
        let upload = |texture: &wgpu::Texture, data: &[u8], width: u32, height: u32| {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(width),
                    rows_per_image: Some(height),
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            )
        };
        upload(&y_texture, &yuv.y, yuv.width, yuv.height);
        upload(&u_texture, &yuv.u, yuv.chroma_width, yuv.chroma_height);
        upload(&v_texture, &yuv.v, yuv.chroma_width, yuv.chroma_height);
        let color_matrix = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("YUV to RGB color matrix"),
            contents: bytemuck::bytes_of(&ShaderColorMatrix::from(color_space)),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let mut binding_entries: Vec<wgpu::BindGroupLayoutEntry> = [0, 1, 2]
            .into_iter()
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            })
            .collect();
        binding_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        });
        binding_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 4,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: Some(
                    std::num::NonZeroU64::new(std::mem::size_of::<ShaderColorMatrix>() as u64)
                        .unwrap(),
                ),
            },
            count: None,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("YUV plane bindings"),
            entries: &binding_entries,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let y_view = y_texture.create_view(&Default::default());
        let u_view = u_texture.create_view(&Default::default());
        let v_view = v_texture.create_view(&Default::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("YUV420 bind group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&y_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&u_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&v_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: color_matrix.as_entire_binding(),
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("YUV420 render pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        Self {
            surface,
            device,
            queue,
            config,
            pipeline,
            bind_group,
            y_texture,
            u_texture,
            v_texture,
        }
    }

    fn upload_frame(&self, image: &Yuv420Image) {
        let upload = |texture: &wgpu::Texture, data: &[u8], width: u32, height: u32| {
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(width),
                    rows_per_image: Some(height),
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
        };
        upload(&self.y_texture, &image.y, image.width, image.height);
        upload(
            &self.u_texture,
            &image.u,
            image.chroma_width,
            image.chroma_height,
        );
        upload(
            &self.v_texture,
            &image.v,
            image.chroma_width,
            image.chroma_height,
        );
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn render(&mut self) -> bool {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            _ => return false,
        };
        let view = frame.texture.create_view(&Default::default());
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.draw(0..6, 0..1);
        }
        self.queue.submit([encoder.finish()]);
        self.queue.present(frame);
        true
    }
}

struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    receiver: Receiver<DecodeEvent>,
    pending_events: VecDeque<DecodeEvent>,
    current_frame: VideoFrame,
    video_queue: VecDeque<VideoFrame>,
    color_space: ColorSpace,
    audio_output: Option<MixerDeviceSink>,
    audio_player: Option<Player>,
    started_at: Instant,
    paused_at: Duration,
    paused: bool,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title("Playing MKV video (av1 + pcm_s16le)"))
                .expect("failed to create window"),
        );
        let renderer = pollster::block_on(Renderer::new(
            window.clone(),
            &self.current_frame.image,
            self.color_space,
        ));
        let audio_output = DeviceSinkBuilder::open_default_sink().expect("no audio device");
        let audio_player = Player::connect_new(audio_output.mixer());
        self.audio_output = Some(audio_output);
        self.audio_player = Some(audio_player);
        self.window = Some(window);
        self.renderer = Some(renderer);
    }
    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(renderer) = &mut self.renderer {
                    renderer.resize(size.width, size.height);
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(renderer) = &mut self.renderer {
                    let elapsed = if self.paused {
                        self.paused_at
                    } else {
                        self.started_at.elapsed()
                    };
                    loop {
                        let event = self.pending_events.pop_front().or_else(|| {
                            match self.receiver.try_recv() {
                                Ok(event) => Some(event),
                                Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => None,
                            }
                        });
                        let Some(event) = event else { break };
                        match event {
                            DecodeEvent::Frame(frame) => {
                                self.video_queue.push_back(frame);
                            }
                            DecodeEvent::Audio {
                                channels,
                                sample_rate,
                                samples,
                            } => {
                                if let Some(player) = &self.audio_player {
                                    player.append(SamplesBuffer::new(
                                        channels.try_into().unwrap(),
                                        sample_rate.try_into().unwrap(),
                                        samples,
                                    ));
                                }
                            }
                            DecodeEvent::End => {}
                            DecodeEvent::Error(error) => eprintln!("decoder error: {error}"),
                        }
                    }
                    while self
                        .video_queue
                        .front()
                        .is_some_and(|frame| frame.timestamp <= elapsed)
                    {
                        self.current_frame = self.video_queue.pop_front().unwrap();
                    }
                    renderer.upload_frame(&self.current_frame.image);
                    renderer.render();
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Released,
                button: MouseButton::Left,
                ..
            } => {
                self.paused = !self.paused;
                if self.paused {
                    self.paused_at = self.started_at.elapsed();
                    if let Some(player) = &self.audio_player {
                        player.pause();
                    }
                } else {
                    self.started_at = Instant::now() - self.paused_at;
                    if let Some(player) = &self.audio_player {
                        player.play();
                    }
                }
            }
            _ => {}
        }
    }
    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let color_space = ColorSpace::BT709;
    let (receiver, current_frame, pending_events) = start_stream("output.mkv")?;
    let event_loop = EventLoop::new()?;
    event_loop.run_app(&mut App {
        window: None,
        renderer: None,
        receiver,
        pending_events,
        current_frame,
        video_queue: VecDeque::new(),
        color_space,
        audio_output: None,
        audio_player: None,
        started_at: Instant::now(),
        paused_at: Duration::ZERO,
        paused: false,
    })?;

    Ok(())
}
