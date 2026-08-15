use bytemuck::{NoUninit, Pod, Zeroable};
use std::{
    borrow::Cow, collections::HashMap, f32::consts::PI, num::NonZeroU64, str::FromStr,
    time::Duration,
};
use wgpu::{
    Backends, InstanceFlags, include_spirv, include_spirv_raw, util::DeviceExt,
    wgc::SubmissionIndex,
};

use fastrand::Rng;
use glam::{Vec2, Vec3, vec2, vec3};

struct SurfaceSample {
    position: Vec3,
    normal: Vec3,
    p: f32, // 1/m^2
}

fn sample_box_surface(rng: &mut Rng, min: Vec3, max: Vec3) -> SurfaceSample {
    let d = max - min;
    let a = Vec3::new(d.y * d.z, d.x * d.z, d.x * d.y);
    let total_area = 2.0 * (a.x + a.y + a.z);

    let r = rng.f32() * (a.x + a.y + a.z);
    let u = rng.f32();
    let v = rng.f32();
    let sgn = rng.bool();

    let (position, normal) = if r < a.x {
        let x = if sgn { max.x } else { min.x };
        (
            vec3(x, min.y + u * d.y, min.z + v * d.z),
            if sgn { Vec3::X } else { Vec3::NEG_X },
        )
    } else if r < a.x + a.y {
        let y = if sgn { max.y } else { min.y };
        (
            vec3(min.x + u * d.x, y, min.z + v * d.z),
            if sgn { Vec3::Y } else { Vec3::NEG_Y },
        )
    } else {
        let z = if sgn { max.z } else { min.z };
        (
            vec3(min.x + u * d.x, min.y + v * d.y, z),
            if sgn { Vec3::Z } else { Vec3::NEG_Z },
        )
    };

    SurfaceSample {
        position,
        normal,
        p: 1.0 / total_area,
    }
}

const MM: f32 = 1e-3;

struct Plate {
    origin: Vec3,
    u: Vec3,
    v: Vec3,
    n: Vec3,
    dim: Vec2,
}

#[derive(Pod, Clone, Copy, Zeroable)]
#[repr(C, align(16))]
struct PlateInfo {
    origin: [f32; 3],
    pad0: f32,
    u: [f32; 3],
    pad1: f32,
    v: [f32; 3],
    pad2: f32,
    n: [f32; 3],
    pad3: f32,
    dim: [f32; 2],
    pad4: [f32; 2],
    box_min: [f32; 3],
    pad5: f32,
    box_max: [f32; 3],
    pad6: f32,
}

const PLATE_COUNT: usize = 2;

#[derive(Pod, Clone, Copy, Zeroable)]
#[repr(C, align(16))]
struct Consts {
    plates: [PlateInfo; PLATE_COUNT],
    seed: u32,
    pad: [u32; 3],
}

impl Plate {
    fn into_plate_info(&self) -> PlateInfo {
        let (box_min, box_max) = self.enclosing_box(5.0 * MM * Vec3::ONE);
        PlateInfo {
            origin: self.origin.to_array(),
            u: self.u.to_array(),
            v: self.v.to_array(),
            n: self.n.to_array(),
            dim: self.dim.to_array(),
            box_min: box_min.to_array(),
            box_max: box_max.to_array(),
            pad0: 0.0,
            pad1: 0.0,
            pad2: 0.0,
            pad3: 0.0,
            pad4: [0.0, 0.0],
            pad5: 0.0,
            pad6: 0.0,
        }
    }

    fn enclosing_box(&self, padding: Vec3) -> (Vec3, Vec3) {
        let min = self.origin - padding.x * self.u - padding.y * self.v - padding.z * self.n;
        let max = self.origin
            + (self.dim.x + padding.x) * self.u
            + (self.dim.y + padding.y) * self.v
            + padding.z * self.n;
        (min, max)
    }

    fn sqdist(&self, sp: Vec3) -> f32 {
        let d = sp - self.origin;

        let x = d.dot(self.u);
        let y = d.dot(self.v);
        let z = d.dot(self.n);

        let cx = x.clamp(0.0, self.dim.x);
        let cy = y.clamp(0.0, self.dim.y);

        let dx = x - cx;
        let dy = y - cy;

        dx * dx + dy * dy + z * z
    }
}

#[inline(always)]
pub fn random_dir(rng: &mut Rng) -> Vec3 {
    const MASK: u64 = (1 << 24) - 1;
    const SCALE: f32 = 1.0 / 8_388_608.0; // 2^-23

    loop {
        let bits = rng.u64(..);

        let x = (bits & MASK) as f32 * SCALE - 1.0;
        let y = ((bits >> 24) & MASK) as f32 * SCALE - 1.0;

        let s = x * x + y * y;

        if s > 0.0 && s < 1.0 {
            let a = 2.0 * (1.0 - s).sqrt();

            return Vec3::new(x * a, y * a, 1.0 - 2.0 * s);
        }
    }
}

fn sample_conditional_return_on_sphere(
    rng: &mut fastrand::Rng,
    x: Vec3,
    center: Vec3,
    a: f32,
) -> Vec3 {
    let v = x - center;
    let r = v.length();
    let e = v / r;
    let xi = rng.f32();
    let inv_d = (1.0 - xi) / (r + a) + xi / (r - a);
    let d = 1.0 / inv_d;

    let cos_theta = ((r * r + a * a - d * d) / (2.0 * r * a)).clamp(-1.0, 1.0);
    let sin_theta = f32::sqrt((1.0 - cos_theta * cos_theta).max(0.0));
    let phi = 2.0 * PI * rng.f32();
    let (t1, t2) = e.any_orthonormal_pair();
    let dir = cos_theta * e + sin_theta * (f32::cos(phi) * t1 + f32::sin(phi) * t2);
    center + a * dir
}

fn main_cpu() {
    let z1 = 10.0 * MM;
    let z2 = -10.0 * MM;

    let p1 = Plate {
        origin: vec3(0.0, 0.0, z1),
        u: Vec3::X,
        v: Vec3::Y,
        n: Vec3::Z,
        dim: vec2(100.0 * MM, 100.0 * MM),
    };
    let p2 = Plate {
        origin: vec3(0.0, 0.0, z2),
        u: Vec3::X,
        v: Vec3::Y,
        n: Vec3::Z,
        dim: vec2(100.0 * MM, 100.0 * MM),
    };

    let g1 = p1.enclosing_box(5.0 * MM * Vec3::ONE);

    const EPSILON_ZERO: f32 = 8.8541878188e-12;

    const SURFACE_TOL: f32 = 1e-6;

    let epsilon = EPSILON_ZERO;

    let mut rng = Rng::new();
    let closest_sqdist = |sp: Vec3| {
        let d1 = p1.sqdist(sp);
        let d2 = p2.sqdist(sp);
        if d1 < d2 { (0, d1) } else { (1, d2) }
    };

    let outside_radius = 1.0;

    let walk_on_sphere = |rng: &mut Rng, mut sp: Vec3| loop {
        let sp_d2 = sp.length_squared();
        if sp_d2 >= 16.0 * outside_radius * outside_radius {
            let return_proba = outside_radius / f32::sqrt(sp_d2);
            if rng.f32() > return_proba {
                return usize::MAX;
            }
            sp = sample_conditional_return_on_sphere(rng, sp, Vec3::ZERO, outside_radius);
            continue;
        }
        let (id, d2) = closest_sqdist(sp);
        if d2 < SURFACE_TOL * SURFACE_TOL {
            return id;
        }
        let d = f32::sqrt(d2);
        let r = 0.999999 * d;
        sp += r * random_dir(rng);
    };

    let mut result = vec![0.0f64; 2];

    let mut sample_count = 0;

    for _ in 0..100000000 {
        let g = sample_box_surface(&mut rng, g1.0, g1.1);

        let x = g.position;
        let n = g.normal;
        let pg = g.p;

        let clearance = f32::sqrt(closest_sqdist(x).1);
        let r = 0.995 * clearance;

        let omega = random_dir(&mut rng);
        let x_p = x + r * omega;
        let x_m = x - r * omega;
        let w = -epsilon * (1.0 / pg) * (3.0 / (2.0 * r)) * Vec3::dot(n, omega);

        let a = walk_on_sphere(&mut rng, x_p);
        let b = walk_on_sphere(&mut rng, x_m);
        if a != usize::MAX {
            result[a] += w as f64;
        }
        if b != usize::MAX {
            result[b] -= w as f64;
        }
        sample_count += 1;
        if sample_count % 100000 == 0 {
            let mut aa = result.clone();
            aa.iter_mut().for_each(|v| {
                *v /= (sample_count as f64);
                *v *= 1e12;
            });
            println!("result = {aa:.3?} ({:.2} Ms)", (sample_count as f32) / 1e6);
        }
    }
}

struct Gpu {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

fn init_gpu() -> Gpu {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = Backends::VULKAN;
    let instance = wgpu::Instance::new(desc);

    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("Failed to create adapter");

    println!("Running on Adapter: {:#?}", adapter.get_info());

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: None,
        required_features: wgpu::Features::SUBGROUP
            | wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
            | wgpu::Features::PASSTHROUGH_SHADERS,
        required_limits: wgpu::Limits::defaults(),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::MemoryUsage,
        trace: wgpu::Trace::Off,
    }))
    .expect("Failed to create device");

    Gpu {
        instance,
        adapter,
        device,
        queue,
    }
}

#[derive(Debug)]
struct Shaders {
    pipeline_layout: wgpu::PipelineLayout,
    pipelines: HashMap<String, wgpu::ComputePipeline>,
    bind_group_layouts: Vec<wgpu::BindGroupLayout>,
}

fn compile_shaders(
    gpu: &Gpu,
    path: &str,
    source: &str,
    //bind_group_layout: wgpu::BindGroupLayout,
    entry_point_names: &[&str],
) -> Shaders {
    let global_session = shader_slang::GlobalSession::new().unwrap();
    println!("tag = {:?}", global_session.build_tag_string());
    let target = shader_slang::TargetDesc::default()
        .format(shader_slang::CompileTarget::Spirv)
        .profile(global_session.find_profile("spirv_1_5"));

    let targets = [target];

    let options = shader_slang::CompilerOptions::default()
        .vulkan_use_entry_point_name(true)
        .debug_information(shader_slang::DebugInfoLevel::Minimal)
        .optimization(shader_slang::OptimizationLevel::None);
    let session_desc = shader_slang::SessionDesc::default()
        .targets(&targets)
        .options(&options);

    let session = global_session
        .create_session(&session_desc)
        .expect("failed to create slang compilation session");
    let shader = session
        .load_module_from_source_string("my_shader", path, source)
        .expect("failed to load slang module");

    let entries: Vec<_> = entry_point_names
        .iter()
        .map(|name| {
            shader
                .find_entry_point_by_name(name)
                .expect("entry point not found")
        })
        .collect();

    let spec_source = format!(
        r#"
        export static const uint PLATE_COUNT = {};
        "#,
        PLATE_COUNT,
    );

    let spec_module = session
        .load_module_from_source_string("spec_foo_16", "spec_foo_16.slang", &spec_source)
        .expect("failed to load spec slang module");

    let program = session
        .create_composite_component_type(&[
            shader.clone().into(),
            spec_module.into(),
            entries[0].clone().into(),
        ])
        .expect("failed to create composite slang component (?)");

    let linked = program.link().expect("failed to link slang program");

    let mut bind_group_layout_entries: Vec<wgpu::BindGroupLayoutEntry> = Vec::new();

    for param in linked.layout(0).unwrap().parameters() {
        assert_eq!(param.binding_space(), 0);
        let type_layout = param.type_layout().unwrap();

        println!("param = {:?}", param.name().unwrap_or("??"));

        match type_layout.kind() {
            shader_slang::TypeKind::ConstantBuffer => {
                bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
                    binding: param.binding_index(),
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            NonZeroU64::new(
                                type_layout
                                    .element_type_layout()
                                    .unwrap()
                                    .size(shader_slang::ParameterCategory::Uniform)
                                    as u64,
                            )
                            .unwrap(),
                        ),
                    },
                    count: None,
                });
            }
            shader_slang::TypeKind::Resource
                if type_layout.resource_shape().unwrap()
                    == shader_slang::ResourceShape::SlangStructuredBuffer =>
            {
                let el = type_layout.element_type_layout().unwrap();
                /*println!("cats = ");
                println!("cat = {:?}", type_layout.parameter_category());
                println!("elcat = {:?}", el.parameter_category());
                println!(
                    "type stride = {:?} {:?}",
                    type_layout.stride(shader_slang::ParameterCategory::DescriptorTableSlot),
                    type_layout
                        .element_stride(shader_slang::ParameterCategory::DescriptorTableSlot)
                );
                println!(
                    "el stride = {:?} {:?}",
                    el.stride(shader_slang::ParameterCategory::Uniform),
                    el.element_stride(shader_slang::ParameterCategory::Uniform)
                );*/

                bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
                    binding: param.binding_index(),
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: match type_layout.resource_access().unwrap() {
                                shader_slang::ResourceAccess::None
                                | shader_slang::ResourceAccess::Read => true,
                                shader_slang::ResourceAccess::Write
                                | shader_slang::ResourceAccess::ReadWrite => false,
                                _ => todo!(),
                            },
                        },
                        has_dynamic_offset: false,
                        // the stride / Uniform thing makes no sense to me
                        min_binding_size: Some(
                            NonZeroU64::new(
                                el.stride(shader_slang::ParameterCategory::Uniform) as u64
                            )
                            .unwrap(),
                        ),
                    },
                    count: None,
                });
            }
            kind => todo!("type param kind: {kind:?}"),
        }
    }

    for ep in linked.layout(0).unwrap().entry_points() {
        println!("thread count = {:?}", ep.compute_thread_group_size());
        println!("EP name = {:?}", ep.name());
    }

    let bind_group_layout = gpu
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &bind_group_layout_entries,
        });

    let spirv_blob = linked
        .entry_point_code(0, 0)
        .expect("failed to get spirv blob");
    let spirv: &[u8] = spirv_blob.as_slice();

    //println!("src = {:?}", wgpu::util::make_spirv(spirv));
    std::fs::write("dump.spv", &spirv).expect("!!");
    let source = wgpu::ShaderModuleDescriptor {
        source: wgpu::util::make_spirv(spirv),
        label: None,
    };
    //let source = include_spirv!(concat!(env!("OUT_DIR"), "/", "shader.slang.spv"));

    let module = if true {
        unsafe {
            gpu.device.create_shader_module_passthrough(
                wgpu::wgt::CreateShaderModuleDescriptorPassthrough {
                    label: None,
                    entry_points: Cow::Owned(
                        linked
                            .layout(0)
                            .unwrap()
                            .entry_points()
                            .map(|ep| {
                                let wg_size = ep.compute_thread_group_size();
                                wgpu::PassthroughShaderEntryPoint {
                                    name: ep.name().unwrap().into(),
                                    workgroup_size: (
                                        wg_size[0] as u32,
                                        wg_size[1] as u32,
                                        wg_size[2] as u32,
                                    ),
                                }
                            })
                            .collect(),
                    ),
                    spirv: Some(Cow::Borrowed(bytemuck::pod_align_to(spirv).1)),
                    dxil: None,
                    hlsl: None,
                    metallib: None,
                    msl: None,
                    glsl: None,
                    wgsl: None,
                },
            )
        }
    } else {
        gpu.device.create_shader_module(source)
    };

    let pipeline_layout = gpu
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

    let bind_group_layouts = vec![bind_group_layout];

    let pipelines: HashMap<String, wgpu::ComputePipeline> = entry_point_names
        .iter()
        .copied()
        .map(|entry| {
            let pipe = gpu
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: None,
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });
            (entry.to_owned(), pipe)
        })
        .collect();

    Shaders {
        pipeline_layout,
        pipelines,
        bind_group_layouts,
    }

    //println!("spirv = {spirv:?}");
}

const WORKGROUP_COUNT: u32 = 4 * 16 * 16; //128 * 256;

fn main_gpu() {
    env_logger::init();

    let z1 = 10.0 * MM;
    let z2 = -10.0 * MM;

    let p1 = Plate {
        origin: vec3(0.0, 0.0, z1),
        u: Vec3::X,
        v: Vec3::Y,
        n: Vec3::Z,
        dim: vec2(100.0 * MM, 100.0 * MM),
    };
    let p2 = Plate {
        origin: vec3(0.0, 0.0, z2),
        u: Vec3::X,
        v: Vec3::Y,
        n: Vec3::Z,
        dim: vec2(100.0 * MM, 100.0 * MM),
    };

    let mut rng = Rng::new();
    let seed = rng.u32(..);

    let consts = Consts {
        plates: [p1.into_plate_info(), p2.into_plate_info()],
        seed,
        pad: [0; 3],
    };
    let gpu = init_gpu();
    let shaders = compile_shaders(
        &gpu,
        "shader.slang",
        include_str!("shader.slang"),
        &["sample_walk"],
    );
    println!("shaders = {shaders:?}");

    let const_buffer = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&consts),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    struct Submission {
        index: Option<wgpu::SubmissionIndex>,
        output_buffer: wgpu::Buffer,
        copy_buffer: wgpu::Buffer,

        query_buffer: wgpu::Buffer,
        query_copy_buffer: wgpu::Buffer,
        query_set: wgpu::QuerySet,
    }
    const QUERY_COUNT: usize = 2;
    fn allocate_submission(gpu: &Gpu) -> Submission {
        let output_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (PLATE_COUNT * size_of::<f32>() * (WORKGROUP_COUNT as usize)) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let copy_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: output_buffer.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let query_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (QUERY_COUNT * size_of::<u64>()) as u64,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let query_copy_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (QUERY_COUNT * size_of::<u64>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let query_set = gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: None,
            ty: wgpu::QueryType::Timestamp,
            count: QUERY_COUNT as u32,
        });

        Submission {
            index: None,
            output_buffer,
            copy_buffer,
            query_buffer,
            query_copy_buffer,
            query_set,
        }
    }

    let mut submission = allocate_submission(&gpu);

    fn submit(
        gpu: &Gpu,
        shaders: &Shaders,
        const_buffer: &wgpu::Buffer,
        submission: &mut Submission,
    ) {
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &shaders.bind_group_layouts[0],
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: const_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: submission.output_buffer.as_entire_binding(),
                },
            ],
        });
        let mut cmd_buff = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        cmd_buff.write_timestamp(&submission.query_set, 0);
        let mut compute_pass = cmd_buff.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(shaders.pipelines.get("sample_walk").unwrap());
        compute_pass.set_bind_group(0, &bind_group, &[]);

        compute_pass.dispatch_workgroups(WORKGROUP_COUNT, 1, 1);

        drop(compute_pass);
        cmd_buff.write_timestamp(&submission.query_set, 1);

        cmd_buff.copy_buffer_to_buffer(
            &submission.output_buffer,
            0,
            &submission.copy_buffer,
            0,
            submission.output_buffer.size(),
        );

        cmd_buff.resolve_query_set(
            &submission.query_set,
            0..QUERY_COUNT as u32,
            &submission.query_buffer,
            0,
        );
        cmd_buff.copy_buffer_to_buffer(
            &submission.query_buffer,
            0,
            &submission.query_copy_buffer,
            0,
            submission.query_buffer.size(),
        );

        let cmd_buff = cmd_buff.finish();
        let submission_index = gpu.queue.submit([cmd_buff]);
        println!("submitted : {submission_index:?}");
        assert!(submission.index.is_none());
        submission.index = Some(submission_index);
    }

    submit(&gpu, &shaders, &const_buffer, &mut submission);

    gpu.device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission.index.unwrap()),
            timeout: None, //Some(Duration::ZERO),
        })
        .expect("failed to poll");
    println!("submission done !");
    println!("map async ...");
    submission
        .copy_buffer
        .map_async(wgpu::MapMode::Read, .., |_| {});
    submission
        .query_copy_buffer
        .map_async(wgpu::MapMode::Read, .., |_| {});
    gpu.device.poll(wgpu::PollType::Poll).unwrap();
    let buffer_map = submission.copy_buffer.get_mapped_range(..).unwrap();
    let buffer_data: &[u8] = &buffer_map;
    /*println!(
        "buffer map data = {:?}",
        bytemuck::allocation::pod_collect_to_vec::<u8, f32>(buffer_data)
    );*/
    let query_buffer_map = submission.query_copy_buffer.get_mapped_range(..).unwrap();
    let query_values = bytemuck::allocation::pod_collect_to_vec::<u8, u64>(&query_buffer_map);
    println!("qv = {query_values:?}");
    let dt = (((query_values[1] - query_values[0]) as f64)
        * (gpu.queue.get_timestamp_period() as f64))
        * 1e-9;
    println!("delta = {:.2} ms", dt * 1e3);
    let wg_size = 64 * 4;
    let n_samples = wg_size * WORKGROUP_COUNT;
    let samples_per_sec = (n_samples as f64) / dt;
    println!("n walks = {:.2} Ms/sec", samples_per_sec / 1e6);
    println!(
        "avg wave time = {:.2} us",
        dt / (WORKGROUP_COUNT * wg_size / 64) as f64 * 1e6
    );

    #[cfg(false)]
    {
        println!("begin record");

        println!("submited !");

        let buffer_slice = download_buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |_| {});

        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

        let data = buffer_slice.get_mapped_range().unwrap();
        let result: Vec<f32> = bytemuck::allocation::pod_collect_to_vec(&data);

        println!("Result: {result:?}");

        let mut mean = vec![0.0; PLATE_COUNT];
        for i in 0..WORKGROUP_COUNT as usize {
            for k in 0..PLATE_COUNT {
                mean[k] += result[PLATE_COUNT * i + k];
            }
        }
        for m in &mut mean {
            *m /= (WORKGROUP_COUNT as f32);
            *m *= 1e12;
        }
        println!("mean = {mean:?}");
    }
}

fn main() {
    //main_cpu();
    main_gpu();
}
