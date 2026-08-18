import common;

struct ViewConsts {
    f32x4x4 view_to_world;
}

[vk_binding(0, 0)]
ConstantBuffer<Consts, CDataLayout> consts;
[vk_binding(1, 0)]
ConstantBuffer<ViewConsts, CDataLayout> view_consts;
[vk_binding(2,0)]
[vk_image_format("rgba8")]
RWTexture2D<float4> backbuffer;


[shader("compute")]
[numthreads(8,8,1)]
void draw(uint3 global_thread_id: SV_DispatchThreadID) {
    uint2 pixel = global_thread_id.xy;
    uint2 backbuffer_dim;
    backbuffer.GetDimensions(backbuffer_dim.x, backbuffer_dim.y);

    f32x2 pixel_center_ndc = 2.0 * (((f32x2)pixel + f32x2(0.5,0.5)) / ((f32x2)backbuffer_dim)) - f32x2(1.0,1.0);

    f32 fov_y = radians(60.0);
    f32 s = tan(fov_y*0.5);
    f32 aspect_ratio = (f32)backbuffer_dim.x / (f32)backbuffer_dim.y;

    f32x3 view_dir = s * f32x3(pixel_center_ndc.x / aspect_ratio, pixel_center_ndc.y, 1.0);
    f32x3 world_view_dir = normalize(mul(view_consts.view_to_world, f32x4(view_dir, 0.0)).xyz);
    f32x3 world_view_origin = mul(view_consts.view_to_world, f32x4(0.xxx, 1.0)).xyz;

    // (o + s * d - po) . n = 0
    PlateInfo p = consts.plates[0];
    f32 t = dot(p.origin - world_view_origin, p.n) / dot(p.n, world_view_dir);
    f32x4 color = f32x4(1,0,1,1);
    if (t >= 0.0) {
        f32x3 intersection = world_view_origin + world_view_dir * t;
        f32x2 uv = f32x2(dot(intersection - p.origin, p.u), dot(intersection - p.origin, p.v));
        if (all(0.xx <= uv) && all(uv <= p.dim)) {
            color = f32x4(0.5+0.5*sin(uv.y*10.0),1,0,1);
        }
    }
    backbuffer[pixel] = color;
}
