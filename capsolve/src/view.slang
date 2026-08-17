struct ViewConsts {
    f32x3 camera_world_position;
    f32x3
}

[vk_binding(0, 0)]
ConstantBuffer<Consts, CDataLayout> consts;
[vk_binding(1,0)]
[vk_image_format("rgba8")]
RWTexture2D<float4> backbuffer;

[shader("compute")]
[numthreads(8,8,1)]
void draw(uint3 global_thread_id: SV_DispatchThreadID) {
    uint2 pixel = global_thread_id.xy;
    backbuffer[pixel] = float4(1,0,1,1);
}
