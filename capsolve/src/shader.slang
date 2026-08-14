typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;
typedef float32_t f32;

typedef float32_t2 f32x2;
typedef float32_t3 f32x3;
typedef float32_t4 f32x4;
typedef float32_t4x4 f32x4x4;

static const f32 PI = 3.14159265358979323846264338327950288;
static const f32 DEG_TO_RAD = PI / 180.0;

[vk_binding(0, 0)]
StructuredBuffer<f32> input;
[vk_binding(1, 0)]
RWStructuredBuffer<f32> output;

static const u32 TILE_SIDE = 8;

struct PlateInfo {
    f32x3 origin;
    f32x3 normal;
    f32x3 u;
    f32x3 v;
    uint2 tile_dim;
    uint2 tex_origin;
}

StructuredBuffer<PlateInfo> plate_infos;
Texture2D<f32> P;
RWTexture2D<f32> AP;

struct Param {
    uint plate_a_index;
    uint plate_b_index;
}

f32x3 cell_center(PlateInfo plate, uint2 tile, uint2 cell) {
    return plate.origin + plate.u * (tile.x * TILE_SIDE + cell.x) + plate.v * (tile.y * TILE_SIDE + cell.y);
}

[shader("compute")]
[numthreads(8,8,1)]
void kernel_(uniform Param param, uint3 group_id: SV_GroupID, uint3 thread_id: SV_GroupThreadID) {
    uint tile_a_index = group_id.x;
    uint tile_b_index = group_id.y;
    PlateInfo plate_a = plate_infos[param.plate_a_index];
    PlateInfo plate_b = plate_infos[param.plate_b_index];
    uint2 tile_a = uint2(tile_a_index % plate_a.tile_dim.x, tile_a_index / plate_a.tile_dim.x);
    uint2 tile_b = uint2(tile_b_index % plate_b.tile_dim.x, tile_b_index / plate_b.tile_dim.x);

    f32x3 position_a = cell_center(plate_a, tile_a, thread_id.xy);
    for (i32 y = 0; y < 8; y++) {
    for (i32 x = 0; x < 8; x++) {
        f32x3 position_b = cell_center(plate_b, tile_b, uint2(x,y));

    }
    }
}
