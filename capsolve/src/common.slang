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



struct PlateInfo {
    f32x3 origin;
    f32x3 u;
    f32x3 v;
    f32x3 n;
    f32x2 dim;

    f32x3 box_min;
    f32x3 box_max;
}

extern static const u32 PLATE_COUNT;

struct Consts {
    PlateInfo plates[PLATE_COUNT];
}
