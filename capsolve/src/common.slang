module common;

public typedef uint8_t u8;
public typedef uint16_t u16;
public typedef uint32_t u32;
public typedef uint64_t u64;
public typedef int8_t i8;
public typedef int16_t i16;
public typedef int32_t i32;
public typedef int64_t i64;
public typedef float32_t f32;

public typedef float32_t2 f32x2;
public typedef float32_t3 f32x3;
public typedef float32_t4 f32x4;
public typedef float32_t4x4 f32x4x4;

public static const f32 PI = 3.14159265358979323846264338327950288;
public static const f32 DEG_TO_RAD = PI / 180.0;



public struct PlateInfo {
    public f32x3 origin;
    public f32x3 u;
    public f32x3 v;
    public f32x3 n;
    public f32x2 dim;

    public f32x3 box_min;
    public f32x3 box_max;
}

public extern static const u32 PLATE_COUNT;

public struct Consts {
    public PlateInfo plates[PLATE_COUNT];
}
