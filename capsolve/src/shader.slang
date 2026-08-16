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
ConstantBuffer<Consts> consts;
[vk_binding(1, 0)]
RWStructuredBuffer<f32> output;

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
//extern static const bool IN;


//static const u32 PLATE_COUNT = 2;

struct Consts {
    PlateInfo plates[PLATE_COUNT];
    u32 seed;
}



uint4 pcg4d(uint4 v)
{
    v = v * 1664525u + 1013904223u;

    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;

    v = v ^ (v >> 16u);

    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;

    return v;
}

float u32_to_f32_01(uint x)
{
    return float(x >> 8u) * (1.0f / 16777216.0f);
}

struct SurfaceSample {
    f32x3 position;
    f32x3 normal;
    f32 p;
}

SurfaceSample sample_box_surface(f32x3 min, f32x3 max, uint4 rng) {
    uint4 rng = pcg4d(rng);
    f32x3 d = max - min;
    f32x3 a = f32x3(d.y * d.z, d.x * d.z, d.x * d.y);
    f32 total_area = 2.0 * (a.x + a.y + a.z);

    f32 r = u32_to_f32_01(rng.x) * (a.x + a.y + a.z);
    f32 u = u32_to_f32_01(rng.y);
    f32 v = u32_to_f32_01(rng.z);
    bool sgn = (rng.w & 1) == 0;

    SurfaceSample result;

    if (r < a.x) {
        f32 x = sgn ? max.x : min.x;
        result.position = f32x3(x, min.y + u * d.y, min.z + v * d.z);
        result.normal = f32x3(1,0,0);
    } else if (r < a.x + a.y) {
        f32 y = sgn ? max.y : min.y;
        result.position = f32x3(min.x + u * d.x, y, min.z + v * d.z);
        result.normal = f32x3(0,1,0);
    } else {
        f32 z = sgn ? max.z : min.z;
        result.position = f32x3(min.x + u * d.x, min.y + v * d.y, z);
        result.normal = f32x3(0,0,1);
    }
    result.normal = sgn ? result.normal : -result.normal;
    result.p = 1.0 / total_area;
    return result;
}

f32x3 random_dir(uint4 seed) {
    while (true) {
        uint4 rng = pcg4d(seed);
        f32 x = 2.0*u32_to_f32_01(rng.x)-1.0;
        f32 y = 2.0*u32_to_f32_01(rng.y)-1.0;
        f32 s = x * x + y * y;

        if (s > 0.0 && s < 1.0) {
            f32 a = 2.0 * sqrt(1.0 - s);

            return f32x3(x * a, y * a, 1.0 - 2.0 * s);
        }
        seed.w += 1;
    }
}

struct ClosestPoint {
    f32 sqdist;
    u32 id;
}

f32 plate_sqdist(PlateInfo plate, f32x3 sp) {
    f32x3 d = sp - plate.origin;

    f32 x = dot(d, plate.u);
    f32 y = dot(d, plate.v);
    f32 z = dot(d, plate.n);

    f32 cx = clamp(x, 0.0, plate.dim.x);
    f32 cy = clamp(y, 0.0, plate.dim.y);

    f32 dx = x - cx;
    f32 dy = y - cy;

    return dx * dx + dy * dy + z * z;
}

ClosestPoint closest(f32x3 sp) {
    f32 closest_sqdist = plate_sqdist(consts.plates[0], sp);
    u32 closest_id = 0;
    for (u32 i = 1; i< PLATE_COUNT; i++) {
        f32 sqdist = plate_sqdist(consts.plates[i], sp);
        if (sqdist < closest_sqdist) {
            closest_sqdist = sqdist;
            closest_id = i;
        }
    }
    ClosestPoint c;
    c.id = closest_id;
    c.sqdist = closest_sqdist;
    return c;
}

static const f32 EPSILON_ZERO = 8.8541878188e-12;

static const u32 INVALID = 0xffffffff;

static const f32 SURFACE_TOL = 1e-6;

f32x3 sample_conditional_return_on_sphere(f32x3 x, f32x3 center, f32 a, f32x2 rand) {
    f32x3 v = x - center;
    f32 r = length(v);
    f32x3 e = v / r;
    f32 xi = rand.x;
    let inv_d = (1.0 - xi) / (r + a) + xi / (r - a);
    let d = 1.0 / inv_d;

    f32 cos_theta = clamp(((r * r + a * a - d * d) / (2.0 * r * a)), -1.0, 1.0);
    f32 sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    f32 phi = 2.0 * PI * rand.y;

    f32x3 helper = abs(e.z) < 0.9 ? f32x3(0,0,1) : f32x3(1,0,0);

    f32x3 t1 = normalize(cross(e, helper));
    f32x3 t2 = cross(e, t1);

    f32x3 dir = cos_theta * e + sin_theta * (cos(phi) * t1 + sin(phi) * t2);
    return center + a * dir;
}

groupshared f32 op_count;
groupshared f32 op_wave_pct;
u32 walk_on_sphere(uint4 seed, f32x3 sp, bool DEBUG, uint thread_id) {
    f32 outside_radius = 0.25;
    i32 step_count = 0;
    op_count = 0;
    op_wave_pct = 0;
    //for (i32 it = 0; it < 3000; it++) {
    while (true) {
        if (DEBUG) {
            uint4 ballot = WaveActiveBallot(true);
            uint64_t wave_mask = (uint64_t)ballot.x | ((uint64_t)ballot.y << 32);
            if (WaveIsFirstLane()) {
                op_count += 1.0;
                op_wave_pct += ((f32)countbits(wave_mask)) / 64.0;
                printf("[%u]    it = %u, lanes = %u, mask = %ullx\n", thread_id, step_count, countbits(wave_mask), wave_mask);
            }
        }
        f32 sp_d2 = dot(sp,sp);
        if (sp_d2 > 4.0 * outside_radius * outside_radius) {
            f32 return_proba = outside_radius / sqrt(sp_d2);
            uint4 rand = pcg4d(seed);
            seed.w += 1;
            if (u32_to_f32_01(rand.x) > return_proba) {
                if (DEBUG) {
                printf("[%u] EXIT: step_count = %u\n", thread_id, step_count);
                }

                return INVALID;
            }
            sp = sample_conditional_return_on_sphere(sp, 0.xxx, outside_radius, f32x2(u32_to_f32_01(rand.y), u32_to_f32_01(rand.z)));
        }
        ClosestPoint cp = closest(sp);
        if (cp.sqdist < SURFACE_TOL * SURFACE_TOL){
            if (DEBUG) {
            printf("[%u] step_count = %u\n", thread_id, step_count);
            }

            return cp.id;
        }
        let d = sqrt(cp.sqdist);
        let r = 0.999999 * d;
        sp += r * random_dir(seed);
        seed.w += 10;
        step_count += 1;
    }
    return INVALID;
}

static const u32 WORKGROUP_SIZE = 64;

groupshared f32 shared_result[WORKGROUP_SIZE * PLATE_COUNT];

struct StartPoint {
    f32x3 x;
    f32 w;
}

struct SampleResult {
    u32 sample_count;
    f32 mean[PLATE_COUNT];
    f32 m2[PLATE_COUNT];
}

void add_sample_result(inout SampleResult r, f32 values[PLATE_COUNT]) {
    r.sample_count += 1;
    for (u32 i = 0; i < PLATE_COUNT; i++) {
        f32 delta = values[i] - r.mean[i];
        r.mean[i] += delta / ((f32)r.sample_count);
        f32 delta2 = values[i] - r.mean[i];
        r.m2[i] += delta*delta2;
    }
}

SampleResult wave_merge_sample_results(SampleResult local) {
    SampleResult r;
    r.sample_count = WaveActiveSum(local.sample_count);
    if (r.sample_count == 0) { return r; }
    for (u32 i = 0; i < PLATE_COUNT; i++) {
        f32 ref = WaveReadLaneFirst(local.mean[i]);
        f32 ni = (f32)local.sample_count;
        f32 d  = local.mean[i] - ref;
        f32 ref_delta = WaveActiveSum(ni * d);

        f32 mean = ref + ref_delta / (f32)r.sample_count;
        f32 dm = local.mean[i] - mean;

        f32 m2 = WaveActiveSum(
            local.m2[i] + ni * dm * dm
        );

        r.mean[i] = mean;
        r.m2[i] = m2;
    }
    return r;
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE,1,1)]
void sample_walk(uint3 global_thread_id: SV_DispatchThreadID, uint3 group_thread_id: SV_GroupThreadID, uint3 workgroup_id: SV_GroupID) {
    bool DEBUG = (workgroup_id.x == 0);

    f32 epsilon = EPSILON_ZERO;

    uint4 seed = uint4(consts.seed, global_thread_id.x, 0, 0);

    SampleResult local_sample_result;
    local_sample_result.sample_count = 0;
    for (u32 i = 0; i < PLATE_COUNT; i++) {
        local_sample_result.mean[i] = 0.0;
        local_sample_result.m2[i] = 0.0;
    }

    for (int K = 0; K < 10; K++) {
    SurfaceSample g = sample_box_surface(consts.plates[0].box_min, consts.plates[0].box_max, seed);
    seed.z += 1;

    f32x3 x = g.position;
    f32x3 n = g.normal;
    f32 pg = g.p;

    f32 clearance = sqrt(closest(x).sqdist);
    f32 r = 0.995 * clearance;

    f32x3 dir = random_dir(seed);
    seed.z += 1;

    f32x3 x_p = x + r*dir;
    f32x3 x_m = x - r*dir;

    f32 w = -epsilon * (1.0 / pg) * (3.0 / (2.0 * r)) * dot(n, dir);

    uint id_a = walk_on_sphere(seed, x_p, DEBUG, group_thread_id.x);
    seed.z += 1;
    uint id_b = walk_on_sphere(seed, x_m, DEBUG, group_thread_id.x);
    seed.z += 1;
    if (DEBUG)
    if (WaveIsFirstLane()) {
        printf("OP count %f, avg eff = %f\n", op_count, op_wave_pct / op_count);
    }

    f32 result[PLATE_COUNT] = {};
    for (u32 i = 0; i < PLATE_COUNT; i++) { result[i] = 0.0; }
    if (id_a != INVALID) {
        result[id_a] += w;
    }
    if (id_b != INVALID) {
        result[id_b] += -w;
    }
    add_sample_result(local_sample_result, result);
    }
#if 1
    GroupMemoryBarrierWithGroupSync();

    SampleResult wave_sample_result = wave_merge_sample_results(local_sample_result);

    if (group_thread_id.x == 0) {
        for (u32 i = 0; i < PLATE_COUNT; i++) {
            output[workgroup_id.x * PLATE_COUNT + i] = wave_sample_result.mean[i];
        }

#if 0
        for (u32 i = 0; i < PLATE_COUNT; i++) {
            f32 result = 0.0;
            for (u32 j = 0; j < WORKGROUP_SIZE; j++) {
                result += shared_result[j*PLATE_COUNT + i];
            }
            output[workgroup_id.x * PLATE_COUNT + i] = result / WORKGROUP_SIZE;
        }
#endif
    }
    #endif
}
