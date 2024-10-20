#include <metal_stdlib>
using namespace metal;

// 定义 PRNG 结构体，用于生成噪声
struct noise_prng {
    uint state;

    // 初始化 PRNG
    noise_prng(uint seed) {
        state = wang_hash(seed);
    }

    // wang_hash 用于生成伪随机数
    uint wang_hash(uint seed) {
        seed = (seed ^ 61u) ^ (seed >> 16u);
        seed *= 9u;
        seed = seed ^ (seed >> 4u);
        seed *= 668265261u;
        seed = seed ^ (seed >> 15u);
        return seed;
    }

    // 生成随机数
    uint myrand() {
        state ^= state << 13u;
        state ^= state >> 17u;
        state ^= state << 5u;
        return state;
    }

    // 生成 [0, 1] 范围内的浮点数
    float myrand_uniform_0_1() {
        return static_cast<float>(myrand()) / 4294967295.0f;
    }
};

// 生成用于 PRNG 的种子
inline uint cellseed(const int x, const int y, const uint offset) {
    const uint period = 65536u; // 65536 = 2^16
    uint s = ((uint(y % int(period))) * period + (uint(x % int(period)))) + offset;
    s |= (s == 0u); // if (s == 0u) s = 1u;
    return s;
}

// 计算两个点的平方距离
inline float sq_distance(const float x1, const float y1, const float x2, const float y2) {
    return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
}

// 泊松分布生成随机数
int my_rand_poisson(thread noise_prng &p, const float lambda, float prod) {
    // 生成随机的 u 值
    const float u = p.myrand_uniform_0_1();
    const float x_max = 10000.0f * lambda;

    float sum = prod;
    float x = 0.0f;

    while (u > sum && x < x_max) {
        x += 1.0f;
        prod *= lambda / x;
//        prod = clamp(prod, 1e-6f, 1e6f);

        sum += prod;
    }

    return int(x);
}

float render_pixel(
    texture2d<float, access::read> src,
    const int width,
    const int height,
    const int x,
    const int y,
    const int num_iterations,
    const float grain_radius_mean,
    const float sigma,
    const int seed,
    device const float* lambda,
    device const float* exp_lambda,
    device const float* x_gaussian,
    device const float* y_gaussian
) {
    const float inv_grain_radius_mean = ceil(1.0f / grain_radius_mean);
    const float ag = grain_radius_mean;
    const float grain_radius_sq = grain_radius_mean * grain_radius_mean;

    int pixel_val = 0;

    // Monte Carlo
    for (int i = 0; i < num_iterations; i++) {
        float x_gauss = float(x) + sigma * x_gaussian[i];
        float y_gauss = float(y) + sigma * y_gaussian[i];

        int x_start = int(floor((x_gauss - grain_radius_mean) * inv_grain_radius_mean));
        int x_end = int(ceil((x_gauss + grain_radius_mean) * inv_grain_radius_mean));
        int y_start = int(floor((y_gauss - grain_radius_mean) * inv_grain_radius_mean));
        int y_end = int(ceil((y_gauss + grain_radius_mean) * inv_grain_radius_mean));

        noise_prng p = noise_prng(cellseed(x_start, y_start, uint(seed)));
        float found_grain = 0.0f;  // 0 表示未找到，1 表示找到

        // 遍历所有可能的grain位置
        for (int ix = x_start; ix <= x_end; ix++) {
            for (int iy = y_start; iy <= y_end; iy++) {
                float cell_x = ag * float(ix);
                float cell_y = ag * float(iy);

                // 计算当前cell中的像素位置
                int px = clamp(int(round(cell_x)), 0, width - 1);
                int py = clamp(int(round(cell_y)), 0, height - 1);

                // 根据纹理值计算索引
                float cellPixelValue = src.read(uint2(px, py)).r;
                int pixelIndex = int(clamp(cellPixelValue * 255.1f, 0.0f, 255.0f));

                // 生成泊松随机数
                int n_cell = my_rand_poisson(p, lambda[pixelIndex], exp_lambda[pixelIndex]);

                // 遍历生成的grain
                for (int k = 0; k < n_cell; k++) {
                    float xCentreGrain = cell_x + ag * p.myrand_uniform_0_1();
                    float yCentreGrain = cell_y + ag * p.myrand_uniform_0_1();

                    // 判断是否在grain的范围内
                    float dist_check = sq_distance(xCentreGrain, yCentreGrain, x_gauss, y_gauss) < grain_radius_sq ? 1.0f : 0.0f;

                    // 更新 pixel_val 和 found_grain
                    pixel_val += int(dist_check * (1.0f - found_grain));  // 只在未找到 grain 的情况下更新 pixel_val
                    found_grain = mix(found_grain, 1.0f, dist_check);  // 一旦找到 grain，保持 found_grain = 1
                }
            }
        }
    }

    // 计算最终的grain值
    float grainValue = float(pixel_val) / float(num_iterations);
//    return clamp(grainValue, 0.0f, 1.0f);
    return grainValue;
}

// Metal 计算着色器内核
kernel void film_grain_rendering_kernel(
    texture2d<float, access::read> src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    constant int& width [[buffer(0)]],
    constant int& height [[buffer(1)]],
    constant int& stride [[buffer(2)]],
    constant int& num_iterations [[buffer(3)]],
    constant float& grain_radius_mean [[buffer(4)]],
    constant float& grain_radius_std [[buffer(5)]],
    constant float& sigma [[buffer(6)]],
    constant int& seed [[buffer(7)]],
    device const float* lambda [[buffer(8)]],
    device const float* exp_lambda [[buffer(9)]],
    device const float* x_gaussian [[buffer(10)]],
    device const float* y_gaussian [[buffer(11)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int x = int(gid.x);
    int y = int(gid.y);

    if (x >= width || y >= height) {
        return;
    }

    // 渲染单个像素
    float pixelValue = render_pixel(
        src,                    // 纹理
        width,                  // 图像宽度
        height,                 // 图像高度
        x,                      // 当前像素 x 坐标
        y,                      // 当前像素 y 坐标
        num_iterations,         // 渲染迭代次数
        grain_radius_mean,      // 颗粒半径均值
        sigma,                  // 高斯分布的 sigma
        seed,                   // 随机种子
        lambda,                 // 泊松分布参数
        exp_lambda,             // 泊松分布的指数衰减值
        x_gaussian,             // 高斯分布的 x 方向偏移
        y_gaussian              // 高斯分布的 y 方向偏移
    );

    // 将结果写入输出纹理
    dst.write(pixelValue, gid);
}
