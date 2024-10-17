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
uint cellseed(int x, int y, uint offset) {
    const uint period = 65536u; // 65536 = 2^16
    uint s = ((uint(y % int(period))) * period + (uint(x % int(period)))) + offset;
    if (s == 0u) s = 1u;
    return s;
}

// 计算两个点的平方距离
inline float sq_distance(float x1, float y1, float x2, float y2) {
    return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
}

// 泊松分布生成随机数
int my_rand_poisson(thread noise_prng &p, float lambda, float prod) {
    // 生成随机的 u 值
    float u = p.myrand_uniform_0_1();

    float sum = prod;
    float x = 0.0f;

    // 限制最大迭代次数避免过大值
    while (u > sum && x < 10000.0f * lambda) {
        x += 1.0f;
        prod *= lambda / x;

        // 防止 prod 过小或过大，导致数值精度问题
        if (prod < 1e-6 || prod > 1e6) {
            break;
        }

        sum += prod;
    }

    return int(x);
}

float render_pixel(
    texture2d<float, access::read> src,
    int width,
    int height,
    int stride,
    int x,
    int y,
    int num_iterations,
    float grain_radius_mean,
    float grain_radius_std,
    float sigma,
    int seed,
    device const float* lambda,
    device const float* exp_lambda,
    device const float* x_gaussian,
    device const float* y_gaussian
) {
    float inv_grain_radius_mean = ceil(1.0f / grain_radius_mean);
    float ag = 1.0f / inv_grain_radius_mean;

    // 使用 stride 计算索引，确保正确访问纹理
    int pixel_val = 0;

    // Monte Carlo iterations
    for (int i = 0; i < num_iterations; i++) {
        float x_gauss = float(x) + sigma * x_gaussian[i];
        float y_gauss = float(y) + sigma * y_gaussian[i];

        int x_start = int(floor((x_gauss - grain_radius_mean) * inv_grain_radius_mean));
        int x_end = int(ceil((x_gauss + grain_radius_mean) * inv_grain_radius_mean));
        int y_start = int(floor((y_gauss - grain_radius_mean) * inv_grain_radius_mean));
        int y_end = int(ceil((y_gauss + grain_radius_mean) * inv_grain_radius_mean));

        bool found_grain = false;  // 用于退出多层循环的标志
        // 遍历所有可能的grain位置
        for (int ix = x_start; ix <= x_end && !found_grain; ix++) {
            for (int iy = y_start; iy <= y_end && !found_grain; iy++) {
                float cell_x = ag * float(ix);
                float cell_y = ag * float(iy);

                // PRNG to generate random values
                noise_prng p = noise_prng(cellseed(ix, iy, uint(seed)));

                // 计算当前cell中的像素位置
                int px = min(max(int(round(cell_x)), 0), width - 1);
                int py = min(max(int(round(cell_y)), 0), height - 1);

                // 根据纹理值计算索引
                float cellPixelValue = src.read(uint2(px, py)).r;
                int pixelIndex = max(0, min(int(cellPixelValue * 255.1f), 255));

                // 生成泊松随机数
                int n_cell = my_rand_poisson(p, lambda[pixelIndex], exp_lambda[pixelIndex]);

                // 遍历生成的grain
                for (int k = 0; k < n_cell; k++) {
                    float xCentreGrain = cell_x + ag * p.myrand_uniform_0_1();
                    float yCentreGrain = cell_y + ag * p.myrand_uniform_0_1();

                    // 判断是否在grain的范围内
                    if (sq_distance(xCentreGrain, yCentreGrain, x_gauss, y_gauss) < grain_radius_mean * grain_radius_mean) {
                        pixel_val += 1;
                        found_grain = true;  // 设置标志，退出所有循环
                        break;  // 退出内层循环
                    }
                }
            }
        }
    }

    // 计算最终的grain值
    float grainValue = (float(pixel_val) / float(num_iterations));
    grainValue = float(min(grainValue, 1.0f));

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
        stride,                 // 步幅
        x,                      // 当前像素 x 坐标
        y,                      // 当前像素 y 坐标
        num_iterations,         // 渲染迭代次数
        grain_radius_mean,       // 颗粒半径均值
        grain_radius_std,        // 颗粒半径标准差
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
