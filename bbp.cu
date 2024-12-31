#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <sstream>

// Thread block size
#define BLOCK_SIZE 512
#define GRID_SIZE 48*3*16
#define LENGTH (BLOCK_SIZE * GRID_SIZE)

// Inverse of a modulo 2^64
// See https://jeffhurchalla.com/2022/04/25/a-faster-multiplicative-inverse-mod-a-power-of-2/
__device__
uint64_t mult_inv(uint64_t a) {
    uint64_t x0 = (3*a)^2;
    uint64_t y = 1 - a*x0;
    uint64_t x1 = x0*(1 + y);
    y *= y;
    uint64_t x2 = x1*(1 + y);
    y *= y;
    uint64_t x3 = x2*(1 + y);
    y *= y;
    uint64_t x4 = x3*(1 + y);
    return x4;
}

// Modulo 2^64
__device__
unsigned __int128 mod_r(unsigned __int128 n) {
    return (uint64_t)n;
}

// Montgomery reduction
// Computes val * 2 ^ -64 (mod n)
__device__
unsigned __int128 redc(unsigned __int128 val, unsigned __int128 n, unsigned __int128 neg_inv) {
    unsigned __int128 m = mod_r(mod_r(val) * neg_inv);
    unsigned __int128 t = (val + m * n) >> 64;
    return t >= n ? t - n : t;
}

// Kernel that does exponentiation mod n (for n < 2^64)
// This is done using left-to-right exponentiation by squaring and Montgomery multiplication.
__device__
double mod_pow16(uint64_t e, uint64_t n) {
    // 2^64 - n^-1
    unsigned __int128 neg_inv = ~mult_inv(n) + 1;

    // 2^128 mod n
    // (a * 2^64) mod n = redc(a * (2^128 mod n))
    unsigned __int128 mont_2_128 = std::numeric_limits<unsigned __int128>::max() % n + 1;
    unsigned __int128 mont_res = redc(mont_2_128, n, neg_inv);
    unsigned __int128 mont_16 = redc(mont_2_128 << 4, n, neg_inv);

    // Bit mask
    uint64_t mask = 1;
    mask <<= 63;

    // Keep going until bit(s) of e are masked
    while (!(mask & e)) {
        mask >>= 1;
    }

    // Keep exponentiating until mask = 0
    while (mask) {
        // Square mont_res
        mont_res = redc(mont_res * mont_res, n, neg_inv);

        // Multiply by mont_16 if e & mask
        if (e & mask) {
            mont_res = redc(mont_res * mont_16, n, neg_inv);
        }

        mask >>= 1;
    }
    // Reduce back and return
    return redc(mont_res, n, neg_inv);
}

__global__
void bbp(uint64_t digit, double *partial_sums, int diff, int offset, int digit_shift) {
    // Have elements in the same block differ by gridDim to maximise identical mod_pow calls
    int index = threadIdx.x * GRID_SIZE + blockIdx.x;
    __shared__ double shared_sums[BLOCK_SIZE];

    // Multiply pi by 16^n and calculate pi*16^n mod 1
    uint64_t n = digit - 1 - digit_shift;
    int64_t exponent;
    double modulo;
    double denom;
    double partial_sum = 0;

    for (uint64_t k = index; k < (n + 14) / 3; k += LENGTH) {
        exponent = n - 3 * k;
        denom = diff * k + offset;
        if (exponent > 0) {
            // Compute 16^(n-3k) (mod denom) / denom
            modulo = mod_pow16(exponent, denom);
            partial_sum += modulo / denom;

            // mod 1 to reduce error
            if (partial_sum >= 1.0) {partial_sum -= 1.0;}
        } else {
            partial_sum += 1 / (denom * double(1ll << (-4 * exponent)));
        }
    }

    shared_sums[threadIdx.x] = partial_sum;

    __syncthreads();

    // Reduction - requires block size a power of 2
    for (int i = BLOCK_SIZE >> 1; i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            shared_sums[threadIdx.x] += shared_sums[threadIdx.x + i];
            if (shared_sums[threadIdx.x] > 1) {shared_sums[threadIdx.x] -= 1;}
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {partial_sums[blockIdx.x] = shared_sums[0];}
}

// Addition modulo 1 (assuming x, y are positive)
double fadd_mod1(double x, double y) {
    double res = x + y;
    return res > 1 ? res - 1 : res;
}

// Floor modulo 1 (ensures output is positive)
double fmod1(double x) {
    return x - floor(x);
}

int main() {
    uint64_t digit;
    std::cout << "Enter from which digit to start from: ";
    std::cin >> digit;

    // Start timer:
    auto start = std::chrono::high_resolution_clock::now();

    double *partial_sums;
    cudaMallocManaged(&partial_sums, 8 * GRID_SIZE);

    // Data about BBP formula
    int terms = 8;
    int mult[terms] = {512, 256, 128, -8, -16, -8, 1, -1};
    int diff[terms] = {8, 6, 12, 6, 12, 8, 6, 12};
    int offset[terms] = {1, 1, 3, 3, 7, 5, 5, 11};
    // We are effectively calculating 256*pi and thus need to "shift" the digit we are looking at
    int digit_shift = 2;

    double temp;
    double res = 0;

    for (int i = 0; i < terms; ++i) {
        temp = 0;
        std::cout << "Computing sum of " << mult[i];
        std::cout << " / (" << diff[i] << "k + " << offset[i] << ")"; 
        std::cout << " * (1 / 4096) ^ k ... ";
        std::cout << " [" << i+1 << "/" << terms << "]" << std::endl;
        bbp<<<GRID_SIZE, BLOCK_SIZE>>>(digit, partial_sums, diff[i], offset[i], digit_shift);
        cudaDeviceSynchronize();

        // Sum partial sums to get the total of one sum
        for (int j = 0; j < GRID_SIZE; ++j) {
            temp = fadd_mod1(temp, partial_sums[j]);
        }
        // Multiply by numerator and add to res
        temp *= mult[i];
        res += temp;
        res = fmod1(res);
    }

    // To write res in hexadecimal, we multiply res by 2^64 (only affecting mantissa),
    // then cast to uint64_t and convert that to hex, before padding.
    std::stringstream hex_stream;
    hex_stream << std::setfill('0') << std::setw(16);
    hex_stream << std::hex << (uint64_t)(res * pow(2, 64)) << std::endl;

    // We want to remove the last 2 digits as they are 0s
    std::string hex_string = hex_stream.str();
    hex_string.erase(14, 2);

    std::cout << "Hex digits: " << hex_string << std::endl;

    // Stop timer
    auto stop = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float> duration = stop - start;

    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

    return 0;
}