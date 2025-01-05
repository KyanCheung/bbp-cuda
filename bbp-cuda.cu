#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <sstream>

// Thread block size
#define BLOCK_SIZE 512
#define GRID_SIZE 48*3*8
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

// Montgomery reduction
// Computes val * 2 ^ -64 (mod denom)
// Requires val < denom * 2^64 (which should be the case for denom < 2^62)
__device__ 
unsigned __int128 redc(unsigned __int128 val, uint64_t denom, uint64_t neg_inv) { 
    uint64_t val_high = val >> 64;
    uint64_t val_low = val;
    uint64_t m = val_low * neg_inv;
    // Equivalent to (m * denom + val) >> 64 but without 128-bit operations
    uint64_t t = __umul64hi(m, denom) + val_high;
    if (val_low != 0) {t++;} // bit carry from lower 64 bits if necessary
    // t now ranges from 0 to 2*denom-1
    return t;
}

// Computes fractional part of 16^e / denom (for n < 2^60) and stores in uint64_t
// This is done by calculating 16^e modulo denom, multiplying by 2^64, and integer dividing.
// Modular exponentiation is done using 16-ary exponentiation and Montgomery multiplication.
// Requires denom coprime to 2^64 (i.e. denom odd)
__device__
uint64_t div_pow16(uint64_t e, uint64_t denom) {
    // 2^64 - n^-1
    uint64_t neg_inv = ~mult_inv(denom) + 1;

    // 2^128 mod n
    // (a * 2^64) mod n = redc(a * (2^128 mod n))
    unsigned __int128 mont_2_128 = std::numeric_limits<unsigned __int128>::max() % denom + 1;

    // Bit mask
    int p = (63 - __clzll(e)) & 0x3c; // Round down (63 - __clzll(e)) to nearest multiple of 4
    uint64_t mask = 0xf; // 1111 in base 2
    mask <<= p;

    // Unroll first step of loop
    // As in the first step res = 16 ^ (4 * e_masked) so mont_res = mont_pow_16
    uint64_t e_masked = (e & mask) >> p; // Get masked bits of e
    uint64_t mont_pow_16; // Montgomery form of 16 ^ (4 * e_masked)
    unsigned __int128 mont_res = redc(mont_2_128 << (4 * e_masked), denom, neg_inv);
    mask >>= 4;
    p -= 4;

    // Keep exponentiating until mask = 0
    while (mask) {
        // Take res to the power of 16
        for (int i = 0; i < 4; i++) {
            mont_res = redc(mont_res * mont_res, denom, neg_inv);
        }

        // Multiply res by 16 ^ (e_masked)
        e_masked = (e & mask) >> p;
        mont_pow_16 = redc(mont_2_128 << (4 * e_masked), denom, neg_inv);
        mont_res = redc(mont_res * mont_pow_16, denom, neg_inv);

        mask >>= 4;
        p -= 4;
    }
    // Essentially we want res * 2^64 / n (mod 2^64)
    // We perform Montmogery reduction, without the division by 2^64 step
    // Numerator equals 0 mod 2^64 and mont_res mod n => equals res*2^64 mod (n*2^64)
    unsigned __int128 m = uint64_t(uint64_t(mont_res) * neg_inv);
    unsigned __int128 numerator = mont_res + m * denom;
    // To round to nearest integer, add n/2 to the numerator
    return (numerator + (denom >> 1)) / denom;
}

__global__
void bbp(uint64_t digit, uint64_t *partial_sums, int diff, int offset, int digit_shift) {
    // Have elements in the same block differ by gridDim to maximise identical mod_pow calls
    int index = threadIdx.x * GRID_SIZE + blockIdx.x;
    __shared__ uint64_t shared_sums[BLOCK_SIZE];

    // Multiply pi by 16^n and calculate the integer part of pi*16^n*2^64 mod 2^64
    // This has the same effect as calcuating the fractional part of pi*16^n
    // and truncating the answer to 16 hex digits
    uint64_t n = digit - 1 - digit_shift;
    int64_t exponent;
    uint64_t denom;
    uint64_t partial_sum = 0;

    // Grid-stride loop
    for (uint64_t k = index; k < (n + 16) / 3; k += LENGTH) {
        exponent = n - 3 * k;
        denom = diff * k + offset;
        if (exponent > 0) {
            // Compute fractional part of 16^(n-3k) / denom
            partial_sum += div_pow16(exponent, denom);
        } else {
            unsigned __int128 res = 1;
            // res = 16^(-exponent) * 2^64
            res <<= (64 + 4 * exponent);
            // To round to nearest integer, add denom/2 to the numerator
            partial_sum += (res + (denom >> 1)) / denom;
        }
    }

    shared_sums[threadIdx.x] = partial_sum;
    __syncthreads();

    // Reduction - requires block size a power of 2
    for (int i = BLOCK_SIZE >> 1; i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            shared_sums[threadIdx.x] += shared_sums[threadIdx.x + i];
        }
        __syncthreads();
    }

    // One thread in block writes to global memory
    if (threadIdx.x == 0) {partial_sums[blockIdx.x] = shared_sums[0];}
}

int main() {
    uint64_t digit;
    std::cout << "Enter from which digit to start from: ";
    std::cin >> digit;

    // Start timer:
    auto start = std::chrono::high_resolution_clock::now();

    uint64_t *partial_sums;
    cudaMallocManaged(&partial_sums, 8 * GRID_SIZE);

    // Data about BBP formula
    int terms = 8;
    int mult[terms] = {512, 256, 128, -8, -16, -8, 1, -1};
    int diff[terms] = {8, 6, 12, 6, 12, 8, 6, 12};
    int offset[terms] = {1, 1, 3, 3, 7, 5, 5, 11};
    // We are effectively calculating 256*pi and thus need to "shift" the digit we are looking at
    int digit_shift = 2;

    uint64_t temp;
    uint64_t res = 0;

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
            temp += partial_sums[j];
        }
        // Multiply by numerator and add to res
        res += temp * mult[i];
    }

    // We want to pad the output of res to 16 characters
    // otherwise 0(s) in the most significant digit(s) disappear
    std::cout << "Hex digits: " << std::setfill('0') << std::setw(16);
    std::cout << std::hex << res << std::endl;

    // Stop timer
    auto stop = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = stop - start;

    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

    return 0;
}