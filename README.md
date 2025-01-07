# bbp-cuda
A hexadecimal digit extractor using a [BBP-type formula](https://en.wikipedia.org/wiki/Bailey%E2%80%93Borwein%E2%80%93Plouffe_formula) for Pi, implemented in CUDA. Works for n up to 2^60 / 3 (about 384 quadrillion).

To compile, download and extract the repository, enter the repository in a terminal, and run `nvcc --expt-relaxed-constexpr -arch=sm_XX -Xptxas -v --maxrregcount 40 bbp-cuda.cu -o bbp-cuda.out` (with `sm_XX` being the architecture of the target device). To run, enter `./bbp-cuda.out` into the terminal.

## Motivation

I wanted to practice writing CUDA, learn more about numeric computation, and try to beat [y-cruncher](http://numberworld.org/y-cruncher/)'s own digit extractor.

## Performance

I used y-cruncher, [GPUPi v3.2](https://www.overclockers.at/news/gpupi-3-is-now-official), and my own code on my desktop (5800X, 3070Ti, no overclock) to calculate a variety of digits of Pi. The other two programs were chosen as they are commonly used for benchmarking and stress-testing.

| Size | y-cruncher v0.8.5 (5800X) | GPUPi v3.2 (3070Ti) | bbp-gpu (3070Ti) |
| ---- | ------------------------- | ------------------- | ---------------- |
| 100m | 0.176                     | 0.544               | 0.204            |
| 1b   | 1.792                     | 6.039               | 0.746            |
| 10b  | 20.183                    | 108.078             | 6.487            |
| 32b  | 68.831                    | 373.081             | 21.160           |
| 100b | 225.153                   | -                   | 68.844           |
| 1t   | 2476.645                  | -                   | 736.628          |

Notably, `bbp-gpu` is 16.6x faster than GPUPi (in the 32b workload). While `bbp-gpu` is slower than y-cruncher for computing the 100 millionth digit of Pi, it is up to 236% faster than y-cruncher. However, unlike y-cruncher, which accurately calculates 32 digits of pi at a time, `bbp-gpu` is only able to calculate 8 digits accurately (if `bbp-cuda` is calculating the 1 trillionth digit - the number of accurate digits decreases with input size).

My desktop's CPU/GPU configuration is a bit arbitrary. So, I additionally ran `bbp-gpu` on the fastest Nvidia consumer GPU (currently the 4090) via vast.ai, and compared them to the median benchmark time posted on [HWBot](https://hwbot.org/) leaderboards for the fastest consumer CPU (9950X) and GPU (4090) for y-cruncher and GPUPi respectively.

| Size | y-cruncher (9950X) | GPUPi v3.2 (4090) | bbp-gpu (4090) |
| ---- | ------------------ | ----------------- | -------------- |
| 1b   | 0.327              | 1.346             | 0.333          |
| 10b  | 3.842              | -                 | 1.830          |
| 32b  | -                  | 84.515            | 5.680          |
| 100b | 41.908             | -                 | 18.131         |

The 4090 is still slower at 1B due to overhead, but is up to 125% faster than the 9950X running y-cruncher, and `bbp-gpu` is also up to 14x faster than GPUPi v3.2.

However, [this CUDA implementation](https://github.com/euphoricpoptarts/bbpCudaImplementation), which heavily utilises PTX, seems to outperform my implementation by about 80%.

## Implementation details

This program uses Huvent's 8-term BBP formula:

![image](https://github.com/user-attachments/assets/ad9d7b4d-ff5b-443c-855c-6b13cbe15141)

This formula can be used for digit extraction (see [Wikipedia](https://en.wikipedia.org/wiki/Bailey%E2%80%93Borwein%E2%80%93Plouffe_formula#BBP_digit-extraction_algorithm_for_%CF%80)). The fractional part of 16^n*pi is represented as an unsigned long long.

This program calls 8 kernels in total, one for each partial sum in Huvent's formula. Each thread in a kernel evaluates and adds some of the terms in the formula. After all threads in a block are finished, the block then reduces each thread's running total into a single number, which is sent to global memory. The CPU then adds all the numbers from each block and multiplies by an appropriate number.

The bulk of the calculation is used in calculating 16^e modulo an odd number. In order to speed this up, I implement [16-ary exponentiation](https://en.wikipedia.org/wiki/Exponentiation_by_squaring#2k-ary_method). However, instead of using a precomputed table, I use bit-shifts to quickly calculate 16^b for b <= 15.

To avoid using remainders (which is expensive), I instead do [Montgomery reduction](https://en.wikipedia.org/wiki/Montgomery_modular_multiplication#The_REDC_algorithm) with R=2^64. The subtraction step is skipped (as it is unnecessary for N < R / 4) and instead of fully computing T + mN, I only compute the upper 64 bits (as the lower 64 bits are always 0). To compute the modular inverse of the denominator modulo 2^64, I use [code from Jeff Hurchalla](https://jeffhurchalla.com/2022/04/25/a-faster-multiplicative-inverse-mod-a-power-of-2/).

In order to maximise occupancy, the `--maxrregcount 40` is necessary, as otherwise the theoretical maximum occupancy is reduced by a third.

## License

This program is licensed under the MIT License.
