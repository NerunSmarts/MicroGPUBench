# MicroGPUBench
FP32 compute micro-benchmark for OpenCL and Vulkan compute on AArch64.

## Build

- make

## Requirements

- OpenCL headers and libOpenCL.so
- Vulkan headers and libvulkan.so
- glslangValidator (for Vulkan shader build)

## Run

### OpenCL:

- ./opencl_bench --gpu
- ./opencl_bench --cpu
- ./opencl_bench --work-items=1048576 --iters=1048576 --chunk-iters=4096 --accum=4 --runs=5 --warmup=3

### Vulkan:

- ./vulkan_bench
- ./vulkan_bench --device=0 --work-items=1048576 --iters=1048576 --chunk-iters=4096 --accum=4 --runs=5 --warmup=3
- ./vulkan_bench --shader=path/to/custom.spv

## Notes

- Work items are rounded up to a multiple of the local size.
- The shader uses 4 accumulators by default (ACCUM=4). Each iteration is one fused
	multiply-add per accumulator, counted as 2 FLOPs each.
- OpenCL builds with fast-math flags enabled for peak throughput.
- Shader compilation uses basic SPIR-V generation flags for compatibility across
	glslangValidator versions.
- chunk-iters splits the total iterations into smaller dispatches to avoid GPU watchdog
	timeouts and device loss.
- Use --accum to select 1-1024 accumulators per work-item.
