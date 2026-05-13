#include <CL/cl.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {
struct Options {
    cl_device_type deviceType = CL_DEVICE_TYPE_GPU;
    uint32_t workItems = 1u << 20; // 1,048,576
    uint32_t iters = 1u << 20;     // 1,048,576
    uint32_t chunkIters = 4096;
    uint32_t warmup = 3;
    uint32_t runs = 5;
    uint32_t localSize = 256;
    uint32_t accum = 4;
};

bool parseArgU32(const char* arg, const char* name, uint32_t& out) {
    size_t nameLen = std::strlen(name);
    if (std::strncmp(arg, name, nameLen) != 0) {
        return false;
    }
    if (arg[nameLen] != '=') {
        return false;
    }
    const char* value = arg + nameLen + 1;
    char* end = nullptr;
    unsigned long parsed = std::strtoul(value, &end, 10);
    if (end == value || *end != '\0') {
        return false;
    }
    out = static_cast<uint32_t>(parsed);
    return true;
}

Options parseOptions(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (std::strcmp(arg, "--cpu") == 0) {
            opt.deviceType = CL_DEVICE_TYPE_CPU;
            continue;
        }
        if (std::strcmp(arg, "--gpu") == 0) {
            opt.deviceType = CL_DEVICE_TYPE_GPU;
            continue;
        }
        if (parseArgU32(arg, "--work-items", opt.workItems)) {
            continue;
        }
        if (parseArgU32(arg, "--iters", opt.iters)) {
            continue;
        }
        if (parseArgU32(arg, "--chunk-iters", opt.chunkIters)) {
            continue;
        }
        if (parseArgU32(arg, "--warmup", opt.warmup)) {
            continue;
        }
        if (parseArgU32(arg, "--runs", opt.runs)) {
            continue;
        }
        if (parseArgU32(arg, "--local-size", opt.localSize)) {
            continue;
        }
        if (parseArgU32(arg, "--accum", opt.accum)) {
            continue;
        }
        std::cerr << "Unknown argument: " << arg << "\n";
        std::exit(1);
    }
        if (!(opt.accum == 1 || opt.accum == 2 || opt.accum == 4 || opt.accum == 8 ||
                    opt.accum == 16 || opt.accum == 32 || opt.accum == 64 || opt.accum == 128)) {
                std::cerr << "accum must be 1, 2, 4, 8, 16, 32, 64, or 128.\n";
        std::exit(1);
    }
    return opt;
}

const char* kernelSrc = R"CLC(
#ifndef ACCUM
#define ACCUM 4
#endif

__kernel void fma_bench(__global float* out, uint iters, uint workItems, float a, float b) {
    uint gid = get_global_id(0);
    if (gid >= workItems) {
        return;
    }
    float x0 = out[gid];
#if ACCUM >= 2
    float x1 = x0 + 0.1f;
#endif
#if ACCUM >= 4
    float x2 = x0 + 0.2f;
    float x3 = x0 + 0.3f;
#endif
#if ACCUM >= 8
    float x4 = x0 + 0.4f;
    float x5 = x0 + 0.5f;
    float x6 = x0 + 0.6f;
    float x7 = x0 + 0.7f;
#endif
#if ACCUM >= 16
    float x8 = x0 + 0.8f;
    float x9 = x0 + 0.9f;
    float x10 = x0 + 1.0f;
    float x11 = x0 + 1.1f;
    float x12 = x0 + 1.2f;
    float x13 = x0 + 1.3f;
    float x14 = x0 + 1.4f;
    float x15 = x0 + 1.5f;
#endif
#if ACCUM >= 32
    float x16 = x0 + 1.6f;
    float x17 = x0 + 1.7f;
    float x18 = x0 + 1.8f;
    float x19 = x0 + 1.9f;
    float x20 = x0 + 2.0f;
    float x21 = x0 + 2.1f;
    float x22 = x0 + 2.2f;
    float x23 = x0 + 2.3f;
    float x24 = x0 + 2.4f;
    float x25 = x0 + 2.5f;
    float x26 = x0 + 2.6f;
    float x27 = x0 + 2.7f;
    float x28 = x0 + 2.8f;
    float x29 = x0 + 2.9f;
    float x30 = x0 + 3.0f;
    float x31 = x0 + 3.1f;
#endif
#if ACCUM >= 64
    float x32 = x0 + 3.2f;
    float x33 = x0 + 3.3f;
    float x34 = x0 + 3.4f;
    float x35 = x0 + 3.5f;
    float x36 = x0 + 3.6f;
    float x37 = x0 + 3.7f;
    float x38 = x0 + 3.8f;
    float x39 = x0 + 3.9f;
    float x40 = x0 + 4.0f;
    float x41 = x0 + 4.1f;
    float x42 = x0 + 4.2f;
    float x43 = x0 + 4.3f;
    float x44 = x0 + 4.4f;
    float x45 = x0 + 4.5f;
    float x46 = x0 + 4.6f;
    float x47 = x0 + 4.7f;
    float x48 = x0 + 4.8f;
    float x49 = x0 + 4.9f;
    float x50 = x0 + 5.0f;
    float x51 = x0 + 5.1f;
    float x52 = x0 + 5.2f;
    float x53 = x0 + 5.3f;
    float x54 = x0 + 5.4f;
    float x55 = x0 + 5.5f;
    float x56 = x0 + 5.6f;
    float x57 = x0 + 5.7f;
    float x58 = x0 + 5.8f;
    float x59 = x0 + 5.9f;
    float x60 = x0 + 6.0f;
    float x61 = x0 + 6.1f;
    float x62 = x0 + 6.2f;
    float x63 = x0 + 6.3f;
#endif
#if ACCUM >= 128
    float x64 = x0 + 6.4f;
    float x65 = x0 + 6.5f;
    float x66 = x0 + 6.6f;
    float x67 = x0 + 6.7f;
    float x68 = x0 + 6.8f;
    float x69 = x0 + 6.9f;
    float x70 = x0 + 7.0f;
    float x71 = x0 + 7.1f;
    float x72 = x0 + 7.2f;
    float x73 = x0 + 7.3f;
    float x74 = x0 + 7.4f;
    float x75 = x0 + 7.5f;
    float x76 = x0 + 7.6f;
    float x77 = x0 + 7.7f;
    float x78 = x0 + 7.8f;
    float x79 = x0 + 7.9f;
    float x80 = x0 + 8.0f;
    float x81 = x0 + 8.1f;
    float x82 = x0 + 8.2f;
    float x83 = x0 + 8.3f;
    float x84 = x0 + 8.4f;
    float x85 = x0 + 8.5f;
    float x86 = x0 + 8.6f;
    float x87 = x0 + 8.7f;
    float x88 = x0 + 8.8f;
    float x89 = x0 + 8.9f;
    float x90 = x0 + 9.0f;
    float x91 = x0 + 9.1f;
    float x92 = x0 + 9.2f;
    float x93 = x0 + 9.3f;
    float x94 = x0 + 9.4f;
    float x95 = x0 + 9.5f;
    float x96 = x0 + 9.6f;
    float x97 = x0 + 9.7f;
    float x98 = x0 + 9.8f;
    float x99 = x0 + 9.9f;
    float x100 = x0 + 10.0f;
    float x101 = x0 + 10.1f;
    float x102 = x0 + 10.2f;
    float x103 = x0 + 10.3f;
    float x104 = x0 + 10.4f;
    float x105 = x0 + 10.5f;
    float x106 = x0 + 10.6f;
    float x107 = x0 + 10.7f;
    float x108 = x0 + 10.8f;
    float x109 = x0 + 10.9f;
    float x110 = x0 + 11.0f;
    float x111 = x0 + 11.1f;
    float x112 = x0 + 11.2f;
    float x113 = x0 + 11.3f;
    float x114 = x0 + 11.4f;
    float x115 = x0 + 11.5f;
    float x116 = x0 + 11.6f;
    float x117 = x0 + 11.7f;
    float x118 = x0 + 11.8f;
    float x119 = x0 + 11.9f;
    float x120 = x0 + 12.0f;
    float x121 = x0 + 12.1f;
    float x122 = x0 + 12.2f;
    float x123 = x0 + 12.3f;
    float x124 = x0 + 12.4f;
    float x125 = x0 + 12.5f;
    float x126 = x0 + 12.6f;
    float x127 = x0 + 12.7f;
#endif

    for (uint i = 0; i < iters; ++i) {
        x0 = fma(x0, a, b);
#if ACCUM >= 2
        x1 = fma(x1, a, b);
#endif
#if ACCUM >= 4
        x2 = fma(x2, a, b);
        x3 = fma(x3, a, b);
#endif
#if ACCUM >= 8
        x4 = fma(x4, a, b);
        x5 = fma(x5, a, b);
        x6 = fma(x6, a, b);
        x7 = fma(x7, a, b);
#endif
#if ACCUM >= 16
        x8 = fma(x8, a, b);
        x9 = fma(x9, a, b);
        x10 = fma(x10, a, b);
        x11 = fma(x11, a, b);
        x12 = fma(x12, a, b);
        x13 = fma(x13, a, b);
        x14 = fma(x14, a, b);
        x15 = fma(x15, a, b);
#endif
#if ACCUM >= 32
        x16 = fma(x16, a, b);
        x17 = fma(x17, a, b);
        x18 = fma(x18, a, b);
        x19 = fma(x19, a, b);
        x20 = fma(x20, a, b);
        x21 = fma(x21, a, b);
        x22 = fma(x22, a, b);
        x23 = fma(x23, a, b);
        x24 = fma(x24, a, b);
        x25 = fma(x25, a, b);
        x26 = fma(x26, a, b);
        x27 = fma(x27, a, b);
        x28 = fma(x28, a, b);
        x29 = fma(x29, a, b);
        x30 = fma(x30, a, b);
        x31 = fma(x31, a, b);
#endif
    #if ACCUM >= 64
        x32 = fma(x32, a, b);
        x33 = fma(x33, a, b);
        x34 = fma(x34, a, b);
        x35 = fma(x35, a, b);
        x36 = fma(x36, a, b);
        x37 = fma(x37, a, b);
        x38 = fma(x38, a, b);
        x39 = fma(x39, a, b);
        x40 = fma(x40, a, b);
        x41 = fma(x41, a, b);
        x42 = fma(x42, a, b);
        x43 = fma(x43, a, b);
        x44 = fma(x44, a, b);
        x45 = fma(x45, a, b);
        x46 = fma(x46, a, b);
        x47 = fma(x47, a, b);
        x48 = fma(x48, a, b);
        x49 = fma(x49, a, b);
        x50 = fma(x50, a, b);
        x51 = fma(x51, a, b);
        x52 = fma(x52, a, b);
        x53 = fma(x53, a, b);
        x54 = fma(x54, a, b);
        x55 = fma(x55, a, b);
        x56 = fma(x56, a, b);
        x57 = fma(x57, a, b);
        x58 = fma(x58, a, b);
        x59 = fma(x59, a, b);
        x60 = fma(x60, a, b);
        x61 = fma(x61, a, b);
        x62 = fma(x62, a, b);
        x63 = fma(x63, a, b);
    #endif
    #if ACCUM >= 128
        x64 = fma(x64, a, b);
        x65 = fma(x65, a, b);
        x66 = fma(x66, a, b);
        x67 = fma(x67, a, b);
        x68 = fma(x68, a, b);
        x69 = fma(x69, a, b);
        x70 = fma(x70, a, b);
        x71 = fma(x71, a, b);
        x72 = fma(x72, a, b);
        x73 = fma(x73, a, b);
        x74 = fma(x74, a, b);
        x75 = fma(x75, a, b);
        x76 = fma(x76, a, b);
        x77 = fma(x77, a, b);
        x78 = fma(x78, a, b);
        x79 = fma(x79, a, b);
        x80 = fma(x80, a, b);
        x81 = fma(x81, a, b);
        x82 = fma(x82, a, b);
        x83 = fma(x83, a, b);
        x84 = fma(x84, a, b);
        x85 = fma(x85, a, b);
        x86 = fma(x86, a, b);
        x87 = fma(x87, a, b);
        x88 = fma(x88, a, b);
        x89 = fma(x89, a, b);
        x90 = fma(x90, a, b);
        x91 = fma(x91, a, b);
        x92 = fma(x92, a, b);
        x93 = fma(x93, a, b);
        x94 = fma(x94, a, b);
        x95 = fma(x95, a, b);
        x96 = fma(x96, a, b);
        x97 = fma(x97, a, b);
        x98 = fma(x98, a, b);
        x99 = fma(x99, a, b);
        x100 = fma(x100, a, b);
        x101 = fma(x101, a, b);
        x102 = fma(x102, a, b);
        x103 = fma(x103, a, b);
        x104 = fma(x104, a, b);
        x105 = fma(x105, a, b);
        x106 = fma(x106, a, b);
        x107 = fma(x107, a, b);
        x108 = fma(x108, a, b);
        x109 = fma(x109, a, b);
        x110 = fma(x110, a, b);
        x111 = fma(x111, a, b);
        x112 = fma(x112, a, b);
        x113 = fma(x113, a, b);
        x114 = fma(x114, a, b);
        x115 = fma(x115, a, b);
        x116 = fma(x116, a, b);
        x117 = fma(x117, a, b);
        x118 = fma(x118, a, b);
        x119 = fma(x119, a, b);
        x120 = fma(x120, a, b);
        x121 = fma(x121, a, b);
        x122 = fma(x122, a, b);
        x123 = fma(x123, a, b);
        x124 = fma(x124, a, b);
        x125 = fma(x125, a, b);
        x126 = fma(x126, a, b);
        x127 = fma(x127, a, b);
    #endif
    }

    float sum = x0;
#if ACCUM >= 2
    sum += x1;
#endif
#if ACCUM >= 4
    sum += x2 + x3;
#endif
#if ACCUM >= 8
    sum += x4 + x5 + x6 + x7;
#endif
#if ACCUM >= 16
    sum += x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15;
#endif
#if ACCUM >= 32
    sum += x16 + x17 + x18 + x19 + x20 + x21 + x22 + x23 + x24 + x25 + x26 + x27 + x28 + x29 +
           x30 + x31;
#endif
#if ACCUM >= 64
    sum += x32 + x33 + x34 + x35 + x36 + x37 + x38 + x39 + x40 + x41 + x42 + x43 + x44 + x45 +
           x46 + x47 + x48 + x49 + x50 + x51 + x52 + x53 + x54 + x55 + x56 + x57 + x58 + x59 +
           x60 + x61 + x62 + x63;
#endif
#if ACCUM >= 128
    sum += x64 + x65 + x66 + x67 + x68 + x69 + x70 + x71 + x72 + x73 + x74 + x75 + x76 + x77 +
           x78 + x79 + x80 + x81 + x82 + x83 + x84 + x85 + x86 + x87 + x88 + x89 + x90 + x91 +
           x92 + x93 + x94 + x95 + x96 + x97 + x98 + x99 + x100 + x101 + x102 + x103 + x104 +
           x105 + x106 + x107 + x108 + x109 + x110 + x111 + x112 + x113 + x114 + x115 + x116 +
           x117 + x118 + x119 + x120 + x121 + x122 + x123 + x124 + x125 + x126 + x127;
#endif
    out[gid] = sum;
}
)CLC";

void check(cl_int status, const char* msg) {
    if (status != CL_SUCCESS) {
        std::cerr << msg << " (error " << status << ")\n";
        std::exit(1);
    }
}

cl_device_id pickDevice(cl_platform_id platform, cl_device_type type) {
    cl_uint count = 0;
    cl_int status = clGetDeviceIDs(platform, type, 0, nullptr, &count);
    if (status != CL_SUCCESS || count == 0) {
        return nullptr;
    }
    std::vector<cl_device_id> devices(count);
    check(clGetDeviceIDs(platform, type, count, devices.data(), nullptr), "clGetDeviceIDs");
    return devices[0];
}

void printDevice(cl_device_id device) {
    char name[256] = {};
    check(clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, nullptr), "clGetDeviceInfo");
    std::cout << "Device: " << name << "\n";
}
} // namespace

int main(int argc, char** argv) {
    Options opt = parseOptions(argc, argv);

    cl_uint platformCount = 0;
    check(clGetPlatformIDs(0, nullptr, &platformCount), "clGetPlatformIDs");
    if (platformCount == 0) {
        std::cerr << "No OpenCL platforms found.\n";
        return 1;
    }
    std::vector<cl_platform_id> platforms(platformCount);
    check(clGetPlatformIDs(platformCount, platforms.data(), nullptr), "clGetPlatformIDs");

    cl_device_id device = nullptr;
    for (cl_platform_id platform : platforms) {
        device = pickDevice(platform, opt.deviceType);
        if (device) {
            break;
        }
    }
    if (!device) {
        std::cerr << "No matching OpenCL device found.\n";
        return 1;
    }

    printDevice(device);

    cl_int status = CL_SUCCESS;
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &status);
    check(status, "clCreateContext");

    cl_command_queue queue = nullptr;
#if defined(CL_VERSION_2_0)
    cl_queue_properties props[] = {0};
    queue = clCreateCommandQueueWithProperties(context, device, props, &status);
    check(status, "clCreateCommandQueueWithProperties");
#else
    queue = clCreateCommandQueue(context, device, 0, &status);
    check(status, "clCreateCommandQueue");
#endif

    if (opt.chunkIters == 0) {
        std::cerr << "chunk-iters must be > 0.\n";
        return 1;
    }

    size_t global = opt.workItems;
    size_t local = opt.localSize;
    if (global % local != 0) {
        global = (global + local - 1) / local * local;
    }

    const size_t outBytes = static_cast<size_t>(global) * sizeof(float);
    cl_mem outBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, outBytes, nullptr, &status);
    check(status, "clCreateBuffer");

    cl_program program = clCreateProgramWithSource(context, 1, &kernelSrc, nullptr, &status);
    check(status, "clCreateProgramWithSource");

    std::string buildOptions =
        "-cl-fast-relaxed-math -cl-mad-enable -cl-no-signed-zeros -DACCUM=" +
        std::to_string(opt.accum);
    status = clBuildProgram(program, 1, &device, buildOptions.c_str(), nullptr, nullptr);
    if (status != CL_SUCCESS) {
        size_t logSize = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::string log(logSize, '\0');
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
        std::cerr << "Build failed:\n" << log << "\n";
        return 1;
    }

    cl_kernel kernel = clCreateKernel(program, "fma_bench", &status);
    check(status, "clCreateKernel");

    float a = 1.0001f;
    float b = 0.9999f;
    cl_uint workItemsU = opt.workItems;
    check(clSetKernelArg(kernel, 0, sizeof(cl_mem), &outBuffer), "clSetKernelArg");
    check(clSetKernelArg(kernel, 2, sizeof(cl_uint), &workItemsU), "clSetKernelArg");
    check(clSetKernelArg(kernel, 3, sizeof(float), &a), "clSetKernelArg");
    check(clSetKernelArg(kernel, 4, sizeof(float), &b), "clSetKernelArg");

    auto runOnce = [&](uint32_t iters) {
        uint32_t remaining = iters;
        while (remaining > 0) {
            uint32_t chunk = remaining > opt.chunkIters ? opt.chunkIters : remaining;
            check(clSetKernelArg(kernel, 1, sizeof(cl_uint), &chunk), "clSetKernelArg");
            check(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, &local, 0, nullptr,
                                         nullptr),
                  "clEnqueueNDRangeKernel");
            remaining -= chunk;
        }
        check(clFinish(queue), "clFinish");
    };

    for (uint32_t i = 0; i < opt.warmup; ++i) {
        runOnce(opt.iters);
    }

    std::vector<double> times;
    times.reserve(opt.runs);
    for (uint32_t i = 0; i < opt.runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        runOnce(opt.iters);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        times.push_back(elapsed.count());
    }

    double total = 0.0;
    double best = times.empty() ? 0.0 : times[0];
    for (double t : times) {
        total += t;
        if (t < best) {
            best = t;
        }
    }
    double avg = times.empty() ? 0.0 : total / times.size();

    double ops = static_cast<double>(opt.workItems) * static_cast<double>(opt.iters) * 2.0 *
                 static_cast<double>(opt.accum);
    double gflopsAvg = ops / avg / 1e9;
    double gflopsBest = ops / best / 1e9;

    std::cout << "Work items: " << opt.workItems << "\n";
    std::cout << "Iters: " << opt.iters << "\n";
    std::cout << "Chunk iters: " << opt.chunkIters << "\n";
    std::cout << "Accumulators: " << opt.accum << "\n";
    std::cout << "Avg time: " << avg << " s\n";
    std::cout << "Best time: " << best << " s\n";
    std::cout << "Avg GFLOPS: " << gflopsAvg << "\n";
    std::cout << "Best GFLOPS: " << gflopsBest << "\n";

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(outBuffer);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
