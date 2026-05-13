#include <vulkan/vulkan.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {
struct Options {
    uint32_t workItems = 1u << 20; // 1,048,576
    uint32_t iters = 1u << 20;     // 1,048,576
    uint32_t chunkIters = 4096;
    uint32_t warmup = 3;
    uint32_t runs = 5;
    uint32_t localSize = 256;
    uint32_t deviceIndex = 0;
    uint32_t accum = 4;
    std::string shaderPath = "vulkan_fma.spv";
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
        if (parseArgU32(arg, "--device", opt.deviceIndex)) {
            continue;
        }
        if (parseArgU32(arg, "--accum", opt.accum)) {
            continue;
        }
        if (std::strncmp(arg, "--shader=", 9) == 0) {
            opt.shaderPath = std::string(arg + 9);
            continue;
        }
        std::cerr << "Unknown argument: " << arg << "\n";
        std::exit(1);
    }
        if (opt.accum == 0 || opt.accum > 1024) {
                std::cerr << "accum must be 1, 2, 4, 8, 16, 32, 64, or 128.\n";
        std::exit(1);
    }
    return opt;
}

void check(VkResult result, const char* msg) {
    if (result != VK_SUCCESS) {
        std::cerr << msg << " (error " << result << ")\n";
        std::exit(1);
    }
}

std::vector<char> readFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Failed to open " << path << "\n";
        std::exit(1);
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> data(size);
    if (!file.read(data.data(), size)) {
        std::cerr << "Failed to read " << path << "\n";
        std::exit(1);
    }
    return data;
}

uint32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags flags,
                        const VkPhysicalDeviceMemoryProperties& props) {
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if ((typeBits & (1u << i)) && (props.memoryTypes[i].propertyFlags & flags) == flags) {
            return i;
        }
    }
    std::cerr << "No suitable memory type found.\n";
    std::exit(1);
}
} // namespace

int main(int argc, char** argv) {
    Options opt = parseOptions(argc, argv);

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "vulkan_bench";
    appInfo.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo instInfo{};
    instInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instInfo.pApplicationInfo = &appInfo;

    VkInstance instance = VK_NULL_HANDLE;
    check(vkCreateInstance(&instInfo, nullptr, &instance), "vkCreateInstance");

    uint32_t deviceCount = 0;
    check(vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr),
          "vkEnumeratePhysicalDevices");
    if (deviceCount == 0) {
        std::cerr << "No Vulkan devices found.\n";
        return 1;
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    check(vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data()),
          "vkEnumeratePhysicalDevices");
    if (opt.deviceIndex >= deviceCount) {
        std::cerr << "Device index out of range.\n";
        return 1;
    }

    VkPhysicalDevice phys = devices[opt.deviceIndex];
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(phys, &props);
    std::cout << "Device: " << props.deviceName << "\n";

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueProps(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &queueFamilyCount, queueProps.data());

    uint32_t computeFamily = UINT32_MAX;
    for (uint32_t i = 0; i < queueFamilyCount; ++i) {
        if (queueProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            computeFamily = i;
            break;
        }
    }
    if (computeFamily == UINT32_MAX) {
        std::cerr << "No compute queue family found.\n";
        return 1;
    }

    float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{};
    queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = computeFamily;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;

    VkDeviceCreateInfo devInfo{};
    devInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    devInfo.queueCreateInfoCount = 1;
    devInfo.pQueueCreateInfos = &queueInfo;

    VkDevice device = VK_NULL_HANDLE;
    check(vkCreateDevice(phys, &devInfo, nullptr, &device), "vkCreateDevice");

    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, computeFamily, 0, &queue);

    VkPhysicalDeviceMemoryProperties memProps{};
    vkGetPhysicalDeviceMemoryProperties(phys, &memProps);

    if (opt.chunkIters == 0) {
        std::cerr << "chunk-iters must be > 0.\n";
        return 1;
    }

    uint32_t localSize = opt.localSize;
    uint32_t globalSize = opt.workItems;
    if (globalSize % localSize != 0) {
        globalSize = (globalSize + localSize - 1) / localSize * localSize;
    }
    uint32_t workgroups = globalSize / localSize;

    VkDeviceSize bufferSize = static_cast<VkDeviceSize>(globalSize) * sizeof(float);

    VkBufferCreateInfo bufInfo{};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size = bufferSize;
    bufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer = VK_NULL_HANDLE;
    check(vkCreateBuffer(device, &bufInfo, nullptr, &buffer), "vkCreateBuffer");

    VkMemoryRequirements memReq{};
    vkGetBufferMemoryRequirements(device, buffer, &memReq);

    uint32_t memType = findMemoryType(memReq.memoryTypeBits,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                      memProps);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = memType;

    VkDeviceMemory memory = VK_NULL_HANDLE;
    check(vkAllocateMemory(device, &allocInfo, nullptr, &memory), "vkAllocateMemory");
    check(vkBindBufferMemory(device, buffer, memory, 0), "vkBindBufferMemory");

    void* mapped = nullptr;
    check(vkMapMemory(device, memory, 0, bufferSize, 0, &mapped), "vkMapMemory");
    std::memset(mapped, 0, static_cast<size_t>(bufferSize));
    vkUnmapMemory(device, memory);

    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &binding;

    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    check(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &setLayout),
          "vkCreateDescriptorSetLayout");

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(uint32_t) * 2 + sizeof(float) * 2;

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &setLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushRange;

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    check(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout),
          "vkCreatePipelineLayout");

    std::vector<char> shaderCode = readFile(opt.shaderPath);
    VkShaderModuleCreateInfo shaderInfo{};
    shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderInfo.codeSize = shaderCode.size();
    shaderInfo.pCode = reinterpret_cast<const uint32_t*>(shaderCode.data());

    VkShaderModule shaderModule = VK_NULL_HANDLE;
    check(vkCreateShaderModule(device, &shaderInfo, nullptr, &shaderModule),
          "vkCreateShaderModule");

    VkSpecializationMapEntry specEntry{};
    specEntry.constantID = 0;
    specEntry.offset = 0;
    specEntry.size = sizeof(uint32_t);

    VkSpecializationInfo specInfo{};
    specInfo.mapEntryCount = 1;
    specInfo.pMapEntries = &specEntry;
    specInfo.dataSize = sizeof(uint32_t);
    specInfo.pData = &opt.accum;

    VkPipelineShaderStageCreateInfo stageInfo{};
    stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = shaderModule;
    stageInfo.pName = "main";
    stageInfo.pSpecializationInfo = &specInfo;

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = stageInfo;
    pipelineInfo.layout = pipelineLayout;

    VkPipeline pipeline = VK_NULL_HANDLE;
    check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline),
          "vkCreateComputePipelines");

    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 1;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;

    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    check(vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool),
          "vkCreateDescriptorPool");

    VkDescriptorSetAllocateInfo allocSetInfo{};
    allocSetInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocSetInfo.descriptorPool = descriptorPool;
    allocSetInfo.descriptorSetCount = 1;
    allocSetInfo.pSetLayouts = &setLayout;

    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    check(vkAllocateDescriptorSets(device, &allocSetInfo, &descriptorSet),
          "vkAllocateDescriptorSets");

    VkDescriptorBufferInfo descBufInfo{};
    descBufInfo.buffer = buffer;
    descBufInfo.offset = 0;
    descBufInfo.range = bufferSize;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = descriptorSet;
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &descBufInfo;

    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);

    VkCommandPoolCreateInfo cmdPoolInfo{};
    cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cmdPoolInfo.queueFamilyIndex = computeFamily;

    VkCommandPool commandPool = VK_NULL_HANDLE;
    check(vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &commandPool),
          "vkCreateCommandPool");

    VkCommandBufferAllocateInfo cmdAllocInfo{};
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool = commandPool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    check(vkAllocateCommandBuffers(device, &cmdAllocInfo, &commandBuffer),
          "vkAllocateCommandBuffers");

    VkCommandBufferBeginInfo cmdBeginInfo{};
    cmdBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    struct PushConstants {
        uint32_t iters;
        uint32_t workItems;
        float a;
        float b;
    };

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

    VkFence fence = VK_NULL_HANDLE;
    check(vkCreateFence(device, &fenceInfo, nullptr, &fence), "vkCreateFence");

    auto runOnce = [&](uint32_t iters) {
        uint32_t remaining = iters;
        while (remaining > 0) {
            uint32_t chunk = remaining > opt.chunkIters ? opt.chunkIters : remaining;

            check(vkResetCommandBuffer(commandBuffer, 0), "vkResetCommandBuffer");
            check(vkBeginCommandBuffer(commandBuffer, &cmdBeginInfo), "vkBeginCommandBuffer");

            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout,
                                    0, 1, &descriptorSet, 0, nullptr);

            PushConstants push{chunk, opt.workItems, 1.0001f, 0.9999f};
            vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               sizeof(PushConstants), &push);
            vkCmdDispatch(commandBuffer, workgroups, 1, 1);

            check(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer");

            check(vkResetFences(device, 1, &fence), "vkResetFences");
            VkSubmitInfo submit{};
            submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &commandBuffer;
            check(vkQueueSubmit(queue, 1, &submit, fence), "vkQueueSubmit");
            check(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");

            remaining -= chunk;
        }
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

    vkDestroyFence(device, fence, nullptr);
    vkDestroyCommandPool(device, commandPool, nullptr);
    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyShaderModule(device, shaderModule, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, setLayout, nullptr);
    vkDestroyBuffer(device, buffer, nullptr);
    vkFreeMemory(device, memory, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);

    return 0;
}
