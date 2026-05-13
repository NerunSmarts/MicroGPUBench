CXX := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra

OPENCL_LIBS := -lOpenCL
VULKAN_LIBS := -lvulkan

OPENCL_BIN := opencl_bench
VULKAN_BIN := vulkan_bench

SHADER_SRC := vulkan_fma.comp
SHADER_SPV := vulkan_fma.spv
GLSLANGFLAGS := -V

all: $(OPENCL_BIN) $(VULKAN_BIN)

$(OPENCL_BIN): opencl_bench.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(OPENCL_LIBS)

$(SHADER_SPV): $(SHADER_SRC)
	glslangValidator $(GLSLANGFLAGS) $(SHADER_SRC) -o $(SHADER_SPV)

$(VULKAN_BIN): vulkan_bench.cpp $(SHADER_SPV)
	$(CXX) $(CXXFLAGS) -o $@ vulkan_bench.cpp $(VULKAN_LIBS)

clean:
	rm -f $(OPENCL_BIN) $(VULKAN_BIN) $(SHADER_SPV)

.PHONY: all clean
