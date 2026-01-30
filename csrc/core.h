#pragma once
#include <sys/types.h>
#include <stdio.h>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <string>
#include "utils.h"
#include "macro.h"

enum class AllocationState {
    // Memory is mapped and accessible
    ACTIVE,
    // Memory is unmapped and inaccessible
    PAUSED
};

struct AllocationMetadata {
      size_t size;
      CUdevice device;
  #if defined(USE_CUDA)
      CUmemGenericAllocationHandle allocHandle;
  #elif defined(USE_ROCM)
      size_t aligned_size;
      std::vector<hipMemGenericAllocationHandle_t> allocHandles;
      std::vector<size_t> chunk_sizes;
  #else
      #error "USE_PLATFORM is not set"
  #endif
      std::string tag;
      AllocationState state;
      bool enable_cpu_backup;
      void* cpu_backup;
  };

#if defined(USE_ROCM)
namespace DeviceUtils {
    // Simple function to get global device ID from local device ID
    static int get_global_device_id(hipDevice_t local_device_id) {
        // Check for HIP_VISIBLE_DEVICES environment variable
        const char* hip_visible = std::getenv("HIP_VISIBLE_DEVICES");
        
        if (hip_visible && strlen(hip_visible) > 0) {
            std::string devices_str(hip_visible);
            std::stringstream ss(devices_str);
            std::string device_str;
            std::vector<int> device_list;
            
            // Parse comma-separated device list
            while (std::getline(ss, device_str, ',')) {
                if (!device_str.empty()) {
                    device_list.push_back(std::atoi(device_str.c_str()));
                }
            }
            
            if (local_device_id < device_list.size()) {
                int global_device_id = device_list[local_device_id];
#ifdef TMS_DEBUG_LOG
                std::cout << "[torch_memory_saver.cpp] HIP_VISIBLE_DEVICES=" << hip_visible 
                        << " local_device_id=" << local_device_id 
                        << " -> global_device_id=" << global_device_id << std::endl;
#endif
                return global_device_id;
            }
        }
        
        // Fallback: return local device ID as-is
#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] No HIP_VISIBLE_DEVICES, using local_device_id=" << local_device_id << std::endl;
#endif
        return local_device_id;
    }
}
#endif 



class TorchMemorySaver {
public:
    static TorchMemorySaver& instance();

    cudaError_t malloc(void** ptr, CUdevice device, size_t size, const std::string& tag, bool enable_cpu_backup);
    cudaError_t free(void* ptr);

    void pause(const std::string& tag);
    void resume(const std::string& tag);

    void pause_async(const std::string& tag, cudaStream_t stream);
    void resume_async(const std::string& tag, cudaStream_t stream);

private:
    TorchMemorySaver();
    ~TorchMemorySaver() = default;
    TorchMemorySaver(const TorchMemorySaver&) = delete;
    TorchMemorySaver& operator=(const TorchMemorySaver&) = delete;

    std::mutex allocator_metadata_mutex_;
    std::unordered_map<void*, AllocationMetadata> allocation_metadata_;
};
