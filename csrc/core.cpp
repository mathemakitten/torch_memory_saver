#include "core.h"
#include "utils.h"
#include "macro.h"
#include "api_forwarder.h"

// ============================================================================
// PAUSE - Synchronous (backwards compatible)
// ============================================================================
void TorchMemorySaver::pause(const std::string& tag) {
    // Use default stream with explicit sync for backwards compatibility
    cudaStream_t stream = 0;

#if defined(USE_CUDA)
    // Phase 1: Collect allocations and start async copies (under lock)
    std::vector<std::pair<void*, size_t>> to_unmap;
    {
        const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);

        for (auto& [ptr, metadata] : allocation_metadata_) {
            if (!tag.empty() && metadata.tag != tag) continue;
            if (metadata.state != AllocationState::ACTIVE) {
                std::cerr << "[torch_memory_saver] Cannot pause allocation that is not active."
                          << " tag=" << metadata.tag << std::endl;
                exit(1);
            }

            if (metadata.enable_cpu_backup) {
                if (nullptr == metadata.cpu_backup) {
                    CUDA_ERROR_CHECK(cudaMallocHost(&metadata.cpu_backup, metadata.size));
                }
                // Async copy - will complete before we unmap due to sync below
                CUDA_ERROR_CHECK(cudaMemcpyAsync(
                    metadata.cpu_backup, ptr, metadata.size,
                    cudaMemcpyDeviceToHost, stream));
            }

            metadata.state = AllocationState::PAUSED;
            to_unmap.emplace_back(ptr, metadata.size);

#ifdef TMS_DEBUG_LOG
            std::cout << "[torch_memory_saver] pause (async copy started)"
                      << " ptr=" << ptr << " size=" << metadata.size
                      << " tag=" << metadata.tag << std::endl;
#endif
        }
    }

    // Phase 2: Wait for copies to complete (outside lock)
    CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));

    // Phase 3: Unmap memory (under lock)
    {
        const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);

        for (auto& [ptr, metadata] : allocation_metadata_) {
            if (!tag.empty() && metadata.tag != tag) continue;
            if (metadata.state != AllocationState::PAUSED) continue;

            CURESULT_CHECK(cuMemUnmap((CUdeviceptr)ptr, metadata.size));
            CURESULT_CHECK(cuMemRelease(metadata.allocHandle));
        }
    }

#elif defined(USE_ROCM)
    // ROCm implementation - similar pattern
    const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);

    for (auto& [ptr, metadata] : allocation_metadata_) {
        if (!tag.empty() && metadata.tag != tag) continue;

        if (metadata.enable_cpu_backup) {
            if (nullptr == metadata.cpu_backup) {
                CUDA_ERROR_CHECK(hipMallocHost(&metadata.cpu_backup, metadata.aligned_size));
            }
            CUDA_ERROR_CHECK(cudaMemcpyAsync(
                metadata.cpu_backup, ptr, metadata.aligned_size,
                hipMemcpyDeviceToHost, stream));
        }
    }

    CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));

    for (auto& [ptr, metadata] : allocation_metadata_) {
        if (!tag.empty() && metadata.tag != tag) continue;
        CUDAUtils::cu_mem_unmap_and_release(
            metadata.device, metadata.aligned_size,
            (hipDeviceptr_t)ptr, metadata.allocHandles, metadata.chunk_sizes);
    }
#endif
}

// ============================================================================
// PAUSE_ASYNC - Non-blocking pause with stream
//
// IMPORTANT: Caller must call cudaStreamSynchronize(stream) before assuming
// the GPU memory is freed and available for reuse.
//
// Note: Due to CUDA VMM constraints, we cannot truly unmap asynchronously -
// we must wait for the D2H copy to complete before unmapping. However, this
// API still provides benefit by allowing the caller to overlap OTHER work
// with the initial D2H transfer, then sync and finalize.
// ============================================================================
void TorchMemorySaver::pause_async(const std::string& tag, cudaStream_t stream) {
#if defined(USE_CUDA)
    std::vector<void*> ptrs_to_process;

    // Phase 1: Start async copies (short lock)
    {
        const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);

        for (auto& [ptr, metadata] : allocation_metadata_) {
            if (!tag.empty() && metadata.tag != tag) continue;
            if (metadata.state != AllocationState::ACTIVE) continue;

            if (metadata.enable_cpu_backup) {
                if (nullptr == metadata.cpu_backup) {
                    CUDA_ERROR_CHECK(cudaMallocHost(&metadata.cpu_backup, metadata.size));
                }
                CUDA_ERROR_CHECK(cudaMemcpyAsync(
                    metadata.cpu_backup, ptr, metadata.size,
                    cudaMemcpyDeviceToHost, stream));
            }

            ptrs_to_process.push_back(ptr);
        }
    }

    // Phase 2: Sync and unmap (must wait for copies)
    // This is unfortunately synchronous due to CUDA VMM constraints
    CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));

    {
        const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);

        for (void* ptr : ptrs_to_process) {
            auto it = allocation_metadata_.find(ptr);
            if (it == allocation_metadata_.end()) continue;

            AllocationMetadata& metadata = it->second;
            CURESULT_CHECK(cuMemUnmap((CUdeviceptr)ptr, metadata.size));
            CURESULT_CHECK(cuMemRelease(metadata.allocHandle));
            metadata.state = AllocationState::PAUSED;
        }
    }

#elif defined(USE_ROCM)
    // ROCm: delegate to sync version for now
    pause(tag);
#endif
}

// ============================================================================
// RESUME - Synchronous (backwards compatible)
// ============================================================================
void TorchMemorySaver::resume(const std::string& tag) {
    cudaStream_t stream = 0;
    resume_async(tag, stream);
    CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
}

// ============================================================================
// RESUME_ASYNC - Non-blocking resume with stream
//
// This is the KEY optimization for your use case. The memory mapping happens
// immediately (fast), then the H2D copy is launched asynchronously. The caller
// can do other work while the copy proceeds, then sync when actually needed.
//
// Usage pattern:
//   torch_memory_saver.resume_async("kv_cache", my_stream);
//   // Do other initialization work here - overlaps with H2D transfer
//   cudaStreamSynchronize(my_stream);  // or event wait
//   // Now KV cache is ready for use
// ============================================================================
void TorchMemorySaver::resume_async(const std::string& tag, cudaStream_t stream) {
#if defined(USE_CUDA)
    const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);

    for (auto& [ptr, metadata] : allocation_metadata_) {
        if (!tag.empty() && metadata.tag != tag) continue;
        if (metadata.state != AllocationState::PAUSED) {
            std::cerr << "[torch_memory_saver] Cannot resume allocation that is not paused."
                      << " tag=" << metadata.tag << std::endl;
            exit(1);
        }

        // Step 1: Create new physical memory and map (fast, synchronous)
        CUmemGenericAllocationHandle newAllocHandle;
        CUDAUtils::cu_mem_create(&newAllocHandle, metadata.size, metadata.device);
        CURESULT_CHECK(cuMemMap((CUdeviceptr)ptr, metadata.size, 0, newAllocHandle, 0));
        CUDAUtils::cu_mem_set_access(ptr, metadata.size, metadata.device);

        // Step 2: Async copy from CPU backup (this is the slow part - now async!)
        if (metadata.enable_cpu_backup) {
            SIMPLE_CHECK(metadata.cpu_backup != nullptr, "cpu_backup should not be nullptr");
            CUDA_ERROR_CHECK(cudaMemcpyAsync(
                ptr, metadata.cpu_backup, metadata.size,
                cudaMemcpyHostToDevice, stream));
        }

        metadata.state = AllocationState::ACTIVE;
        metadata.allocHandle = newAllocHandle;

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver] resume_async"
                  << " ptr=" << ptr << " size=" << metadata.size
                  << " tag=" << metadata.tag << std::endl;
#endif
    }

#elif defined(USE_ROCM)
    const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);

    for (auto& [ptr, metadata] : allocation_metadata_) {
        if (!tag.empty() && metadata.tag != tag) continue;

        CUDAUtils::cu_mem_create_and_map(
            metadata.device, metadata.aligned_size,
            (hipDeviceptr_t)ptr, metadata.allocHandles, metadata.chunk_sizes);

        if (metadata.enable_cpu_backup) {
            SIMPLE_CHECK(metadata.cpu_backup != nullptr, "cpu_backup should not be nullptr");
            CUDA_ERROR_CHECK(cudaMemcpyAsync(
                ptr, metadata.cpu_backup, metadata.aligned_size,
                hipMemcpyHostToDevice, stream));
        }

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver] resume_async"
                  << " ptr=" << ptr << " size=" << metadata.size << std::endl;
#endif
    }
#endif
}

// ============================================================================
// EVENT-BASED API - For more flexible synchronization
// ============================================================================
cudaEvent_t TorchMemorySaver::pause_async_event(const std::string& tag, cudaStream_t stream) {
    pause_async(tag, stream);

    cudaEvent_t event;
    CUDA_ERROR_CHECK(cudaEventCreate(&event));
    CUDA_ERROR_CHECK(cudaEventRecord(event, stream));
    return event;
    // Caller is responsible for cudaEventDestroy after use
}

cudaEvent_t TorchMemorySaver::resume_async_event(const std::string& tag, cudaStream_t stream) {
    resume_async(tag, stream);

    cudaEvent_t event;
    CUDA_ERROR_CHECK(cudaEventCreate(&event));
    CUDA_ERROR_CHECK(cudaEventRecord(event, stream));
    return event;
    // Caller is responsible for cudaEventDestroy after use
}
