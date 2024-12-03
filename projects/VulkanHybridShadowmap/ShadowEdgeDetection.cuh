#include <cuda_runtime_api.h>
#include <VersionHelpers.h>
#include "VulkanRTCommon.h"

struct ExternalMemoryObject {
	cudaStream_t m_stream;
	VkDeviceMemory m_vulkanMemory;
	VkSemaphore m_vkTimelineSemaphore;
	cudaExternalMemory_t m_cudaMemory;
	cudaExternalSemaphore_t m_cudaTimelineSemaphore;
};