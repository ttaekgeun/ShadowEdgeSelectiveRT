/*
 * Vulkan Hybrid with Shadow mapping
 *
 * Sogang Univ, Graphics Lab
 *
 * 2024
 */

#ifdef _WIN64
#include <aclapi.h>
#include <dxgi1_2.h>
#include <windows.h>
#include <VersionHelpers.h>
#define _USE_MATH_DEFINES
#endif

#include "VulkanRTCommon.h"
#include "VulkanglTFModel.h"
#include "ShadowEdgeDetection.cuh"
#include <thread>

#include "../../base/Define.h"
#define DIR_PATH "VulkanHybridShadowmap/"

static __inline__ struct cudaExtent make_cudaExtent(size_t w, size_t h, size_t d)
{
	struct cudaExtent e;

	e.width = w;
	e.height = h;
	e.depth = d;

	return e;
}

#ifdef _WIN64
class WindowsSecurityAttributes {
protected:
	SECURITY_ATTRIBUTES m_winSecurityAttributes;
	PSECURITY_DESCRIPTOR m_winPSecurityDescriptor;

public:
	WindowsSecurityAttributes();
	SECURITY_ATTRIBUTES* operator&();
	~WindowsSecurityAttributes();
};

WindowsSecurityAttributes::WindowsSecurityAttributes() {
	m_winPSecurityDescriptor = (PSECURITY_DESCRIPTOR)calloc(
		1, SECURITY_DESCRIPTOR_MIN_LENGTH + 2 * sizeof(void**));

	PSID* ppSID =
		(PSID*)((PBYTE)m_winPSecurityDescriptor + SECURITY_DESCRIPTOR_MIN_LENGTH);
	PACL* ppACL = (PACL*)((PBYTE)ppSID + sizeof(PSID*));

	InitializeSecurityDescriptor(m_winPSecurityDescriptor,
		SECURITY_DESCRIPTOR_REVISION);

	SID_IDENTIFIER_AUTHORITY sidIdentifierAuthority =
		SECURITY_WORLD_SID_AUTHORITY;
	AllocateAndInitializeSid(&sidIdentifierAuthority, 1, SECURITY_WORLD_RID, 0, 0,
		0, 0, 0, 0, 0, ppSID);

	EXPLICIT_ACCESS explicitAccess;
	ZeroMemory(&explicitAccess, sizeof(EXPLICIT_ACCESS));
	explicitAccess.grfAccessPermissions =
		STANDARD_RIGHTS_ALL | SPECIFIC_RIGHTS_ALL;
	explicitAccess.grfAccessMode = SET_ACCESS;
	explicitAccess.grfInheritance = INHERIT_ONLY;
	explicitAccess.Trustee.TrusteeForm = TRUSTEE_IS_SID;
	explicitAccess.Trustee.TrusteeType = TRUSTEE_IS_WELL_KNOWN_GROUP;
	explicitAccess.Trustee.ptstrName = (LPTSTR)*ppSID;

	SetEntriesInAcl(1, &explicitAccess, NULL, ppACL);

	SetSecurityDescriptorDacl(m_winPSecurityDescriptor, TRUE, *ppACL, FALSE);

	m_winSecurityAttributes.nLength = sizeof(m_winSecurityAttributes);
	m_winSecurityAttributes.lpSecurityDescriptor = m_winPSecurityDescriptor;
	m_winSecurityAttributes.bInheritHandle = TRUE;
}

SECURITY_ATTRIBUTES* WindowsSecurityAttributes::operator&() {
	return &m_winSecurityAttributes;
}

WindowsSecurityAttributes::~WindowsSecurityAttributes() {
	PSID* ppSID =
		(PSID*)((PBYTE)m_winPSecurityDescriptor + SECURITY_DESCRIPTOR_MIN_LENGTH);
	PACL* ppACL = (PACL*)((PBYTE)ppSID + sizeof(PSID*));

	if (*ppSID) {
		FreeSid(*ppSID);
	}
	if (*ppACL) {
		LocalFree(*ppACL);
	}
	free(m_winPSecurityDescriptor);
}
#endif

class VulkanHybridShadowmap : public VulkanRTCommon
{
public:

#if ASSET == 0
	float zNear = 1.0f;
	float zFar = 400.0f;

	float lightFOV = 45.0f;
#elif ASSET == 1
	float zNear = 1.0f;
	float zFar = 400.0f;

	float lightFOV = 100.0f;
#elif ASSET == 2
	float zNear = 0.1f;
	float zFar = 100.0f;

	float lightFOV = 90.0f;
#endif

	struct UniformDataOffscreen {
		glm::mat4 modelMatrix;
		glm::mat4 modelViewProjectionMatrix;
		glm::mat4 modelMatrixInvTrans;
	} uniformDataOffscreen;

	struct UniformDataComposition {
		alignas(16) glm::mat4 viewInverse;
		alignas(16) glm::mat4 projInverse;
		alignas(16) glm::mat4 depthBiasMVP;
		alignas(4) uint32_t frame { 0 };
		alignas(16) glm::vec4 lightPos[1];
	} uniformDataComposition;

	struct UniformDataShadowmap {
		glm::mat4 depthMVP;
	} uniformDataShadowmap;

	struct {
		vks::Buffer shadowmap{ VK_NULL_HANDLE };
		vks::Buffer offscreen{ VK_NULL_HANDLE };
		vks::Buffer composition{ VK_NULL_HANDLE };
	} uniformBuffers;

	struct {
		VkPipeline shadowmap{ VK_NULL_HANDLE };
		VkPipeline offscreen{ VK_NULL_HANDLE };
		VkPipeline composition{ VK_NULL_HANDLE };
	} pipelines;

	struct {
		VkPipelineLayout shadowmap{ VK_NULL_HANDLE };
		VkPipelineLayout offscreen{ VK_NULL_HANDLE };
		VkPipelineLayout composition{ VK_NULL_HANDLE };
	} pipelineLayouts;

	struct {
		VkDescriptorSet shadowmap{ VK_NULL_HANDLE };
		VkDescriptorSet offscreen{ VK_NULL_HANDLE };
		VkDescriptorSet composition{ VK_NULL_HANDLE };
	} descriptorSets;

	struct {
		VkDescriptorSetLayout shadowmap{ VK_NULL_HANDLE };
		VkDescriptorSetLayout offscreen{ VK_NULL_HANDLE };
		VkDescriptorSetLayout composition{ VK_NULL_HANDLE };
	} descriptorSetLayouts;

	vkglTF::Model scene;
	vks::Texture cubeMap;

	// Framebuffers holding the deferred attachments
	struct FrameBufferAttachment {
		VkImage image;
		VkDeviceMemory mem;
		VkImageView view;
		VkFormat format;
	};
	struct FrameBufferForGeometry {
		int32_t width, height;
		VkFramebuffer frameBuffer;
		// One attachment for every component required for a deferred rendering setup
		FrameBufferAttachment position, normal, albedo, metallicRoughness, emissive;
		FrameBufferAttachment depth;
		VkRenderPass renderPass;
	} geometryFrameBuf{};

	struct FrameBufferForShadowmap {
		int32_t width, height;
		VkFramebuffer frameBuffer;
		FrameBufferAttachment depth;
		VkRenderPass renderPass;
		VkSampler depthSampler;
		VkDescriptorImageInfo descriptor;
	} shadowmapFrameBuf{};

#if defined(__ANDROID__)
	// Use a smaller size on Android for performance reasons
	const uint32_t shadowMapSize{ 1024 };
#else
	const uint32_t shadowMapSize{ 16384 };
	//const uint32_t shadowMapSize{ 8192 };
	//const uint32_t shadowMapSize{ 4096 };
	//const uint32_t shadowMapSize{ 2048 };
	//const uint32_t shadowMapSize{ 1024 };
	//const uint32_t shadowMapSize{ 512 };
	//const uint32_t shadowMapSize{ 128 };

#endif

	// Depth bias (and slope) are used to avoid shadowing artifacts
	// Constant depth bias factor (always applied)
	float depthBiasConstant = 1.25f;
	// Slope depth bias factor, applied depending on polygon's slope
	float depthBiasSlope = 1.75f;

	// One sampler for the frame buffer color attachments
	VkSampler colorSampler{ VK_NULL_HANDLE };

	std::vector<VkCommandBuffer> shadowmapCmdBuffers;
	//VkCommandBuffer offScreenCmdBuffer{ VK_NULL_HANDLE };
	std::vector<VkCommandBuffer> geometryCmdBuffers;

	VkSubmitInfo shadowmapSubmitInfo;
	VkSubmitInfo geometrySubmitInfo;
	VkSubmitInfo ligtingSubmitInfo;

	// Semaphore used to synchronize between {shadow map generation, offscreen} and final scene rendering
	AccelerationStructure bottomLevelAS{};
	AccelerationStructure topLevelAS{};

	vks::Buffer transformBuffer;

	struct GeometryNode {
		uint64_t vertexBufferDeviceAddress;
		uint64_t indexBufferDeviceAddress;
		int32_t textureIndexBaseColor;
		int32_t textureIndexOcclusion;
		int32_t textureIndexNormal;
		int32_t textureIndexMetallicRoughness;
		int32_t textureIndexEmissive;
		float reflectance;
		float refractance;
		float ior;
	};
	vks::Buffer geometryNodesBuffer;

	std::vector<VkRayTracingShaderGroupCreateInfoKHR> shaderGroups{};
	struct ShaderBindingTables {
		ShaderBindingTable raygen;
		ShaderBindingTable miss;
		ShaderBindingTable hit;
	} shaderBindingTables;

	VkPhysicalDeviceDescriptorIndexingFeaturesEXT physicalDeviceDescriptorIndexingFeatures{};


	/// <External Memory Use>
	VkSemaphore cudaUpdateDoneVkSemaphore, geometryDoneVkSemaphore, shadowDoneVkSemaphore;
	size_t shadowMapImageMemSize, shadowEdgeImageMemSize, positionImageMemSize, lightImageMemSize;
	unsigned int mipLevels = 1;

	VkImage shadowEdgeTextureImage;
	VkDeviceMemory shadowEdgeTextureImageMemory;
	VkImageView shadowEdgeTextureImageView;
	VkSampler shadowEdgeTextureSampler;

	VkImage lightTextureImage;
	VkDeviceMemory lightTextureImageMemory;
	VkImageView lightTextureImageView;
	VkSampler lightTextureSampler;
#ifdef _WIN64
	PFN_vkGetMemoryWin32HandleKHR fpGetMemoryWin32HandleKHR;
	PFN_vkGetSemaphoreWin32HandleKHR fpGetSemaphoreWin32HandleKHR;
#else
	PFN_vkGetMemoryFdKHR fpGetMemoryFdKHR = NULL;
	PFN_vkGetSemaphoreFdKHR fpGetSemaphoreFdKHR = NULL;
#endif
	/// <CUDA objects>
	cudaExternalMemory_t cudaExtMemShadowMapImageBuffer, cudaExtMemShadowEdgeImageBuffer, cudaExtMemPositionImageBuffer, cudaExtMemLightImageBuffer;
	cudaMipmappedArray_t cudaMipmappedImageArrayShadowEdge, cudaMipmappedImageArrayShadowMap, cudaMipmappedImageArrayPosition, cudaMipmappedImageArrayLight;
	std::vector<cudaSurfaceObject_t> surfaceObjectListShadowEdge, surfaceObjectListLight;
	cudaSurfaceObject_t* d_surfaceObjectListShadowEdge, *d_surfaceObjectListLight;
	cudaTextureObject_t textureObjShadowMap, textureObjPosition, textureObjLight;

	cudaExternalSemaphore_t cudaUpdateDoneSemaphore;
	cudaExternalSemaphore_t geometryDoneSemaphore;
	cudaExternalSemaphore_t shadowDoneSemaphore;
	cudaStream_t streamToRun;
	/// </<CUDA objects>>
	/// </External Memory Use>

	VulkanHybridShadowmap() : VulkanRTCommon()
	{
		title = "Sogang Univ - Vulkan Hybrid with Shadow mapping";
		camera.type = Camera::CameraType::SG_camera;
		camera.movementSpeed = 20.0f;
#ifndef __ANDROID__
		camera.rotationSpeed = 0.25f;
#endif
		camera.setPerspective(60.0f, (float)width / (float)height, 0.1f, 5000.0f);

#if ASSET == 0
#if VIEW == 0
		camera.setTranslation(glm::vec3(1.146842, 2.282518, 1.067378));
		camera.setRotation(glm::vec3(-22.524939, 58.374725, 0.000000));
#elif VIEW == 1
		camera.setTranslation(glm::vec3(-6.497121, 1.637290, -1.421643));
		camera.setRotation(glm::vec3(10.925017, -102.249245, 0.000000));
#elif VIEW == 2
		camera.setTranslation(glm::vec3(4.291043, 4.683933, -1.352913));
		camera.setRotation(glm::vec3(-20.874960, 106.026215, 0.000000));
#endif
#elif ASSET == 1
#if VIEW == 0
		camera.setTranslation(glm::vec3(1.146842, 2.282518, 1.067378));
		camera.setRotation(glm::vec3(-22.524939, 58.374725, 0.000000));
#elif VIEW == 1
		camera.setTranslation(glm::vec3(-6.497121, 1.637290, -1.421643));
		camera.setRotation(glm::vec3(10.925017, -102.249245, 0.000000));
#elif VIEW == 2
		camera.setTranslation(glm::vec3(4.291043, 4.683933, -1.352913));
		camera.setRotation(glm::vec3(-20.874960, 106.026215, 0.000000));
#endif
#elif ASSET == 2
#if VIEW == 0
		camera.setTranslation(glm::vec3(-2.039184, -2.108208, 13.222129));
		camera.setRotation(glm::vec3(9.474999, 346.226501, 0.000000));
#elif VIEW == 1
		camera.setTranslation(glm::vec3(3.786976, -1.408663, -4.825589));
		camera.setRotation(glm::vec3(2.899944, 504.574402, 0.000000));
#elif VIEW == 2
		camera.setTranslation(glm::vec3(-4.241950, 2.714610, -3.181164));
		camera.setRotation(glm::vec3(-41.275059, 595.162048, 0.000000));
#endif
#endif
		enableExtensions();

		// Buffer device address requires the 64-bit integer feature to be enabled
		enabledFeatures.shaderInt64 = VK_TRUE;

		enabledDeviceExtensions.push_back(VK_KHR_MAINTENANCE3_EXTENSION_NAME);
		enabledDeviceExtensions.push_back(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);

		/// <External Memory Use>
		enabledDeviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
		enabledDeviceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
		enabledDeviceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
#ifdef _WIN64
		enabledDeviceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
		enabledDeviceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
#else
		enabledDeviceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
		enabledDeviceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
#endif
		/// </External Memory Use>
	}

	~VulkanHybridShadowmap()
	{
		if (device) {
			vkDestroySampler(device, colorSampler, nullptr);

			// Shadow map
			vkDestroyImageView(device, shadowmapFrameBuf.depth.view, nullptr);
			vkDestroyImage(device, shadowmapFrameBuf.depth.image, nullptr);
			vkFreeMemory(device, shadowmapFrameBuf.depth.mem, nullptr);

			// Color attachments
			vkDestroyImageView(device, geometryFrameBuf.position.view, nullptr);	// position
			vkDestroyImage(device, geometryFrameBuf.position.image, nullptr);
			vkFreeMemory(device, geometryFrameBuf.position.mem, nullptr);

			vkDestroyImageView(device, geometryFrameBuf.normal.view, nullptr);		// normal
			vkDestroyImage(device, geometryFrameBuf.normal.image, nullptr);
			vkFreeMemory(device, geometryFrameBuf.normal.mem, nullptr);

			vkDestroyImageView(device, geometryFrameBuf.albedo.view, nullptr);		// albedo
			vkDestroyImage(device, geometryFrameBuf.albedo.image, nullptr);
			vkFreeMemory(device, geometryFrameBuf.albedo.mem, nullptr);

			vkDestroyImageView(device, geometryFrameBuf.metallicRoughness.view, nullptr);	// metallic roughness
			vkDestroyImage(device, geometryFrameBuf.metallicRoughness.image, nullptr);
			vkFreeMemory(device, geometryFrameBuf.metallicRoughness.mem, nullptr);

			vkDestroyImageView(device, geometryFrameBuf.emissive.view, nullptr);	// emissive
			vkDestroyImage(device, geometryFrameBuf.emissive.image, nullptr);
			vkFreeMemory(device, geometryFrameBuf.emissive.mem, nullptr);

			// Depth attachment
			vkDestroyImageView(device, geometryFrameBuf.depth.view, nullptr);
			vkDestroyImage(device, geometryFrameBuf.depth.image, nullptr);
			vkFreeMemory(device, geometryFrameBuf.depth.mem, nullptr);

			vkDestroyFramebuffer(device, shadowmapFrameBuf.frameBuffer, nullptr);
			vkDestroyFramebuffer(device, geometryFrameBuf.frameBuffer, nullptr);

			vkDestroyPipeline(device, pipelines.shadowmap, nullptr);
			vkDestroyPipeline(device, pipelines.composition, nullptr);
			vkDestroyPipeline(device, pipelines.offscreen, nullptr);

			vkDestroyPipelineLayout(device, pipelineLayouts.shadowmap, nullptr);
			vkDestroyPipelineLayout(device, pipelineLayouts.offscreen, nullptr);
			vkDestroyPipelineLayout(device, pipelineLayouts.composition, nullptr);

			vkDestroyDescriptorSetLayout(device, descriptorSetLayouts.shadowmap, nullptr);
			vkDestroyDescriptorSetLayout(device, descriptorSetLayouts.offscreen, nullptr);
			vkDestroyDescriptorSetLayout(device, descriptorSetLayouts.composition, nullptr);

			vkFreeCommandBuffers(device, cmdPool, static_cast<uint32_t>(shadowmapCmdBuffers.size()), shadowmapCmdBuffers.data());
			vkFreeCommandBuffers(device, cmdPool, static_cast<uint32_t>(geometryCmdBuffers.size()), geometryCmdBuffers.data());

			// Uniform buffers
			uniformBuffers.shadowmap.destroy();
			uniformBuffers.offscreen.destroy();
			uniformBuffers.composition.destroy();

			vkDestroyRenderPass(device, shadowmapFrameBuf.renderPass, nullptr);
			vkDestroyRenderPass(device, geometryFrameBuf.renderPass, nullptr);

			vkDestroySemaphore(device, shadowDoneVkSemaphore, nullptr);
			vkDestroySemaphore(device, geometryDoneVkSemaphore, nullptr);
			vkDestroySemaphore(device, cudaUpdateDoneVkSemaphore, nullptr);

			deleteAccelerationStructure(bottomLevelAS);
			deleteAccelerationStructure(topLevelAS);

			transformBuffer.destroy();
			geometryNodesBuffer.destroy();

			shaderBindingTables.raygen.destroy();
			shaderBindingTables.miss.destroy();
			shaderBindingTables.hit.destroy();

			/// <External Memory Use>
			CUDA_CALL(cudaFree(d_surfaceObjectListShadowEdge));
			CUDA_CALL(cudaFreeMipmappedArray(cudaMipmappedImageArrayShadowMap));
			CUDA_CALL(cudaFreeMipmappedArray(cudaMipmappedImageArrayShadowEdge));
			CUDA_CALL(cudaDestroyTextureObject(textureObjShadowMap));
			CUDA_CALL(cudaDestroyExternalMemory(cudaExtMemShadowMapImageBuffer));
			CUDA_CALL(cudaDestroyExternalSemaphore(cudaUpdateDoneSemaphore));
			CUDA_CALL(cudaDestroyExternalSemaphore(geometryDoneSemaphore));
			CUDA_CALL(cudaDestroyExternalSemaphore(shadowDoneSemaphore));
			/// </External Memory Use>
		}
	}

	virtual void getEnabledFeatures()
	{
		// Enable features required for ray tracing using feature chaining via pNext		
		enabledBufferDeviceAddresFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
		enabledBufferDeviceAddresFeatures.bufferDeviceAddress = VK_TRUE;

		enabledRayTracingPipelineFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
		enabledRayTracingPipelineFeatures.rayTracingPipeline = VK_TRUE;
		enabledRayTracingPipelineFeatures.pNext = &enabledBufferDeviceAddresFeatures;

		enabledAccelerationStructureFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
		enabledAccelerationStructureFeatures.accelerationStructure = VK_TRUE;
		enabledAccelerationStructureFeatures.pNext = &enabledRayTracingPipelineFeatures;

		physicalDeviceDescriptorIndexingFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT;
		physicalDeviceDescriptorIndexingFeatures.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
		physicalDeviceDescriptorIndexingFeatures.runtimeDescriptorArray = VK_TRUE;
		physicalDeviceDescriptorIndexingFeatures.descriptorBindingVariableDescriptorCount = VK_TRUE;
		physicalDeviceDescriptorIndexingFeatures.pNext = &enabledAccelerationStructureFeatures;

		deviceCreatepNextChain = &physicalDeviceDescriptorIndexingFeatures;

		enabledFeatures.samplerAnisotropy = VK_TRUE;
	}

	/// <External Memory Use>
#ifdef _WIN64  // For windows
	HANDLE getVkImageMemHandle(
		VkExternalMemoryHandleTypeFlagsKHR externalMemoryHandleType, VkDeviceMemory deviceMemory) {
		HANDLE handle;

		VkMemoryGetWin32HandleInfoKHR vkMemoryGetWin32HandleInfoKHR = {};
		vkMemoryGetWin32HandleInfoKHR.sType =
			VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
		vkMemoryGetWin32HandleInfoKHR.pNext = NULL;
		vkMemoryGetWin32HandleInfoKHR.memory = deviceMemory;
		vkMemoryGetWin32HandleInfoKHR.handleType =
			(VkExternalMemoryHandleTypeFlagBitsKHR)externalMemoryHandleType;

		fpGetMemoryWin32HandleKHR(device, &vkMemoryGetWin32HandleInfoKHR, &handle);
		return handle;
	}
	HANDLE getVkSemaphoreHandle(
		VkExternalSemaphoreHandleTypeFlagBitsKHR externalSemaphoreHandleType, VkSemaphore& semVkCuda) {
		HANDLE handle;

		VkSemaphoreGetWin32HandleInfoKHR vulkanSemaphoreGetWin32HandleInfoKHR = {};
		vulkanSemaphoreGetWin32HandleInfoKHR.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
		vulkanSemaphoreGetWin32HandleInfoKHR.pNext = NULL;
		vulkanSemaphoreGetWin32HandleInfoKHR.semaphore = semVkCuda;
		vulkanSemaphoreGetWin32HandleInfoKHR.handleType = externalSemaphoreHandleType;

		fpGetSemaphoreWin32HandleKHR(device, &vulkanSemaphoreGetWin32HandleInfoKHR, &handle);

		return handle;
	}
#else
	int getVkImageMemHandle(
		VkExternalMemoryHandleTypeFlagsKHR externalMemoryHandleType) {
		if (externalMemoryHandleType ==
			VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR) {
			int fd;

			VkMemoryGetFdInfoKHR vkMemoryGetFdInfoKHR = {};
			vkMemoryGetFdInfoKHR.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
			vkMemoryGetFdInfoKHR.pNext = NULL;
			vkMemoryGetFdInfoKHR.memory = textureImageMemory;
			vkMemoryGetFdInfoKHR.handleType =
				VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;

			fpGetMemoryFdKHR(device, &vkMemoryGetFdInfoKHR, &fd);

			return fd;
		}
		return -1;
	}

	int getVkSemaphoreHandle(
		VkExternalSemaphoreHandleTypeFlagBitsKHR externalSemaphoreHandleType,
		VkSemaphore& semVkCuda) {
		if (externalSemaphoreHandleType ==
			VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT) {
			int fd;

			VkSemaphoreGetFdInfoKHR vulkanSemaphoreGetFdInfoKHR = {};
			vulkanSemaphoreGetFdInfoKHR.sType =
				VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
			vulkanSemaphoreGetFdInfoKHR.pNext = NULL;
			vulkanSemaphoreGetFdInfoKHR.semaphore = semVkCuda;
			vulkanSemaphoreGetFdInfoKHR.handleType =
				VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;

			fpGetSemaphoreFdKHR(device, &vulkanSemaphoreGetFdInfoKHR, &fd);

			return fd;
		}
		return -1;
	}
#endif

	VkCommandBuffer beginSingleTimeCommands() {
		VkCommandBufferAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandPool = cmdPool;
		allocInfo.commandBufferCount = 1;

		VkCommandBuffer commandBuffer;
		vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

		VkCommandBufferBeginInfo beginInfo = {};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		vkBeginCommandBuffer(commandBuffer, &beginInfo);

		return commandBuffer;
	}

	void endSingleTimeCommands(VkCommandBuffer commandBuffer) {
		vkEndCommandBuffer(commandBuffer);

		VkSubmitInfo submitInfo = {};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;

		vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
		vkQueueWaitIdle(queue);

		vkFreeCommandBuffers(device, cmdPool, 1, &commandBuffer);
	}

	void getKhrExtensionsFn() {
#ifdef _WIN64

		fpGetSemaphoreWin32HandleKHR = (PFN_vkGetSemaphoreWin32HandleKHR)vkGetDeviceProcAddr(device, "vkGetSemaphoreWin32HandleKHR");
		if (fpGetSemaphoreWin32HandleKHR == NULL) {
			throw std::runtime_error("Vulkan: Proc address for \"vkGetSemaphoreWin32HandleKHR\" not ""found.\n");
		}
		fpGetMemoryWin32HandleKHR =
			(PFN_vkGetMemoryWin32HandleKHR)vkGetInstanceProcAddr(
				instance, "vkGetMemoryWin32HandleKHR");
		if (fpGetMemoryWin32HandleKHR == NULL) {
			throw std::runtime_error(
				"Vulkan: Proc address for \"vkGetMemoryWin32HandleKHR\" not "
				"found.\n");
	}
#else
		fpGetSemaphoreFdKHR = (PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(
			device, "vkGetSemaphoreFdKHR");
		if (fpGetSemaphoreFdKHR == NULL) {
			throw std::runtime_error(
				"Vulkan: Proc address for \"vkGetSemaphoreFdKHR\" not found.\n");
		}
		fpGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)vkGetInstanceProcAddr(
			instance, "vkGetMemoryFdKHR");
		if (fpGetMemoryFdKHR == NULL) {
			throw std::runtime_error(
				"Vulkan: Proc address for \"vkGetMemoryFdKHR\" not found.\n");
		}
		else {
			std::cout << "Vulkan proc address for vkGetMemoryFdKHR - "
				<< fpGetMemoryFdKHR << std::endl;
		}
#endif
	}
	/// </External Memory Use>

	// Create a frame buffer attachment
	void createAttachment(
		VkFormat format,
		VkImageUsageFlagBits usage,
		FrameBufferAttachment* attachment)
	{
		VkImageAspectFlags aspectMask = 0;
		VkImageLayout imageLayout;

		attachment->format = format;

		if (usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
		{
			aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		}
		if (usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
		{
			aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
			if (format >= VK_FORMAT_D16_UNORM_S8_UINT)
				aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
			imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		}

		assert(aspectMask > 0);

		VkImageCreateInfo image = vks::initializers::imageCreateInfo();
		image.imageType = VK_IMAGE_TYPE_2D;
		image.format = format;
		image.extent.width = geometryFrameBuf.width;
		image.extent.height = geometryFrameBuf.height;
		image.extent.depth = 1;
		image.mipLevels = mipLevels;
		image.arrayLayers = 1;
		image.samples = VK_SAMPLE_COUNT_1_BIT;
		image.tiling = VK_IMAGE_TILING_OPTIMAL;
		image.usage = usage | VK_IMAGE_USAGE_SAMPLED_BIT;

		VkMemoryAllocateInfo memAlloc = vks::initializers::memoryAllocateInfo();
		VkMemoryRequirements memReqs;

		VK_CHECK_RESULT(vkCreateImage(device, &image, nullptr, &attachment->image));
		vkGetImageMemoryRequirements(device, attachment->image, &memReqs);
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &attachment->mem));
		VK_CHECK_RESULT(vkBindImageMemory(device, attachment->image, attachment->mem, 0));

		VkImageViewCreateInfo imageView = vks::initializers::imageViewCreateInfo();
		imageView.viewType = VK_IMAGE_VIEW_TYPE_2D;
		imageView.format = format;
		imageView.subresourceRange = {};
		imageView.subresourceRange.aspectMask = aspectMask;
		imageView.subresourceRange.baseMipLevel = 0;
		imageView.subresourceRange.levelCount = mipLevels;
		imageView.subresourceRange.baseArrayLayer = 0;
		imageView.subresourceRange.layerCount = 1;
		imageView.image = attachment->image;
		VK_CHECK_RESULT(vkCreateImageView(device, &imageView, nullptr, &attachment->view));
	}

	/// <External Memory Use>
	void createPositionAttachment(
		VkFormat format,
		VkImageUsageFlagBits usage,
		FrameBufferAttachment* attachment)
	{
		VkImageAspectFlags aspectMask = 0;
		VkImageLayout imageLayout;

		attachment->format = format;

		if (usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
		{
			aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		}
		if (usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
		{
			aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
			if (format >= VK_FORMAT_D16_UNORM_S8_UINT)
				aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
			imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		}

		assert(aspectMask > 0);

		VkImageCreateInfo image = vks::initializers::imageCreateInfo();
		image.imageType = VK_IMAGE_TYPE_2D;
		image.format = format;
		image.extent.width = geometryFrameBuf.width;
		image.extent.height = geometryFrameBuf.height;
		image.extent.depth = 1;
		image.mipLevels = mipLevels;
		image.arrayLayers = 1;
		image.samples = VK_SAMPLE_COUNT_1_BIT;
		image.tiling = VK_IMAGE_TILING_OPTIMAL;
		image.usage = usage | VK_IMAGE_USAGE_SAMPLED_BIT;

		VkExternalMemoryImageCreateInfo vkExternalMemImageCreateInfo = {};
		vkExternalMemImageCreateInfo.sType =
			VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
		vkExternalMemImageCreateInfo.pNext = NULL;
#ifdef _WIN64
		vkExternalMemImageCreateInfo.handleTypes =
			VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
		vkExternalMemImageCreateInfo.handleTypes =
			VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
#endif
		image.pNext = &vkExternalMemImageCreateInfo;

		VkMemoryAllocateInfo memAlloc = vks::initializers::memoryAllocateInfo();
		VkMemoryRequirements memReqs;

		/// <External Memory Use>
#ifdef _WIN64
		WindowsSecurityAttributes winSecurityAttributes;

		VkExportMemoryWin32HandleInfoKHR vulkanExportMemoryWin32HandleInfoKHR = {};
		vulkanExportMemoryWin32HandleInfoKHR.sType =
			VK_STRUCTURE_TYPE_EXPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
		vulkanExportMemoryWin32HandleInfoKHR.pNext = NULL;
		vulkanExportMemoryWin32HandleInfoKHR.pAttributes = &winSecurityAttributes;
		vulkanExportMemoryWin32HandleInfoKHR.dwAccess =
			DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE;
		vulkanExportMemoryWin32HandleInfoKHR.name = (LPCWSTR)NULL;
#endif
		VkExportMemoryAllocateInfoKHR vulkanExportMemoryAllocateInfoKHR = {};
		vulkanExportMemoryAllocateInfoKHR.sType =
			VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
#ifdef _WIN64
		vulkanExportMemoryAllocateInfoKHR.pNext =
			IsWindows8OrGreater() ? &vulkanExportMemoryWin32HandleInfoKHR : NULL;
		vulkanExportMemoryAllocateInfoKHR.handleTypes =
			IsWindows8OrGreater()
			? VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT
			: VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT;
#else
		vulkanExportMemoryAllocateInfoKHR.pNext = NULL;
		vulkanExportMemoryAllocateInfoKHR.handleTypes =
			VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
#endif

		/// </External Memory Use>
		memAlloc.pNext = &vulkanExportMemoryAllocateInfoKHR;

		VK_CHECK_RESULT(vkCreateImage(device, &image, nullptr, &attachment->image));
		vkGetImageMemoryRequirements(device, attachment->image, &memReqs);
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &attachment->mem));
		VK_CHECK_RESULT(vkBindImageMemory(device, attachment->image, attachment->mem, 0));

		positionImageMemSize = memReqs.size;

		VkImageViewCreateInfo imageView = vks::initializers::imageViewCreateInfo();
		imageView.viewType = VK_IMAGE_VIEW_TYPE_2D;
		imageView.format = format;
		imageView.subresourceRange = {};
		imageView.subresourceRange.aspectMask = aspectMask;
		imageView.subresourceRange.baseMipLevel = 0;
		imageView.subresourceRange.levelCount = mipLevels;
		imageView.subresourceRange.baseArrayLayer = 0;
		imageView.subresourceRange.layerCount = 1;
		imageView.image = attachment->image;
		VK_CHECK_RESULT(vkCreateImageView(device, &imageView, nullptr, &attachment->view));
	}
	/// </External Memory Use>

	void prepareShadowmapRenderpass()
	{
		VkAttachmentDescription attachmentDescription{};
		attachmentDescription.format = shadowmapFrameBuf.depth.format;
		attachmentDescription.samples = VK_SAMPLE_COUNT_1_BIT;
		attachmentDescription.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachmentDescription.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachmentDescription.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachmentDescription.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachmentDescription.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachmentDescription.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

		VkAttachmentReference depthReference = {};
		depthReference.attachment = 0;
		depthReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpass = {};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 0;
		subpass.pDepthStencilAttachment = &depthReference;

		std::array<VkSubpassDependency, 2> dependencies;

		dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[0].dstSubpass = 0;
		dependencies[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		dependencies[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
		dependencies[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
		dependencies[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		dependencies[1].srcSubpass = 0;
		dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[1].srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
		dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		dependencies[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		VkRenderPassCreateInfo renderPassCreateInfo = vks::initializers::renderPassCreateInfo();
		renderPassCreateInfo.attachmentCount = 1;
		renderPassCreateInfo.pAttachments = &attachmentDescription;
		renderPassCreateInfo.subpassCount = 1;
		renderPassCreateInfo.pSubpasses = &subpass;
		renderPassCreateInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
		renderPassCreateInfo.pDependencies = dependencies.data();

		VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassCreateInfo, nullptr, &shadowmapFrameBuf.renderPass));
	}

	void prepareShadowmapFramebuffer()
	{
		shadowmapFrameBuf.width = shadowMapSize;
		shadowmapFrameBuf.height = shadowMapSize;

		shadowmapFrameBuf.depth.format = VK_FORMAT_D32_SFLOAT;

		// For shadow mapping we only need a depth attachment
		VkImageCreateInfo image = vks::initializers::imageCreateInfo();
		image.imageType = VK_IMAGE_TYPE_2D;
		image.extent.width = shadowmapFrameBuf.width;
		image.extent.height = shadowmapFrameBuf.height;
		image.extent.depth = 1;
		image.mipLevels = 1;
		image.arrayLayers = 1;
		image.samples = VK_SAMPLE_COUNT_1_BIT;
		image.tiling = VK_IMAGE_TILING_OPTIMAL;
		image.format = shadowmapFrameBuf.depth.format;
		image.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
		image.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		/// <External Memory Use>
		VkExternalMemoryImageCreateInfo vkExternalMemImageCreateInfo = {};
		vkExternalMemImageCreateInfo.sType =
			VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
		vkExternalMemImageCreateInfo.pNext = NULL;
#ifdef _WIN64
		vkExternalMemImageCreateInfo.handleTypes =
			VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
		vkExternalMemImageCreateInfo.handleTypes =
			VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
#endif
		image.pNext = &vkExternalMemImageCreateInfo;
		/// </External Memory Use>

		VK_CHECK_RESULT(vkCreateImage(device, &image, nullptr, &shadowmapFrameBuf.depth.image));

		VkMemoryAllocateInfo memAlloc = vks::initializers::memoryAllocateInfo();
		VkMemoryRequirements memReqs;
		vkGetImageMemoryRequirements(device, shadowmapFrameBuf.depth.image, &memReqs);
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

		/// <External Memory Use>
		shadowMapImageMemSize = memReqs.size;

#ifdef _WIN64
		WindowsSecurityAttributes winSecurityAttributes;

		VkExportMemoryWin32HandleInfoKHR vulkanExportMemoryWin32HandleInfoKHR = {};
		vulkanExportMemoryWin32HandleInfoKHR.sType =
			VK_STRUCTURE_TYPE_EXPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
		vulkanExportMemoryWin32HandleInfoKHR.pNext = NULL;
		vulkanExportMemoryWin32HandleInfoKHR.pAttributes = &winSecurityAttributes;
		vulkanExportMemoryWin32HandleInfoKHR.dwAccess =
			DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE;
		vulkanExportMemoryWin32HandleInfoKHR.name = (LPCWSTR)NULL;
#endif
		VkExportMemoryAllocateInfoKHR vulkanExportMemoryAllocateInfoKHR = {};
		vulkanExportMemoryAllocateInfoKHR.sType =
			VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
#ifdef _WIN64
		vulkanExportMemoryAllocateInfoKHR.pNext =
			IsWindows8OrGreater() ? &vulkanExportMemoryWin32HandleInfoKHR : NULL;
		vulkanExportMemoryAllocateInfoKHR.handleTypes =
			IsWindows8OrGreater()
			? VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT
			: VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT;
#else
		vulkanExportMemoryAllocateInfoKHR.pNext = NULL;
		vulkanExportMemoryAllocateInfoKHR.handleTypes =
			VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
#endif
		//generateMipmaps(shadowmapFrameBuf.depth.image, shadowmapFrameBuf.depth.format);

		/// </External Memory Use>
		memAlloc.pNext = &vulkanExportMemoryAllocateInfoKHR;

		VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &shadowmapFrameBuf.depth.mem));
		VK_CHECK_RESULT(vkBindImageMemory(device, shadowmapFrameBuf.depth.image, shadowmapFrameBuf.depth.mem, 0));

		VkImageViewCreateInfo depthStencilView = vks::initializers::imageViewCreateInfo();
		depthStencilView.viewType = VK_IMAGE_VIEW_TYPE_2D;
		depthStencilView.format = shadowmapFrameBuf.depth.format;
		depthStencilView.subresourceRange = {};
		depthStencilView.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
		depthStencilView.subresourceRange.baseMipLevel = 0;
		depthStencilView.subresourceRange.levelCount = 1;
		depthStencilView.subresourceRange.baseArrayLayer = 0;
		depthStencilView.subresourceRange.layerCount = 1;
		depthStencilView.image = shadowmapFrameBuf.depth.image;
		VK_CHECK_RESULT(vkCreateImageView(device, &depthStencilView, nullptr, &shadowmapFrameBuf.depth.view));

		// Create sampler to sample from to depth attachment
		// Used to sample in the fragment shader for shadowed rendering
		VkFilter shadowmap_filter = VK_FILTER_NEAREST;
		//vks::tools::formatIsFilterable(physicalDevice, shadowmapFrameBuf.depth.format, VK_IMAGE_TILING_OPTIMAL) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
		VkSamplerCreateInfo sampler = vks::initializers::samplerCreateInfo();
		sampler.magFilter = shadowmap_filter;
		sampler.minFilter = shadowmap_filter;
		sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		sampler.addressModeV = sampler.addressModeU;
		sampler.addressModeW = sampler.addressModeU;
		sampler.mipLodBias = 0.0f;
		sampler.maxAnisotropy = 1.0f;
		sampler.minLod = 0.0f;
		sampler.maxLod = static_cast<float>(mipLevels);
		sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
		VK_CHECK_RESULT(vkCreateSampler(device, &sampler, nullptr, &shadowmapFrameBuf.depthSampler));

		prepareShadowmapRenderpass();

		// Create frame buffer
		VkFramebufferCreateInfo fbufCreateInfo = vks::initializers::framebufferCreateInfo();
		fbufCreateInfo.renderPass = shadowmapFrameBuf.renderPass;
		fbufCreateInfo.attachmentCount = 1;
		fbufCreateInfo.pAttachments = &shadowmapFrameBuf.depth.view;
		fbufCreateInfo.width = shadowmapFrameBuf.width;
		fbufCreateInfo.height = shadowmapFrameBuf.height;
		fbufCreateInfo.layers = 1;

		VK_CHECK_RESULT(vkCreateFramebuffer(device, &fbufCreateInfo, nullptr, &shadowmapFrameBuf.frameBuffer));
	}

	// Prepare a new framebuffer and attachments for offscreen rendering (G-Buffer)
	void prepareGeometryFramebuffer()
	{
		geometryFrameBuf.width = VulkanRTBase::width;
		geometryFrameBuf.height = VulkanRTBase::height;

		// Color attachments
		{
			createPositionAttachment(									// (World space) Positions
				VK_FORMAT_R16G16B16A16_SFLOAT,
				VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
				&geometryFrameBuf.position);
			createAttachment(									// (World space) Normals
				VK_FORMAT_R16G16B16A16_SFLOAT,
				VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
				&geometryFrameBuf.normal);
			createAttachment(									// Albedo (color)
				VK_FORMAT_R8G8B8A8_UNORM,
				VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
				&geometryFrameBuf.albedo);
			createAttachment(									// MetallicRoughness
				VK_FORMAT_R8G8B8A8_UNORM,
				VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
				&geometryFrameBuf.metallicRoughness);
			createAttachment(									// Emissive
				VK_FORMAT_R8G8B8A8_UNORM,
				VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
				&geometryFrameBuf.emissive);
		}

		// Depth attachment
		VkFormat attDepthFormat;
		VkBool32 validDepthFormat = vks::tools::getSupportedDepthFormat(physicalDevice, &attDepthFormat);
		assert(validDepthFormat);
		createAttachment(
			attDepthFormat,
			VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
			&geometryFrameBuf.depth);

		// Set up separate renderpass with references to the color and depth attachments
		std::array<VkAttachmentDescription, 6> attachmentDescs = {};

		// Init attachment properties
		for (uint32_t i = 0; i < 6; ++i)
		{
			attachmentDescs[i].samples = VK_SAMPLE_COUNT_1_BIT;
			attachmentDescs[i].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
			attachmentDescs[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
			attachmentDescs[i].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
			attachmentDescs[i].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
			if (i == 5)
			{
				attachmentDescs[i].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
				attachmentDescs[i].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
			}
			else
			{
				attachmentDescs[i].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
				attachmentDescs[i].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			}
		}

		// Formats
		attachmentDescs[0].format = geometryFrameBuf.position.format;
		attachmentDescs[1].format = geometryFrameBuf.normal.format;
		attachmentDescs[2].format = geometryFrameBuf.albedo.format;
		attachmentDescs[3].format = geometryFrameBuf.metallicRoughness.format;
		attachmentDescs[4].format = geometryFrameBuf.emissive.format;
		attachmentDescs[5].format = geometryFrameBuf.depth.format;

		std::vector<VkAttachmentReference> colorReferences;
		colorReferences.push_back({ 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });
		colorReferences.push_back({ 1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });
		colorReferences.push_back({ 2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });
		colorReferences.push_back({ 3, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });
		colorReferences.push_back({ 4, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });

		VkAttachmentReference depthReference = {};
		depthReference.attachment = 5;
		depthReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpass = {};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.pColorAttachments = colorReferences.data();
		subpass.colorAttachmentCount = static_cast<uint32_t>(colorReferences.size());
		subpass.pDepthStencilAttachment = &depthReference;

		// Use subpass dependencies for attachment layout transitions
		std::array<VkSubpassDependency, 2> dependencies;

		dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[0].dstSubpass = 0;
		dependencies[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
		dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
		dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		dependencies[1].srcSubpass = 0;
		dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
		dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
		dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		VkRenderPassCreateInfo renderPassInfo = {};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.pAttachments = attachmentDescs.data();
		renderPassInfo.attachmentCount = static_cast<uint32_t>(attachmentDescs.size());
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &subpass;
		renderPassInfo.dependencyCount = 2;
		renderPassInfo.pDependencies = dependencies.data();

		VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassInfo, nullptr, &geometryFrameBuf.renderPass));

		std::array<VkImageView, 6> attachments;
		attachments[0] = geometryFrameBuf.position.view;
		attachments[1] = geometryFrameBuf.normal.view;
		attachments[2] = geometryFrameBuf.albedo.view;
		attachments[3] = geometryFrameBuf.metallicRoughness.view;
		attachments[4] = geometryFrameBuf.emissive.view;
		attachments[5] = geometryFrameBuf.depth.view;

		VkFramebufferCreateInfo fbufCreateInfo = {};
		fbufCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		fbufCreateInfo.pNext = NULL;
		fbufCreateInfo.renderPass = geometryFrameBuf.renderPass;
		fbufCreateInfo.pAttachments = attachments.data();
		fbufCreateInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
		fbufCreateInfo.width = geometryFrameBuf.width;
		fbufCreateInfo.height = geometryFrameBuf.height;
		fbufCreateInfo.layers = 1;
		VK_CHECK_RESULT(vkCreateFramebuffer(device, &fbufCreateInfo, nullptr, &geometryFrameBuf.frameBuffer));

		// Create sampler to sample from the color attachments
		VkSamplerCreateInfo sampler = vks::initializers::samplerCreateInfo();
		sampler.magFilter = VK_FILTER_NEAREST;
		sampler.minFilter = VK_FILTER_NEAREST;
		sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		sampler.addressModeV = sampler.addressModeU;
		sampler.addressModeW = sampler.addressModeU;
		sampler.mipLodBias = 0.0f;
		sampler.maxAnisotropy = 1.0f;
		sampler.minLod = 0.0f;
		sampler.maxLod = 1.0f;
		sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
		VK_CHECK_RESULT(vkCreateSampler(device, &sampler, nullptr, &colorSampler));
	}

	// [Pass 0]
	void buildShadowmapCommandBuffers()
	{
		VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		VkClearValue clearValue;
		VkViewport viewport;
		VkRect2D scissor;

		for (int32_t i = 0; i < shadowmapCmdBuffers.size(); ++i)
		{
			VK_CHECK_RESULT(vkBeginCommandBuffer(shadowmapCmdBuffers[i], &cmdBufInfo));

			/*
				First render pass: Generate shadow map by rendering the scene from light's POV
			*/
			{
				clearValue.depthStencil = { 1.0f, 0 };

				VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
				renderPassBeginInfo.renderPass = shadowmapFrameBuf.renderPass;
				renderPassBeginInfo.framebuffer = shadowmapFrameBuf.frameBuffer;
				renderPassBeginInfo.renderArea.extent.width = shadowmapFrameBuf.width;
				renderPassBeginInfo.renderArea.extent.height = shadowmapFrameBuf.height;
				renderPassBeginInfo.clearValueCount = 1;
				renderPassBeginInfo.pClearValues = &clearValue;

				vkCmdBeginRenderPass(shadowmapCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

				viewport = vks::initializers::viewport((float)shadowmapFrameBuf.width, (float)shadowmapFrameBuf.height, 0.0f, 1.0f);
				vkCmdSetViewport(shadowmapCmdBuffers[i], 0, 1, &viewport);

				scissor = vks::initializers::rect2D(shadowmapFrameBuf.width, shadowmapFrameBuf.height, 0, 0);
				vkCmdSetScissor(shadowmapCmdBuffers[i], 0, 1, &scissor);

				// Set depth bias (aka "Polygon offset")
				// Required to avoid shadow mapping artifacts
				vkCmdSetDepthBias(
					shadowmapCmdBuffers[i],
					depthBiasConstant,
					0.0f,
					depthBiasSlope);

				vkCmdBindPipeline(shadowmapCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.shadowmap);
				vkCmdBindDescriptorSets(shadowmapCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts.shadowmap, 0, 1, &descriptorSets.shadowmap, 0, nullptr);
				//scene.drawShadowmap(shadowmapCmdBuffers[i], vkglTF::RenderFlags::BindImages, pipelineLayouts.shadowmap, 0, &descriptorSets.shadowmap);
				scene.draw(shadowmapCmdBuffers[i]);

				vkCmdEndRenderPass(shadowmapCmdBuffers[i]);
			}

			VK_CHECK_RESULT(vkEndCommandBuffer(shadowmapCmdBuffers[i]));
		}
	}

	// [Pass 1]
	void buildGeometryCommandBuffer()
	{
		VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		// Clear values for all attachments written in the fragment shader
		std::array<VkClearValue, 6> clearValues;
		clearValues[0].color = { { 0.0f, 0.0f, 0.0f, 0.0f } };
		clearValues[1].color = { { 0.0f, 0.0f, 0.0f, 0.0f } };
		clearValues[2].color = { { 0.0f, 0.0f, 0.0f, 0.0f } };
		clearValues[3].color = { { 0.0f, 0.0f, 0.0f, 0.0f } };
		clearValues[4].color = { { 0.0f, 0.0f, 0.0f, 0.0f } };
		clearValues[5].depthStencil = { 1.0f, 0 };

		VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
		renderPassBeginInfo.renderPass = geometryFrameBuf.renderPass;
		renderPassBeginInfo.framebuffer = geometryFrameBuf.frameBuffer;
		renderPassBeginInfo.renderArea.extent.width = geometryFrameBuf.width;
		renderPassBeginInfo.renderArea.extent.height = geometryFrameBuf.height;
		renderPassBeginInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
		renderPassBeginInfo.pClearValues = clearValues.data();

		for (int32_t i = 0; i < geometryCmdBuffers.size(); ++i)
		{
			VK_CHECK_RESULT(vkBeginCommandBuffer(geometryCmdBuffers[i], &cmdBufInfo));

			vkCmdBeginRenderPass(geometryCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

			VkViewport viewport = vks::initializers::viewport((float)geometryFrameBuf.width, (float)geometryFrameBuf.height, 0.0f, 1.0f);
			vkCmdSetViewport(geometryCmdBuffers[i], 0, 1, &viewport);

			VkRect2D scissor = vks::initializers::rect2D(geometryFrameBuf.width, geometryFrameBuf.height, 0, 0);
			vkCmdSetScissor(geometryCmdBuffers[i], 0, 1, &scissor);

			vkCmdBindPipeline(geometryCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.offscreen);

			scene.draw(geometryCmdBuffers[i], vkglTF::RenderFlags::BindImages, pipelineLayouts.offscreen, 0);

			vkCmdEndRenderPass(geometryCmdBuffers[i]);

			VK_CHECK_RESULT(vkEndCommandBuffer(geometryCmdBuffers[i]));
		}
	}

	void loadCubemap(std::string filename, VkFormat format)
	{
		ktxResult result;
		ktxTexture* ktxTexture;

#if defined(__ANDROID__)
		// Textures are stored inside the apk on Android (compressed)
		// So they need to be loaded via the asset manager
		AAsset* asset = AAssetManager_open(androidApp->activity->assetManager, filename.c_str(), AASSET_MODE_STREAMING);
		if (!asset) {
			vks::tools::exitFatal("Could not load texture from " + filename + "\n\nMake sure the assets submodule has been checked out and is up-to-date.", -1);
		}
		size_t size = AAsset_getLength(asset);
		assert(size > 0);

		ktx_uint8_t* textureData = new ktx_uint8_t[size];
		AAsset_read(asset, textureData, size);
		AAsset_close(asset);
		result = ktxTexture_CreateFromMemory(textureData, size, KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &ktxTexture);
		delete[] textureData;
#else
		if (!vks::tools::fileExists(filename)) {
			vks::tools::exitFatal("Could not load texture from " + filename + "\n\nMake sure the assets submodule has been checked out and is up-to-date.", -1);
		}
		result = ktxTexture_CreateFromNamedFile(filename.c_str(), KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &ktxTexture);
#endif
		assert(result == KTX_SUCCESS);

		// Get properties required for using and upload texture data from the ktx texture object
		cubeMap.width = ktxTexture->baseWidth;
		cubeMap.height = ktxTexture->baseHeight;
		cubeMap.mipLevels = ktxTexture->numLevels;
		ktx_uint8_t* ktxTextureData = ktxTexture_GetData(ktxTexture);
		ktx_size_t ktxTextureSize = ktxTexture_GetSize(ktxTexture);

		VkMemoryAllocateInfo memAllocInfo = vks::initializers::memoryAllocateInfo();
		VkMemoryRequirements memReqs;

		// Create a host-visible staging buffer that contains the raw image data
		VkBuffer stagingBuffer;
		VkDeviceMemory stagingMemory;

		VkBufferCreateInfo bufferCreateInfo = vks::initializers::bufferCreateInfo();
		bufferCreateInfo.size = ktxTextureSize;
		// This buffer is used as a transfer source for the buffer copy
		bufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
		bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		VK_CHECK_RESULT(vkCreateBuffer(device, &bufferCreateInfo, nullptr, &stagingBuffer));

		// Get memory requirements for the staging buffer (alignment, memory type bits)
		vkGetBufferMemoryRequirements(device, stagingBuffer, &memReqs);
		memAllocInfo.allocationSize = memReqs.size;
		// Get memory type index for a host visible buffer
		memAllocInfo.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device, &memAllocInfo, nullptr, &stagingMemory));
		VK_CHECK_RESULT(vkBindBufferMemory(device, stagingBuffer, stagingMemory, 0));

		// Copy texture data into staging buffer
		uint8_t* data;
		VK_CHECK_RESULT(vkMapMemory(device, stagingMemory, 0, memReqs.size, 0, (void**)&data));
		memcpy(data, ktxTextureData, ktxTextureSize);
		vkUnmapMemory(device, stagingMemory);

		// Create optimal tiled target image
		VkImageCreateInfo imageCreateInfo = vks::initializers::imageCreateInfo();
		imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
		imageCreateInfo.format = format;
		imageCreateInfo.mipLevels = cubeMap.mipLevels;
		imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageCreateInfo.extent = { cubeMap.width, cubeMap.height, 1 };
		imageCreateInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
		// Cube faces count as array layers in Vulkan
		imageCreateInfo.arrayLayers = 6;
		// This flag is required for cube map images
		imageCreateInfo.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;

		VK_CHECK_RESULT(vkCreateImage(device, &imageCreateInfo, nullptr, &cubeMap.image));

		vkGetImageMemoryRequirements(device, cubeMap.image, &memReqs);

		memAllocInfo.allocationSize = memReqs.size;
		memAllocInfo.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

		VK_CHECK_RESULT(vkAllocateMemory(device, &memAllocInfo, nullptr, &cubeMap.deviceMemory));
		VK_CHECK_RESULT(vkBindImageMemory(device, cubeMap.image, cubeMap.deviceMemory, 0));

		VkCommandBuffer copyCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

		// Setup buffer copy regions for each face including all of its miplevels
		std::vector<VkBufferImageCopy> bufferCopyRegions;
		uint32_t offset = 0;

		for (uint32_t face = 0; face < 6; face++)
		{
			for (uint32_t level = 0; level < cubeMap.mipLevels; level++)
			{
				// Calculate offset into staging buffer for the current mip level and face
				ktx_size_t offset;
				KTX_error_code ret = ktxTexture_GetImageOffset(ktxTexture, level, 0, face, &offset);
				assert(ret == KTX_SUCCESS);
				VkBufferImageCopy bufferCopyRegion = {};
				bufferCopyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
				bufferCopyRegion.imageSubresource.mipLevel = level;
				bufferCopyRegion.imageSubresource.baseArrayLayer = face;
				bufferCopyRegion.imageSubresource.layerCount = 1;
				bufferCopyRegion.imageExtent.width = ktxTexture->baseWidth >> level;
				bufferCopyRegion.imageExtent.height = ktxTexture->baseHeight >> level;
				bufferCopyRegion.imageExtent.depth = 1;
				bufferCopyRegion.bufferOffset = offset;
				bufferCopyRegions.push_back(bufferCopyRegion);
			}
		}

		// Image barrier for optimal image (target)
		// Set initial layout for all array layers (faces) of the optimal (target) tiled texture
		VkImageSubresourceRange subresourceRange = {};
		subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		subresourceRange.baseMipLevel = 0;
		subresourceRange.levelCount = cubeMap.mipLevels;
		subresourceRange.layerCount = 6;

		vks::tools::setImageLayout(
			copyCmd,
			cubeMap.image,
			VK_IMAGE_LAYOUT_UNDEFINED,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			subresourceRange);

		// Copy the cube map faces from the staging buffer to the optimal tiled image
		vkCmdCopyBufferToImage(
			copyCmd,
			stagingBuffer,
			cubeMap.image,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			static_cast<uint32_t>(bufferCopyRegions.size()),
			bufferCopyRegions.data()
		);

		// Change texture image layout to shader read after all faces have been copied
		cubeMap.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		vks::tools::setImageLayout(
			copyCmd,
			cubeMap.image,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			cubeMap.imageLayout,
			subresourceRange);

		vulkanDevice->flushCommandBuffer(copyCmd, queue, true);

		// Create sampler
		VkSamplerCreateInfo sampler = vks::initializers::samplerCreateInfo();
		sampler.magFilter = VK_FILTER_LINEAR;
		sampler.minFilter = VK_FILTER_LINEAR;
		sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		sampler.addressModeV = sampler.addressModeU;
		sampler.addressModeW = sampler.addressModeU;
		sampler.mipLodBias = 0.0f;
		sampler.compareOp = VK_COMPARE_OP_NEVER;
		sampler.minLod = 0.0f;
		sampler.maxLod = static_cast<float>(cubeMap.mipLevels);
		sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
		sampler.maxAnisotropy = 1.0f;
		if (vulkanDevice->features.samplerAnisotropy)
		{
			sampler.maxAnisotropy = vulkanDevice->properties.limits.maxSamplerAnisotropy;
			sampler.anisotropyEnable = VK_TRUE;
		}
		VK_CHECK_RESULT(vkCreateSampler(device, &sampler, nullptr, &cubeMap.sampler));

		// Create image view
		VkImageViewCreateInfo view = vks::initializers::imageViewCreateInfo();
		// Cube map view type
		view.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
		view.format = format;
		view.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
		// 6 array layers (faces)
		view.subresourceRange.layerCount = 6;
		// Set number of mip levels
		view.subresourceRange.levelCount = cubeMap.mipLevels;
		view.image = cubeMap.image;
		VK_CHECK_RESULT(vkCreateImageView(device, &view, nullptr, &cubeMap.view));

		// Clean up staging resources
		vkFreeMemory(device, stagingMemory, nullptr);
		vkDestroyBuffer(device, stagingBuffer, nullptr);
		ktxTexture_Destroy(ktxTexture);
	}

	void loadAssets()
	{
		const uint32_t glTFLoadingFlags = vkglTF::FileLoadingFlags::PreMultiplyVertexColors | vkglTF::FileLoadingFlags::PreTransformVertices;
		vkglTF::memoryPropertyFlags = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

		scene.loadFromFile(getAssetPath() + ASSET_PATH, vulkanDevice, queue, glTFLoadingFlags);
		loadCubemap(getAssetPath() + CUBEMAP_TEXTURE_PATH, VK_FORMAT_R8G8B8A8_UNORM);
	}

	void createAccelerationStructureBuffer(AccelerationStructure& accelerationStructure, VkAccelerationStructureBuildSizesInfoKHR buildSizeInfo)
	{
		VkBufferCreateInfo bufferCreateInfo{};
		bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferCreateInfo.size = buildSizeInfo.accelerationStructureSize;
		bufferCreateInfo.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
		VK_CHECK_RESULT(vkCreateBuffer(device, &bufferCreateInfo, nullptr, &accelerationStructure.buffer));
		VkMemoryRequirements memoryRequirements{};
		vkGetBufferMemoryRequirements(device, accelerationStructure.buffer, &memoryRequirements);
		VkMemoryAllocateFlagsInfo memoryAllocateFlagsInfo{};
		memoryAllocateFlagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
		memoryAllocateFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;
		VkMemoryAllocateInfo memoryAllocateInfo{};
		memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		memoryAllocateInfo.pNext = &memoryAllocateFlagsInfo;
		memoryAllocateInfo.allocationSize = memoryRequirements.size;
		memoryAllocateInfo.memoryTypeIndex = vulkanDevice->getMemoryType(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device, &memoryAllocateInfo, nullptr, &accelerationStructure.memory));
		VK_CHECK_RESULT(vkBindBufferMemory(device, accelerationStructure.buffer, accelerationStructure.memory, 0));
	}

	void createBottomLevelAccelerationStructure()
	{
		// Use transform matrices from the glTF nodes
		std::vector<VkTransformMatrixKHR> transformMatrices{};
		for (auto node : scene.linearNodes) {
			if (node->mesh) {
				for (auto primitive : node->mesh->primitives) {
					if (primitive->indexCount > 0) {
						VkTransformMatrixKHR transformMatrix{};
						//auto m = glm::mat3x4(glm::transpose(node->getMatrix()));
						auto m = glm::mat3x4(glm::mat4(1.0f));
						memcpy(&transformMatrix, (void*)&m, sizeof(glm::mat3x4));
						transformMatrices.push_back(transformMatrix);
					}
				}
			}
		}

		// Transform buffer
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&transformBuffer,
			static_cast<uint32_t>(transformMatrices.size()) * sizeof(VkTransformMatrixKHR),
			transformMatrices.data()));

		// Build
		// One geometry per glTF node, so we can index materials using gl_GeometryIndexEXT
		uint32_t maxPrimCount{ 0 };
		std::vector<uint32_t> maxPrimitiveCounts;
		std::vector<VkAccelerationStructureGeometryKHR> geometries;
		std::vector<VkAccelerationStructureBuildRangeInfoKHR> buildRangeInfos;
		std::vector<VkAccelerationStructureBuildRangeInfoKHR*> pBuildRangeInfos;
		std::vector<GeometryNode> geometryNodes;

		uint32_t geometryNodesSizeTotal = 0;
		for (auto node : scene.linearNodes) {
			if (node->mesh) {
				for (auto primitive : node->mesh->primitives) {
					if (primitive->indexCount > 0) {
						VkDeviceOrHostAddressConstKHR vertexBufferDeviceAddress{};
						VkDeviceOrHostAddressConstKHR indexBufferDeviceAddress{};
						VkDeviceOrHostAddressConstKHR transformBufferDeviceAddress{};

						vertexBufferDeviceAddress.deviceAddress = getBufferDeviceAddress(scene.vertices.buffer);// +primitive->firstVertex * sizeof(vkglTF::Vertex);
						indexBufferDeviceAddress.deviceAddress = getBufferDeviceAddress(scene.indices.buffer) + primitive->firstIndex * sizeof(uint32_t);
						transformBufferDeviceAddress.deviceAddress = getBufferDeviceAddress(transformBuffer.buffer) + static_cast<uint32_t>(geometryNodes.size()) * sizeof(VkTransformMatrixKHR);

						VkAccelerationStructureGeometryKHR geometry{};
						geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
						geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
						geometry.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
						geometry.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
						geometry.geometry.triangles.vertexData = vertexBufferDeviceAddress;
						geometry.geometry.triangles.maxVertex = scene.vertices.count;
						geometry.geometry.triangles.vertexStride = sizeof(vkglTF::Vertex);
						geometry.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
						geometry.geometry.triangles.indexData = indexBufferDeviceAddress;
						geometry.geometry.triangles.transformData = transformBufferDeviceAddress;
						geometries.push_back(geometry);
						maxPrimitiveCounts.push_back(primitive->indexCount / 3);
						maxPrimCount += primitive->indexCount / 3;

						VkAccelerationStructureBuildRangeInfoKHR buildRangeInfo{};
						buildRangeInfo.firstVertex = 0;
						buildRangeInfo.primitiveOffset = 0; // primitive->firstIndex * sizeof(uint32_t);
						buildRangeInfo.primitiveCount = primitive->indexCount / 3;
						buildRangeInfo.transformOffset = 0;
						buildRangeInfos.push_back(buildRangeInfo);

						GeometryNode geometryNode{};
						geometryNode.vertexBufferDeviceAddress = vertexBufferDeviceAddress.deviceAddress;
						geometryNode.indexBufferDeviceAddress = indexBufferDeviceAddress.deviceAddress;
						geometryNode.textureIndexBaseColor = primitive->material.baseColorTexture ? primitive->material.baseColorTexture->index : -1;
						geometryNode.textureIndexOcclusion = primitive->material.occlusionTexture ? primitive->material.occlusionTexture->index : -1;
						geometryNode.textureIndexNormal = primitive->material.normalTexture ? primitive->material.normalTexture->index : -1;
						geometryNode.textureIndexMetallicRoughness = primitive->material.metallicRoughnessTexture ? primitive->material.metallicRoughnessTexture->index : -1;
						geometryNode.textureIndexEmissive = primitive->material.emissiveTexture ? primitive->material.emissiveTexture->index : -1;
						//Y&Y added begin
						geometryNode.reflectance = primitive->material.Kr;
						geometryNode.refractance = primitive->material.Kt;
						geometryNode.ior = primitive->material.ior;
						//Y&Y added end
						// @todo: map material id to global texture array
						geometryNodes.push_back(geometryNode);
					}
				}
			}
		}

		for (auto& rangeInfo : buildRangeInfos) {
			pBuildRangeInfos.push_back(&rangeInfo);
		}

		// @todo: stage to device

		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&geometryNodesBuffer,
			static_cast<uint32_t>(geometryNodes.size()) * sizeof(GeometryNode),
			geometryNodes.data()));

		// Get size info
		VkAccelerationStructureBuildGeometryInfoKHR accelerationStructureBuildGeometryInfo{};
		accelerationStructureBuildGeometryInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
		accelerationStructureBuildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
		accelerationStructureBuildGeometryInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
		accelerationStructureBuildGeometryInfo.geometryCount = static_cast<uint32_t>(geometries.size());
		accelerationStructureBuildGeometryInfo.pGeometries = geometries.data();

		const uint32_t numTriangles = maxPrimitiveCounts[0];

		VkAccelerationStructureBuildSizesInfoKHR accelerationStructureBuildSizesInfo{};
		accelerationStructureBuildSizesInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
		vkGetAccelerationStructureBuildSizesKHR(
			device,
			VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
			&accelerationStructureBuildGeometryInfo,
			maxPrimitiveCounts.data(),
			&accelerationStructureBuildSizesInfo);

		createAccelerationStructureBuffer(bottomLevelAS, accelerationStructureBuildSizesInfo);

		VkAccelerationStructureCreateInfoKHR accelerationStructureCreateInfo{};
		accelerationStructureCreateInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
		accelerationStructureCreateInfo.buffer = bottomLevelAS.buffer;
		accelerationStructureCreateInfo.size = accelerationStructureBuildSizesInfo.accelerationStructureSize;
		accelerationStructureCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
		vkCreateAccelerationStructureKHR(device, &accelerationStructureCreateInfo, nullptr, &bottomLevelAS.handle);

		// Create a small scratch buffer used during build of the bottom level acceleration structure
		ScratchBuffer scratchBuffer = createScratchBuffer(accelerationStructureBuildSizesInfo.buildScratchSize);

		accelerationStructureBuildGeometryInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
		accelerationStructureBuildGeometryInfo.dstAccelerationStructure = bottomLevelAS.handle;
		accelerationStructureBuildGeometryInfo.scratchData.deviceAddress = scratchBuffer.deviceAddress;

		const VkAccelerationStructureBuildRangeInfoKHR* buildOffsetInfo = buildRangeInfos.data();

		// Build the acceleration structure on the device via a one-time command buffer submission
		// Some implementations may support acceleration structure building on the host (VkPhysicalDeviceAccelerationStructureFeaturesKHR->accelerationStructureHostCommands), but we prefer device builds
		VkCommandBuffer commandBuffer = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		vkCmdBuildAccelerationStructuresKHR(
			commandBuffer,
			1,
			&accelerationStructureBuildGeometryInfo,
			pBuildRangeInfos.data());
		vulkanDevice->flushCommandBuffer(commandBuffer, queue);

		VkAccelerationStructureDeviceAddressInfoKHR accelerationDeviceAddressInfo{};
		accelerationDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
		accelerationDeviceAddressInfo.accelerationStructure = bottomLevelAS.handle;
		bottomLevelAS.deviceAddress = vkGetAccelerationStructureDeviceAddressKHR(device, &accelerationDeviceAddressInfo);

		deleteScratchBuffer(scratchBuffer);
	}

	void createTopLevelAccelerationStructure()
	{
		VkTransformMatrixKHR transformMatrix = {
			1.0f, 0.0f, 0.0f, 0.0f,
			0.0f, 1.0f, 0.0f, 0.0f,
			0.0f, 0.0f, 1.0f, 0.0f };

		VkAccelerationStructureInstanceKHR instance{};
		instance.transform = transformMatrix;
		instance.instanceCustomIndex = 0;
		instance.mask = 0xFF;
		instance.instanceShaderBindingTableRecordOffset = 0;
		instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
		instance.accelerationStructureReference = bottomLevelAS.deviceAddress;

		// Buffer for instance data
		vks::Buffer instancesBuffer;
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&instancesBuffer,
			sizeof(VkAccelerationStructureInstanceKHR),
			&instance));

		VkDeviceOrHostAddressConstKHR instanceDataDeviceAddress{};
		instanceDataDeviceAddress.deviceAddress = getBufferDeviceAddress(instancesBuffer.buffer);

		VkAccelerationStructureGeometryKHR accelerationStructureGeometry{};
		accelerationStructureGeometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
		accelerationStructureGeometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
		accelerationStructureGeometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
		accelerationStructureGeometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
		accelerationStructureGeometry.geometry.instances.arrayOfPointers = VK_FALSE;
		accelerationStructureGeometry.geometry.instances.data = instanceDataDeviceAddress;

		// Get size info
		/*
		The pSrcAccelerationStructure, dstAccelerationStructure, and mode members of pBuildInfo are ignored. Any VkDeviceOrHostAddressKHR members of pBuildInfo are ignored by this command, except that the hostAddress member of VkAccelerationStructureGeometryTrianglesDataKHR::transformData will be examined to check if it is NULL.*
		*/
		VkAccelerationStructureBuildGeometryInfoKHR accelerationStructureBuildGeometryInfo{};
		accelerationStructureBuildGeometryInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
		accelerationStructureBuildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
		accelerationStructureBuildGeometryInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
		accelerationStructureBuildGeometryInfo.geometryCount = 1;
		accelerationStructureBuildGeometryInfo.pGeometries = &accelerationStructureGeometry;

		uint32_t primitive_count = 1;

		VkAccelerationStructureBuildSizesInfoKHR accelerationStructureBuildSizesInfo{};
		accelerationStructureBuildSizesInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
		vkGetAccelerationStructureBuildSizesKHR(
			device,
			VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
			&accelerationStructureBuildGeometryInfo,
			&primitive_count,
			&accelerationStructureBuildSizesInfo);

		createAccelerationStructureBuffer(topLevelAS, accelerationStructureBuildSizesInfo);

		VkAccelerationStructureCreateInfoKHR accelerationStructureCreateInfo{};
		accelerationStructureCreateInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
		accelerationStructureCreateInfo.buffer = topLevelAS.buffer;
		accelerationStructureCreateInfo.size = accelerationStructureBuildSizesInfo.accelerationStructureSize;
		accelerationStructureCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
		vkCreateAccelerationStructureKHR(device, &accelerationStructureCreateInfo, nullptr, &topLevelAS.handle);

		// Create a small scratch buffer used during build of the top level acceleration structure
		ScratchBuffer scratchBuffer = createScratchBuffer(accelerationStructureBuildSizesInfo.buildScratchSize);

		VkAccelerationStructureBuildGeometryInfoKHR accelerationBuildGeometryInfo{};
		accelerationBuildGeometryInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
		accelerationBuildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
		accelerationBuildGeometryInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
		accelerationBuildGeometryInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
		accelerationBuildGeometryInfo.dstAccelerationStructure = topLevelAS.handle;
		accelerationBuildGeometryInfo.geometryCount = 1;
		accelerationBuildGeometryInfo.pGeometries = &accelerationStructureGeometry;
		accelerationBuildGeometryInfo.scratchData.deviceAddress = scratchBuffer.deviceAddress;

		VkAccelerationStructureBuildRangeInfoKHR accelerationStructureBuildRangeInfo{};
		accelerationStructureBuildRangeInfo.primitiveCount = primitive_count;
		accelerationStructureBuildRangeInfo.primitiveOffset = 0;
		accelerationStructureBuildRangeInfo.firstVertex = 0;
		accelerationStructureBuildRangeInfo.transformOffset = 0;
		std::vector<VkAccelerationStructureBuildRangeInfoKHR*> accelerationBuildStructureRangeInfos = { &accelerationStructureBuildRangeInfo };

		// Build the acceleration structure on the device via a one-time command buffer submission
		// Some implementations may support acceleration structure building on the host (VkPhysicalDeviceAccelerationStructureFeaturesKHR->accelerationStructureHostCommands), but we prefer device builds
		VkCommandBuffer commandBuffer = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		vkCmdBuildAccelerationStructuresKHR(
			commandBuffer,
			1,
			&accelerationBuildGeometryInfo,
			accelerationBuildStructureRangeInfos.data());
		vulkanDevice->flushCommandBuffer(commandBuffer, queue);

		VkAccelerationStructureDeviceAddressInfoKHR accelerationDeviceAddressInfo{};
		accelerationDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
		accelerationDeviceAddressInfo.accelerationStructure = topLevelAS.handle;

		deleteScratchBuffer(scratchBuffer);
		instancesBuffer.destroy();
	}

	void handleResize()
	{
		std::cout << "handleResize function called.\n";

		// Recreate image
		createStorageImage(swapChain.colorFormat, { width, height, 1 });
		// Update descriptor
		VkDescriptorImageInfo storageImageDescriptor{ VK_NULL_HANDLE, storageImage.view, VK_IMAGE_LAYOUT_GENERAL };
		VkWriteDescriptorSet resultImageWrite = vks::initializers::writeDescriptorSet(descriptorSets.composition, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, &storageImageDescriptor);
		vkUpdateDescriptorSets(device, 1, &resultImageWrite, 0, VK_NULL_HANDLE);
		resized = false;
	}

	void buildCommandBuffers()
	{
		if (resized)
		{
			handleResize();
		}

		VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		VkImageSubresourceRange subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

		for (int32_t i = 0; i < drawCmdBuffers.size(); ++i)
		{
			VK_CHECK_RESULT(vkBeginCommandBuffer(drawCmdBuffers[i], &cmdBufInfo));

			/*
				Dispatch the ray tracing commands
			*/
			vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipelines.composition);
			vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipelineLayouts.composition, 0, 1, &descriptorSets.composition, 0, 0);

			VkStridedDeviceAddressRegionKHR emptySbtEntry = {};
			vkCmdTraceRaysKHR(
				drawCmdBuffers[i],
				&shaderBindingTables.raygen.stridedDeviceAddressRegion,
				&shaderBindingTables.miss.stridedDeviceAddressRegion,
				&shaderBindingTables.hit.stridedDeviceAddressRegion,
				&emptySbtEntry,
				width,
				height,
				1);

			/*
				Copy ray tracing output to swap chain image
			*/

			// Prepare current swap chain image as transfer destination
			vks::tools::setImageLayout(
				drawCmdBuffers[i],
				swapChain.images[i],
				VK_IMAGE_LAYOUT_UNDEFINED,
				VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				subresourceRange);

			// Prepare ray tracing output image as transfer source
			vks::tools::setImageLayout(
				drawCmdBuffers[i],
				storageImage.image,
				VK_IMAGE_LAYOUT_GENERAL,
				VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				subresourceRange);

			VkImageCopy copyRegion{};
			copyRegion.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
			copyRegion.srcOffset = { 0, 0, 0 };
			copyRegion.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
			copyRegion.dstOffset = { 0, 0, 0 };
			copyRegion.extent = { width, height, 1 };
			vkCmdCopyImage(drawCmdBuffers[i], storageImage.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, swapChain.images[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

			// Transition swap chain image back for presentation
			vks::tools::setImageLayout(
				drawCmdBuffers[i],
				swapChain.images[i],
				VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
				subresourceRange);

			// Transition ray tracing output image back to general layout
			vks::tools::setImageLayout(
				drawCmdBuffers[i],
				storageImage.image,
				VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				VK_IMAGE_LAYOUT_GENERAL,
				subresourceRange);

			drawUI(drawCmdBuffers[i], frameBuffers[i]);

			VK_CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
		}
	}

	void createDescriptorSets()
	{
		int imageSamplerCount = 0;
		int materialCount = 0;
		int uboCount = 1;
		for (auto& material : scene.materials) {
			imageSamplerCount += 4;
			materialCount++;
			uboCount++;
		}

		// Pool
		std::vector<VkDescriptorPoolSize> poolSizes = {
			// [Pass 0]
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1),

			// [Pass 1]
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, uboCount),
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, imageSamplerCount),

			// [Pass 2]
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1),
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1),
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1),
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1),
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 8),
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, static_cast<uint32_t>(scene.textures.size()))
		};

		VkDescriptorPoolCreateInfo descriptorPoolInfo = vks::initializers::descriptorPoolCreateInfo(poolSizes, 3 + materialCount);
		VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));

		std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings;
		VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI;

		// [Pass 0] 
		setLayoutBindings = {
			// Binding 0 : [shadowmap.vert] 
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0)
		};
		descriptorSetLayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCI, nullptr, &descriptorSetLayouts.shadowmap));

		VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayouts.shadowmap, 1);
		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSets.shadowmap));

		std::vector<VkWriteDescriptorSet> writeDescriptorSets;
		writeDescriptorSets = {
			vks::initializers::writeDescriptorSet(descriptorSets.shadowmap, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &uniformBuffers.shadowmap.descriptor)
		};
		vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, NULL);

		// [Pass 1] 
		setLayoutBindings = {
			// Binding 0 : [mrt.frag] Scene Colormap
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0),
			// Binding 1 : [mrt.frag] Scene MetallicRoughnessmap
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1),
			// Binding 2 : [mrt.frag] Scene Normalmap
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 2),
			// Binding 3 : [mrt.frag] Scene Emissivemap
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 3),
			// Binding 4 : [mrt.vert] Uniform Buffer
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 4),
		};
		descriptorSetLayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCI, nullptr, &descriptorSetLayouts.offscreen));

		allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayouts.offscreen, 1);

		for (auto& material : scene.materials) {
			VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &material.descriptorSet));

			std::vector<VkDescriptorImageInfo> imageDescriptors = {
				material.baseColorTexture ? material.baseColorTexture->descriptor : scene.emptyTexture.descriptor,
				material.metallicRoughnessTexture ? material.metallicRoughnessTexture->descriptor : scene.emptyTexture.descriptor,
				material.normalTexture ? material.normalTexture->descriptor : scene.emptyTexture.descriptor,
				material.emissiveTexture ? material.emissiveTexture->descriptor : scene.emptyTexture.descriptor
			};
			std::array<VkWriteDescriptorSet, 5> writeDescriptorSets{};
			for (size_t i = 0; i < 5; i++)
			{
				if (i < 4)
					writeDescriptorSets[i] = vks::initializers::writeDescriptorSet(material.descriptorSet, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, i, &imageDescriptors[i]);
				else
					writeDescriptorSets[i] = vks::initializers::writeDescriptorSet(material.descriptorSet, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, i, &uniformBuffers.offscreen.descriptor);
			}
			vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, NULL);
		}

		// [Pass 2]
		setLayoutBindings = {
			// Binding 0: Top level acceleration structure
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 0),
			// Binding 1: Ray tracing result image
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_KHR, 1),
			// Binding 2: Geometry node information SSBO(Shader Storage Buffer Object)
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR | VK_SHADER_STAGE_INTERSECTION_BIT_KHR, 2),
			// Binding 3: Uniform Buffer
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 3),
			// Binding 4: Position texture target
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_RAYGEN_BIT_KHR, 4),
			// Binding 5: Normals texture target
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_RAYGEN_BIT_KHR, 5),
			// Binding 6: Albedo texture target
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_RAYGEN_BIT_KHR, 6),
			// Binding 7: MetallicRoughness texture target
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_RAYGEN_BIT_KHR, 7),
			// Binding 8: Emissive texture target
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_RAYGEN_BIT_KHR, 8),
			// Binding 9: Cubemap sampler for miss shader
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR, 9),
			// Binding 10: Shadow Edge
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 10),
			// Binding 11: Light
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 11),
			// Binding 12: All images used by the glTF model
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR, 12, static_cast<uint32_t>(scene.textures.size()))
		};
		descriptorSetLayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);

		// Unbound set
		VkDescriptorSetLayoutBindingFlagsCreateInfoEXT setLayoutBindingFlags{};
		setLayoutBindingFlags.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO_EXT;
		setLayoutBindingFlags.bindingCount = 13;
		std::vector<VkDescriptorBindingFlagsEXT> descriptorBindingFlags = {
			0,
			0,
			0,
			0,
			0,
			0,
			0,
			0,
			0,
			0,
			0,
			0,
			VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT_EXT
		};
		setLayoutBindingFlags.pBindingFlags = descriptorBindingFlags.data();

		descriptorSetLayoutCI.pNext = &setLayoutBindingFlags;
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCI, nullptr, &descriptorSetLayouts.composition));

		allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayouts.composition, 1);

		// Image descriptors for the offscreen color attachments
		VkDescriptorImageInfo texDescriptorPosition =
			vks::initializers::descriptorImageInfo(
				colorSampler,
				geometryFrameBuf.position.view,
				VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

		VkDescriptorImageInfo texDescriptorNormal =
			vks::initializers::descriptorImageInfo(
				colorSampler,
				geometryFrameBuf.normal.view,
				VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

		VkDescriptorImageInfo texDescriptorAlbedo =
			vks::initializers::descriptorImageInfo(
				colorSampler,
				geometryFrameBuf.albedo.view,
				VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

		VkDescriptorImageInfo texDescriptorMetallicRoughness =
			vks::initializers::descriptorImageInfo(
				colorSampler,
				geometryFrameBuf.metallicRoughness.view,
				VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

		VkDescriptorImageInfo texDescriptorEmissive =
			vks::initializers::descriptorImageInfo(
				colorSampler,
				geometryFrameBuf.emissive.view,
				VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

		VkDescriptorSetVariableDescriptorCountAllocateInfoEXT variableDescriptorCountAllocInfo{};
		uint32_t variableDescCounts[] = { static_cast<uint32_t>(scene.textures.size()) };
		variableDescriptorCountAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO_EXT;
		variableDescriptorCountAllocInfo.descriptorSetCount = 1;
		variableDescriptorCountAllocInfo.pDescriptorCounts = variableDescCounts;

		allocInfo.pNext = &variableDescriptorCountAllocInfo;

		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSets.composition));

		VkWriteDescriptorSetAccelerationStructureKHR descriptorAccelerationStructureInfo = vks::initializers::writeDescriptorSetAccelerationStructureKHR();
		descriptorAccelerationStructureInfo.accelerationStructureCount = 1;
		descriptorAccelerationStructureInfo.pAccelerationStructures = &topLevelAS.handle;

		VkWriteDescriptorSet accelerationStructureWrite{};
		accelerationStructureWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		// The specialized acceleration structure descriptor has to be chained
		accelerationStructureWrite.pNext = &descriptorAccelerationStructureInfo;
		accelerationStructureWrite.dstSet = descriptorSets.composition;
		accelerationStructureWrite.dstBinding = 0;
		accelerationStructureWrite.descriptorCount = 1;
		accelerationStructureWrite.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

		VkDescriptorImageInfo storageImageDescriptor{ VK_NULL_HANDLE, storageImage.view, VK_IMAGE_LAYOUT_GENERAL };

		// Composition WriteDescriptorSets
		writeDescriptorSets = {
			// Binding 0: Top level acceleration structure
			accelerationStructureWrite,
			// Binding 1: Ray tracing result image
			vks::initializers::writeDescriptorSet(descriptorSets.composition, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, &storageImageDescriptor),
			// Binding 2: Geometry node information SSBO(Shader Storage Buffer Object)
			vks::initializers::writeDescriptorSet(descriptorSets.composition, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2, &geometryNodesBuffer.descriptor),
			// Binding 3: Uniform Buffer
			vks::initializers::writeDescriptorSet(descriptorSets.composition, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3, &uniformBuffers.composition.descriptor),
			// Binding 4: Position texture target
			vks::initializers::writeDescriptorSet(descriptorSets.composition, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4, &texDescriptorPosition),
			// Binding 5: Normals texture target
			vks::initializers::writeDescriptorSet(descriptorSets.composition, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 5, &texDescriptorNormal),
			// Binding 6: Albedo texture target
			vks::initializers::writeDescriptorSet(descriptorSets.composition, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 6, &texDescriptorAlbedo),
			// Binding 7: MetallicRoughness texture target
			vks::initializers::writeDescriptorSet(descriptorSets.composition, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 7, &texDescriptorMetallicRoughness),
			// Binding 8: Emissive texture target
			vks::initializers::writeDescriptorSet(descriptorSets.composition, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 8, &texDescriptorEmissive),
		};
		// Binding 9: Cubemap sampler for miss shader
		VkDescriptorImageInfo cubeMapDescriptor = vks::initializers::descriptorImageInfo(cubeMap.sampler, cubeMap.view, cubeMap.imageLayout);
		writeDescriptorSets.push_back(vks::initializers::writeDescriptorSet(descriptorSets.composition, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 9, &cubeMapDescriptor));

		// Binding 10 : shadow edge
		VkDescriptorImageInfo shadowMapDescriptor =
			vks::initializers::descriptorImageInfo(
				shadowEdgeTextureSampler,
				shadowEdgeTextureImageView,
				VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		writeDescriptorSets.push_back(vks::initializers::writeDescriptorSet(descriptorSets.composition, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 10, &shadowMapDescriptor));

		// Binding 11 : light
		VkDescriptorImageInfo lightDescriptor =
			vks::initializers::descriptorImageInfo(
				lightTextureSampler,
				lightTextureImageView,
				VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		writeDescriptorSets.push_back(vks::initializers::writeDescriptorSet(descriptorSets.composition, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 11, &lightDescriptor));

		// Binding 12: All images used by the glTF model
		VkWriteDescriptorSet writeDescriptorImgArray{};
		std::vector<VkDescriptorImageInfo> textureDescriptors{};
		for (auto texture : scene.textures) {
			VkDescriptorImageInfo descriptor{};
			descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			descriptor.sampler = texture.sampler;
			descriptor.imageView = texture.view;
			textureDescriptors.push_back(descriptor);
		}
		writeDescriptorImgArray.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writeDescriptorImgArray.dstBinding = 12;
		writeDescriptorImgArray.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		writeDescriptorImgArray.descriptorCount = static_cast<uint32_t>(scene.textures.size());
		writeDescriptorImgArray.dstSet = descriptorSets.composition;
		writeDescriptorImgArray.pImageInfo = textureDescriptors.data();
		writeDescriptorSets.push_back(writeDescriptorImgArray);

		vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, VK_NULL_HANDLE);
	}

	/*
		Create the Shader Binding Tables that binds the programs and top-level acceleration structure

		SBT Layout used in VulkanHybridShadowmap project:

		/---------------------------------------\
		| Raygen                                |
		|---------------------------------------|
		| Miss(Sky Box)                         |
		|---------------------------------------|
		| Shadow Miss                           |
		|---------------------------------------|
		| Closest Hit Reflection                |
		|---------------------------------------|
		| Closest Hit Transmission              |
		\---------------------------------------/
	*/
	void createShaderBindingTables() {
		const uint32_t handleSize = rayTracingPipelineProperties.shaderGroupHandleSize;
		const uint32_t handleSizeAligned = vks::tools::alignedSize(rayTracingPipelineProperties.shaderGroupHandleSize, rayTracingPipelineProperties.shaderGroupHandleAlignment);
		const uint32_t groupCount = static_cast<uint32_t>(shaderGroups.size());
		const uint32_t sbtSize = groupCount * handleSizeAligned;

		std::vector<uint8_t> shaderHandleStorage(sbtSize);
		VK_CHECK_RESULT(vkGetRayTracingShaderGroupHandlesKHR(device, pipelines.composition, 0, groupCount, sbtSize, shaderHandleStorage.data()));

		createShaderBindingTable(shaderBindingTables.raygen, 1);
		createShaderBindingTable(shaderBindingTables.miss, 2);
		createShaderBindingTable(shaderBindingTables.hit, 1);

		// Copy handles
		// Global[0], Group Local[0]: raygen shader
		memcpy(shaderBindingTables.raygen.mapped, shaderHandleStorage.data(), handleSize);
		// Global[1 ~ 2], Group Local[0 ~ 1]: miss shader
		memcpy(shaderBindingTables.miss.mapped, shaderHandleStorage.data() + handleSizeAligned, handleSize * 2);
		// Global[3 ~ 4], Group Local[0 ~ 1]: closest hit shader
		memcpy(shaderBindingTables.hit.mapped, shaderHandleStorage.data() + handleSizeAligned * 3, handleSize);
	}

	void preparePipelines()
	{
		// Pipeline layout
		VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo;
		VkGraphicsPipelineCreateInfo pipelineCreateInfo;

		// [Pass 0] Generating Shadow map
		pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayouts.shadowmap, 1);
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayouts.shadowmap));

		std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages;
		shaderStages[0] = loadShader(getShadersPath() + DIR_PATH + "shadowmap.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
		shaderStages[1] = loadShader(getShadersPath() + DIR_PATH + "shadowmap.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);

		pipelineCreateInfo = vks::initializers::pipelineCreateInfo(pipelineLayouts.shadowmap, shadowmapFrameBuf.renderPass);

		VkPipelineInputAssemblyStateCreateInfo inputAssemblyState = vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);
		VkPipelineRasterizationStateCreateInfo rasterizationState = vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0);
		VkPipelineColorBlendAttachmentState blendAttachmentState = vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
		VkPipelineColorBlendStateCreateInfo colorBlendState = vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);
		VkPipelineDepthStencilStateCreateInfo depthStencilState = vks::initializers::pipelineDepthStencilStateCreateInfo(VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL);
		VkPipelineViewportStateCreateInfo viewportState = vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);
		VkPipelineMultisampleStateCreateInfo multisampleState = vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT, 0);
		std::vector<VkDynamicState> dynamicStateEnables = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_DEPTH_BIAS };
		VkPipelineDynamicStateCreateInfo dynamicState = vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);
		//VkPipelineVertexInputStateCreateInfo emptyInputState = vks::initializers::pipelineVertexInputStateCreateInfo();

		rasterizationState.cullMode = VK_CULL_MODE_NONE;
		rasterizationState.depthBiasEnable = VK_TRUE;
		colorBlendState.attachmentCount = 0;

		pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
		pipelineCreateInfo.pRasterizationState = &rasterizationState;
		pipelineCreateInfo.pColorBlendState = &colorBlendState;
		pipelineCreateInfo.pMultisampleState = &multisampleState;
		pipelineCreateInfo.pViewportState = &viewportState;
		pipelineCreateInfo.pDepthStencilState = &depthStencilState;
		pipelineCreateInfo.pDynamicState = &dynamicState;
		pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
		pipelineCreateInfo.pStages = shaderStages.data();
		pipelineCreateInfo.pVertexInputState = vkglTF::Vertex::getPipelineVertexInputState({ vkglTF::VertexComponent::Position });

		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipelines.shadowmap));

		// [Pass 1] Geometry pass
		pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayouts.offscreen, 1);
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayouts.offscreen));

		shaderStages[0] = loadShader(getShadersPath() + DIR_PATH + "mrt.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
		shaderStages[1] = loadShader(getShadersPath() + DIR_PATH + "mrt.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);

		pipelineCreateInfo = vks::initializers::pipelineCreateInfo(pipelineLayouts.offscreen, renderPass);

		rasterizationState.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterizationState.depthBiasEnable = VK_FALSE;
		std::array<VkPipelineColorBlendAttachmentState, 5> blendAttachmentStates = {
			vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE),
			vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE),
			vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE),
			vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE),
			vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE)
		};

		colorBlendState.attachmentCount = static_cast<uint32_t>(blendAttachmentStates.size());
		colorBlendState.pAttachments = blendAttachmentStates.data();
		dynamicStateEnables = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
		dynamicState = vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);

		pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
		pipelineCreateInfo.pRasterizationState = &rasterizationState;
		pipelineCreateInfo.pColorBlendState = &colorBlendState;
		pipelineCreateInfo.pMultisampleState = &multisampleState;
		pipelineCreateInfo.pViewportState = &viewportState;
		pipelineCreateInfo.pDepthStencilState = &depthStencilState;
		pipelineCreateInfo.pDynamicState = &dynamicState;
		pipelineCreateInfo.pVertexInputState = vkglTF::Vertex::getPipelineVertexInputState({ vkglTF::VertexComponent::Position, vkglTF::VertexComponent::UV, vkglTF::VertexComponent::Normal, vkglTF::VertexComponent::Color, vkglTF::VertexComponent::Tangent , vkglTF::VertexComponent::ObjectID });
		pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
		pipelineCreateInfo.pStages = shaderStages.data();
		pipelineCreateInfo.renderPass = geometryFrameBuf.renderPass;

		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipelines.offscreen));

		// [Pass 2] Lighting pass
		pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayouts.composition, 1);
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayouts.composition));

		std::vector<VkPipelineShaderStageCreateInfo> shaderStagesRT;

		uint32_t anyHitIdx;
		{
			shaderStagesRT.push_back(loadShader(getShadersPath() + DIR_PATH + "anyhit.rahit.spv", VK_SHADER_STAGE_ANY_HIT_BIT_KHR));
			anyHitIdx = static_cast<uint32_t>(shaderStagesRT.size()) - 1;
		}

		// Ray generation group
		{
			shaderStagesRT.push_back(loadShader(getShadersPath() + DIR_PATH + "raygen.rgen.spv", VK_SHADER_STAGE_RAYGEN_BIT_KHR));
			VkRayTracingShaderGroupCreateInfoKHR shaderGroup{};
			shaderGroup.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
			shaderGroup.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
			shaderGroup.generalShader = static_cast<uint32_t>(shaderStagesRT.size()) - 1;
			shaderGroup.closestHitShader = VK_SHADER_UNUSED_KHR;
			shaderGroup.anyHitShader = VK_SHADER_UNUSED_KHR;
			shaderGroup.intersectionShader = VK_SHADER_UNUSED_KHR;
			shaderGroups.push_back(shaderGroup);
		}
		// Miss group
		{
			shaderStagesRT.push_back(loadShader(getShadersPath() + DIR_PATH + "miss.rmiss.spv", VK_SHADER_STAGE_MISS_BIT_KHR));
			VkRayTracingShaderGroupCreateInfoKHR shaderGroup{};
			shaderGroup.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
			shaderGroup.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
			shaderGroup.generalShader = static_cast<uint32_t>(shaderStagesRT.size()) - 1;
			shaderGroup.closestHitShader = VK_SHADER_UNUSED_KHR;
			shaderGroup.anyHitShader = VK_SHADER_UNUSED_KHR;
			shaderGroup.intersectionShader = VK_SHADER_UNUSED_KHR;
			shaderGroups.push_back(shaderGroup);
			// Second shader for shadows
			shaderStagesRT.push_back(loadShader(getShadersPath() + DIR_PATH + "shadow.rmiss.spv", VK_SHADER_STAGE_MISS_BIT_KHR));
			shaderGroup.generalShader = static_cast<uint32_t>(shaderStagesRT.size()) - 1;
			shaderGroups.push_back(shaderGroup);
		}

		// Closest hit group : Reflection / Transmission
		{
			shaderStagesRT.push_back(loadShader(getShadersPath() + DIR_PATH + "closesthit.rchit.spv", VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR));
			VkRayTracingShaderGroupCreateInfoKHR shaderGroup{};
			shaderGroup.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
			shaderGroup.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
			shaderGroup.generalShader = VK_SHADER_UNUSED_KHR;
			shaderGroup.closestHitShader = static_cast<uint32_t>(shaderStagesRT.size()) - 1;
			shaderGroup.intersectionShader = VK_SHADER_UNUSED_KHR;
			shaderGroup.anyHitShader = anyHitIdx;
			shaderGroups.push_back(shaderGroup);
		}

		/*
			Create the ray tracing pipeline
		*/
		VkRayTracingPipelineCreateInfoKHR rayTracingPipelineCI{};
		rayTracingPipelineCI.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
		rayTracingPipelineCI.stageCount = static_cast<uint32_t>(shaderStagesRT.size());
		rayTracingPipelineCI.pStages = shaderStagesRT.data();
		rayTracingPipelineCI.groupCount = static_cast<uint32_t>(shaderGroups.size());
		rayTracingPipelineCI.pGroups = shaderGroups.data();
		rayTracingPipelineCI.maxPipelineRayRecursionDepth = 1;
		rayTracingPipelineCI.layout = pipelineLayouts.composition;
		VK_CHECK_RESULT(vkCreateRayTracingPipelinesKHR(device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &rayTracingPipelineCI, nullptr, &pipelines.composition));
	}

	// Prepare and initialize uniform buffer containing shader uniforms
	void createUniformBuffers()
	{
		// Shadow map vertex shader
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&uniformBuffers.shadowmap,
			sizeof(UniformDataShadowmap)));

		// Offscreen vertex shader
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&uniformBuffers.offscreen,
			sizeof(UniformDataOffscreen)));

		// Ray tracing shaders
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&uniformBuffers.composition,
			sizeof(UniformDataComposition)));

		// Map persistent
		VK_CHECK_RESULT(uniformBuffers.shadowmap.map());
		VK_CHECK_RESULT(uniformBuffers.offscreen.map());
		VK_CHECK_RESULT(uniformBuffers.composition.map());

		// Update
		updateUniformBufferOffscreen();
		updateUniformBufferComposition();
		updateUniformBufferShadowmap();
	}

	void updateUniformBufferShadowmap()
	{
		//shyun added begin
		// Matrix from light's point of view
		glm::mat4 depthProjectionMatrix = glm::perspective(glm::radians(lightFOV), 1.0f, zNear, zFar);
		glm::mat4 depthViewMatrix = glm::lookAt(glm::vec3(uniformDataComposition.lightPos[0]), glm::vec3(0.0f), glm::vec3(0, -1, 0));
		glm::mat4 depthModelMatrix = glm::mat4(1.0f);

		uniformDataShadowmap.depthMVP = depthProjectionMatrix * depthViewMatrix * depthModelMatrix;

		memcpy(uniformBuffers.shadowmap.mapped, &uniformDataShadowmap, sizeof(uniformDataShadowmap));
		//shyun added end
	}

	// Update matrices used for the offscreen rendering of the scene
	void updateUniformBufferOffscreen()
	{
		uniformDataOffscreen.modelMatrix = glm::mat4(1.0f);
		uniformDataOffscreen.modelViewProjectionMatrix = camera.matrices.perspective * camera.matrices.view;
		uniformDataOffscreen.modelMatrixInvTrans = glm::inverseTranspose(glm::mat4(1.0f));
		memcpy(uniformBuffers.offscreen.mapped, &uniformDataOffscreen, sizeof(UniformDataOffscreen));
	}

	void updateUniformBufferComposition()
	{
		uniformDataComposition.projInverse = glm::inverse(camera.matrices.perspective);
		uniformDataComposition.viewInverse = glm::inverse(camera.matrices.view);
		uniformDataComposition.depthBiasMVP = uniformDataShadowmap.depthMVP;
		// This value is used to accumulate multiple frames into the finale picture
		// It's required as ray tracing needs to do multiple passes for transparency
		// In this sample we use noise offset by this frame index to shoot rays for transparency into different directions
		// Once enough frames with random ray directions have been accumulated, it looks like proper transparency
		uniformDataComposition.frame++;

#if ASSET == 0
		uniformDataComposition.lightPos[0] = glm::vec4(0.0f + cos(glm::radians(timer * 360.0f)) * 50.0f,
			100.0f, 0.0f + sin(glm::radians(timer * 360.0f)) * 15.0f, 1.0f);
#elif ASSET == 1
		uniformDataComposition.lightPos[0] = glm::vec4(1.0f, 100.0f, 0.0f, 1.0f);
#elif ASSET == 2
		uniformDataComposition.lightPos[0] = glm::vec4(-0.911594f, 3.861007f, -1.508170f, 1.0f);
#endif

		memcpy(uniformBuffers.composition.mapped, &uniformDataComposition, sizeof(uniformDataComposition));
	}

	bool initVulkan() {
		// Auto-compile shaders
		// Remove 'pause' from batch file for speedy execution
		system("cd ..\\..\\shaders\\glsl\\base\\ && baseCompile.bat");
		std::cout << "\t...base project shaders compile completed.\n";
		system("cd ..\\..\\shaders\\glsl\\VulkanHybridShadowmap\\ && VulkanHybridShadowmapCompile.bat");
		std::cout << "\t...Vulkan Hybrid Shadow map project shaders compile completed.\n";

		bool result = VulkanRTBase::initVulkan();
		if (!result) {
			std::cout << "VulkanHybridShadowmap initVulkan failed.\n";
			return false;
		}

		return true;
	}

	void createShadowmapCommandBuffers()
	{
		// Create one command buffer for each swap chain image and reuse for rendering
		shadowmapCmdBuffers.resize(swapChain.imageCount);

		VkCommandBufferAllocateInfo cmdBufAllocateInfo =
			vks::initializers::commandBufferAllocateInfo(
				cmdPool,
				VK_COMMAND_BUFFER_LEVEL_PRIMARY,
				static_cast<uint32_t>(shadowmapCmdBuffers.size()));

		VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, shadowmapCmdBuffers.data()));
	}

	void createGeometryCommandBuffers()
	{
		// Create one command buffer for each swap chain image and reuse for rendering
		geometryCmdBuffers.resize(swapChain.imageCount);

		VkCommandBufferAllocateInfo cmdBufAllocateInfo =
			vks::initializers::commandBufferAllocateInfo(
				cmdPool,
				VK_COMMAND_BUFFER_LEVEL_PRIMARY,
				static_cast<uint32_t>(geometryCmdBuffers.size()));

		VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, geometryCmdBuffers.data()));
	}


	/// <External Memory Use>
	void transitionImageLayout(VkImage image, VkFormat format,
		VkImageLayout oldLayout, VkImageLayout newLayout, VkAccessFlags srcAccessMask, VkAccessFlags dstAccessMask, VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage, VkImageAspectFlags aspectFlags) {
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkImageMemoryBarrier barrier = {};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.oldLayout = oldLayout;
		barrier.newLayout = newLayout;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.image = image;
		barrier.subresourceRange.aspectMask = aspectFlags;
		barrier.subresourceRange.baseMipLevel = 0;
		barrier.subresourceRange.levelCount = mipLevels;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 1;

		VkPipelineStageFlags sourceStage;
		VkPipelineStageFlags destinationStage;

		barrier.srcAccessMask = srcAccessMask;
		barrier.dstAccessMask = dstAccessMask;

		vkCmdPipelineBarrier(commandBuffer, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);

		endSingleTimeCommands(commandBuffer);
	}

	void createImage(uint32_t width, uint32_t height, VkFormat format,
		VkImageTiling tiling, VkImageUsageFlags usage,
		VkMemoryPropertyFlags properties, VkImage& image,
		VkDeviceMemory& imageMemory, size_t& totalImageMemSize) {
		VkImageCreateInfo imageInfo = {};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.extent.width = width;
		imageInfo.extent.height = height;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = mipLevels;
		imageInfo.arrayLayers = 1;
		imageInfo.format = format;
		imageInfo.tiling = tiling;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageInfo.usage = usage;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		VkExternalMemoryImageCreateInfo vkExternalMemImageCreateInfo = {};
		vkExternalMemImageCreateInfo.sType =
			VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
		vkExternalMemImageCreateInfo.pNext = NULL;
#ifdef _WIN64
		vkExternalMemImageCreateInfo.handleTypes =
			VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
		vkExternalMemImageCreateInfo.handleTypes =
			VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
#endif

		imageInfo.pNext = &vkExternalMemImageCreateInfo;

		if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
			throw std::runtime_error("failed to create image!");
		}

		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(device, image, &memRequirements);

#ifdef _WIN64
		WindowsSecurityAttributes winSecurityAttributes;

		VkExportMemoryWin32HandleInfoKHR vulkanExportMemoryWin32HandleInfoKHR = {};
		vulkanExportMemoryWin32HandleInfoKHR.sType =
			VK_STRUCTURE_TYPE_EXPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
		vulkanExportMemoryWin32HandleInfoKHR.pNext = NULL;
		vulkanExportMemoryWin32HandleInfoKHR.pAttributes = &winSecurityAttributes;
		vulkanExportMemoryWin32HandleInfoKHR.dwAccess =
			DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE;
		vulkanExportMemoryWin32HandleInfoKHR.name = (LPCWSTR)NULL;
#endif
		VkExportMemoryAllocateInfoKHR vulkanExportMemoryAllocateInfoKHR = {};
		vulkanExportMemoryAllocateInfoKHR.sType =
			VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
#ifdef _WIN64
		vulkanExportMemoryAllocateInfoKHR.pNext =
			IsWindows8OrGreater() ? &vulkanExportMemoryWin32HandleInfoKHR : NULL;
		vulkanExportMemoryAllocateInfoKHR.handleTypes =
			IsWindows8OrGreater()
			? VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT
			: VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT;
#else
		vulkanExportMemoryAllocateInfoKHR.pNext = NULL;
		vulkanExportMemoryAllocateInfoKHR.handleTypes =
			VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
#endif
		VkMemoryRequirements vkMemoryRequirements = {};
		vkGetImageMemoryRequirements(device, image, &vkMemoryRequirements);

		VkMemoryAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.pNext = &vulkanExportMemoryAllocateInfoKHR;
		allocInfo.memoryTypeIndex = vulkanDevice->getMemoryType(vkMemoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

		totalImageMemSize = vkMemoryRequirements.size;

		if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) !=
			VK_SUCCESS) {
			throw std::runtime_error("failed to allocate image memory!");
		}

		vkBindImageMemory(device, image, imageMemory, 0);
	}

	void createTextureImage() {
		//Shadow Edge Texture
		createImage(geometryFrameBuf.width, geometryFrameBuf.width, VK_FORMAT_R32_SFLOAT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,	VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, shadowEdgeTextureImage, shadowEdgeTextureImageMemory, shadowEdgeImageMemSize);

		transitionImageLayout(shadowEdgeTextureImage, VK_FORMAT_R32_SFLOAT,			VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_IMAGE_ASPECT_COLOR_BIT);

		//Light Texture
		createImage(geometryFrameBuf.width, geometryFrameBuf.height, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, lightTextureImage, lightTextureImageMemory, lightImageMemSize);

		transitionImageLayout(lightTextureImage, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
	}

	VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags flags) {
		VkImageViewCreateInfo viewInfo = {};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = format;
		viewInfo.subresourceRange.aspectMask = flags;
		viewInfo.subresourceRange.baseMipLevel = 0;
		viewInfo.subresourceRange.levelCount = mipLevels;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount = 1;

		VkImageView imageView;
		if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) !=
			VK_SUCCESS) {
			throw std::runtime_error("failed to create texture image view!");
		}

		return imageView;
	}

	void createTextureSampler(VkSampler &textureSampler) {
		VkSamplerCreateInfo samplerInfo = {};
		samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerInfo.magFilter = VK_FILTER_LINEAR;
		samplerInfo.minFilter = VK_FILTER_LINEAR;
		samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
		samplerInfo.unnormalizedCoordinates = VK_FALSE;
		samplerInfo.compareEnable = VK_FALSE;
		samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
		samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		samplerInfo.minLod = 0;  // Optional
		samplerInfo.maxLod = static_cast<float>(mipLevels);
		samplerInfo.mipLodBias = 0;  // Optional

		if (vkCreateSampler(device, &samplerInfo, nullptr, &textureSampler) !=
			VK_SUCCESS) {
			throw std::runtime_error("failed to create texture sampler!");
		}
	}

	void cudaVkImportSemaphore() {
		cudaExternalSemaphoreHandleDesc externalSemaphoreHandleDesc;
		memset(&externalSemaphoreHandleDesc, 0,
			sizeof(externalSemaphoreHandleDesc));
#ifdef _WIN64
		externalSemaphoreHandleDesc.type =
			IsWindows8OrGreater() ? cudaExternalSemaphoreHandleTypeOpaqueWin32
			: cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt;
		externalSemaphoreHandleDesc.handle.win32.handle = getVkSemaphoreHandle(
			IsWindows8OrGreater()
			? VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT
			: VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT,
			cudaUpdateDoneVkSemaphore);
#else
		externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
		externalSemaphoreHandleDesc.handle.fd = getVkSemaphoreHandle(
			VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT, cudaUpdateVkSemaphore);
#endif
		externalSemaphoreHandleDesc.flags = 0;

		CUDA_CALL(cudaImportExternalSemaphore(&cudaUpdateDoneSemaphore, &externalSemaphoreHandleDesc));

		memset(&externalSemaphoreHandleDesc, 0, sizeof(externalSemaphoreHandleDesc));
#ifdef _WIN64
		externalSemaphoreHandleDesc.type =
			IsWindows8OrGreater() ? cudaExternalSemaphoreHandleTypeOpaqueWin32
			: cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt;
		;
		externalSemaphoreHandleDesc.handle.win32.handle = getVkSemaphoreHandle(
			IsWindows8OrGreater()
			? VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT
			: VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT,
			geometryDoneVkSemaphore);
#else
		externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
		externalSemaphoreHandleDesc.handle.fd = getVkSemaphoreHandle(
			VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT, vkUpdateCudaSemaphore);
#endif
		externalSemaphoreHandleDesc.flags = 0;
		CUDA_CALL(cudaImportExternalSemaphore(&geometryDoneSemaphore,
			&externalSemaphoreHandleDesc));

		memset(&externalSemaphoreHandleDesc, 0, sizeof(externalSemaphoreHandleDesc));
#ifdef _WIN64
		externalSemaphoreHandleDesc.type =
			IsWindows8OrGreater() ? cudaExternalSemaphoreHandleTypeOpaqueWin32
			: cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt;
		;
		externalSemaphoreHandleDesc.handle.win32.handle = getVkSemaphoreHandle(
			IsWindows8OrGreater()
			? VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT
			: VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT,
			shadowDoneVkSemaphore);
#else
		externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
		externalSemaphoreHandleDesc.handle.fd = getVkSemaphoreHandle(
			VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT, vkUpdateCudaSemaphore);
#endif
		externalSemaphoreHandleDesc.flags = 0;
		CUDA_CALL(cudaImportExternalSemaphore(&shadowDoneSemaphore,
			&externalSemaphoreHandleDesc));

		printf("CUDA Imported Vulkan semaphore\n");
	}

	void cudaVkImportImageMem() {
		cudaExternalMemoryHandleDesc cudaExtMemHandleDesc;

		/// <Shadow Map>
		memset(&cudaExtMemHandleDesc, 0, sizeof(cudaExtMemHandleDesc));
#ifdef _WIN64
		cudaExtMemHandleDesc.type =
			IsWindows8OrGreater() ? cudaExternalMemoryHandleTypeOpaqueWin32
			: cudaExternalMemoryHandleTypeOpaqueWin32Kmt;
		cudaExtMemHandleDesc.handle.win32.handle = getVkImageMemHandle(
			IsWindows8OrGreater()
			? VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT
			: VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT, shadowmapFrameBuf.depth.mem);
#else
		cudaExtMemHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;

		cudaExtMemHandleDesc.handle.fd =
			getVkImageMemHandle(VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR);
#endif
		cudaExtMemHandleDesc.size = shadowMapImageMemSize;

		CUDA_CALL(cudaImportExternalMemory(&cudaExtMemShadowMapImageBuffer, &cudaExtMemHandleDesc));
		/// </Shadow Map>

		/// <Shadow Edge>
		memset(&cudaExtMemHandleDesc, 0, sizeof(cudaExtMemHandleDesc));
#ifdef _WIN64
		cudaExtMemHandleDesc.type =
			IsWindows8OrGreater() ? cudaExternalMemoryHandleTypeOpaqueWin32
			: cudaExternalMemoryHandleTypeOpaqueWin32Kmt;
		cudaExtMemHandleDesc.handle.win32.handle = getVkImageMemHandle(
			IsWindows8OrGreater()
			? VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT
			: VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT, shadowEdgeTextureImageMemory);
#else
		cudaExtMemHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;

		cudaExtMemHandleDesc.handle.fd =
			getVkImageMemHandle(VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR);
#endif
		cudaExtMemHandleDesc.size = shadowEdgeImageMemSize;

		CUDA_CALL(cudaImportExternalMemory(&cudaExtMemShadowEdgeImageBuffer, &cudaExtMemHandleDesc));
		/// </Shadow Edge>

		/// <Position>
		memset(&cudaExtMemHandleDesc, 0, sizeof(cudaExtMemHandleDesc));
#ifdef _WIN64
		cudaExtMemHandleDesc.type =
			IsWindows8OrGreater() ? cudaExternalMemoryHandleTypeOpaqueWin32
			: cudaExternalMemoryHandleTypeOpaqueWin32Kmt;
		cudaExtMemHandleDesc.handle.win32.handle = getVkImageMemHandle(
			IsWindows8OrGreater()
			? VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT
			: VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT, geometryFrameBuf.position.mem);
#else
		cudaExtMemHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;

		cudaExtMemHandleDesc.handle.fd =
			getVkImageMemHandle(VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR);
#endif
		cudaExtMemHandleDesc.size = positionImageMemSize;

		CUDA_CALL(cudaImportExternalMemory(&cudaExtMemPositionImageBuffer, &cudaExtMemHandleDesc));
		/// </Position>

		/// <Light>
		memset(&cudaExtMemHandleDesc, 0, sizeof(cudaExtMemHandleDesc));
#ifdef _WIN64
		cudaExtMemHandleDesc.type =
			IsWindows8OrGreater() ? cudaExternalMemoryHandleTypeOpaqueWin32
			: cudaExternalMemoryHandleTypeOpaqueWin32Kmt;
		cudaExtMemHandleDesc.handle.win32.handle = getVkImageMemHandle(
			IsWindows8OrGreater()
			? VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT
			: VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT, lightTextureImageMemory);
#else
		cudaExtMemHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;

		cudaExtMemHandleDesc.handle.fd =
			getVkImageMemHandle(VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR);
#endif
		cudaExtMemHandleDesc.size = lightImageMemSize;

		CUDA_CALL(cudaImportExternalMemory(&cudaExtMemLightImageBuffer, &cudaExtMemHandleDesc));
		/// </Light>

		cudaExternalMemoryMipmappedArrayDesc shadowMapExternalMemoryMipmappedArrayDesc;
		cudaExternalMemoryMipmappedArrayDesc shadowEdgeExternalMemoryMipmappedArrayDesc;
		cudaExternalMemoryMipmappedArrayDesc positionExternalMemoryMipmappedArrayDesc;
		cudaExternalMemoryMipmappedArrayDesc lightExternalMemoryMipmappedArrayDesc;

		memset(&shadowMapExternalMemoryMipmappedArrayDesc, 0, sizeof(shadowMapExternalMemoryMipmappedArrayDesc));

		memset(&shadowEdgeExternalMemoryMipmappedArrayDesc, 0, sizeof(shadowEdgeExternalMemoryMipmappedArrayDesc));

		memset(&positionExternalMemoryMipmappedArrayDesc, 0, sizeof(positionExternalMemoryMipmappedArrayDesc));

		memset(&lightExternalMemoryMipmappedArrayDesc, 0, sizeof(lightExternalMemoryMipmappedArrayDesc));

		cudaExtent shadowMapExtent = make_cudaExtent(static_cast<size_t>(shadowMapSize), static_cast<size_t>(shadowMapSize), 0);
		cudaExtent colorExtent = make_cudaExtent(static_cast<size_t>(geometryFrameBuf.width), static_cast<size_t>(geometryFrameBuf.height), 0);

		cudaChannelFormatDesc formatDesc;
		formatDesc.x = 32;
		formatDesc.y = 0;
		formatDesc.z = 0;
		formatDesc.w = 0;
		formatDesc.f = cudaChannelFormatKindFloat;

		shadowMapExternalMemoryMipmappedArrayDesc.offset = 0;
		shadowMapExternalMemoryMipmappedArrayDesc.formatDesc = formatDesc;
		shadowMapExternalMemoryMipmappedArrayDesc.extent = shadowMapExtent;
		shadowMapExternalMemoryMipmappedArrayDesc.flags = 0;
		shadowMapExternalMemoryMipmappedArrayDesc.numLevels = mipLevels;

		shadowEdgeExternalMemoryMipmappedArrayDesc.offset = 0;
		shadowEdgeExternalMemoryMipmappedArrayDesc.formatDesc = formatDesc;
		shadowEdgeExternalMemoryMipmappedArrayDesc.extent = colorExtent;
		shadowEdgeExternalMemoryMipmappedArrayDesc.flags = 0;
		shadowEdgeExternalMemoryMipmappedArrayDesc.numLevels = mipLevels;

		formatDesc.x = 16;
		formatDesc.y = 16;
		formatDesc.z = 16;
		formatDesc.w = 16;
		formatDesc.f = cudaChannelFormatKindFloat;

		positionExternalMemoryMipmappedArrayDesc.offset = 0;
		positionExternalMemoryMipmappedArrayDesc.formatDesc = formatDesc;
		positionExternalMemoryMipmappedArrayDesc.extent = colorExtent;
		positionExternalMemoryMipmappedArrayDesc.flags = 0;
		positionExternalMemoryMipmappedArrayDesc.numLevels = mipLevels;

		lightExternalMemoryMipmappedArrayDesc.offset = 0;
		lightExternalMemoryMipmappedArrayDesc.formatDesc = formatDesc;
		lightExternalMemoryMipmappedArrayDesc.extent = colorExtent;
		lightExternalMemoryMipmappedArrayDesc.flags = 0;
		lightExternalMemoryMipmappedArrayDesc.numLevels = mipLevels;

		CUDA_CALL(cudaExternalMemoryGetMappedMipmappedArray(&cudaMipmappedImageArrayShadowMap, cudaExtMemShadowMapImageBuffer, &shadowMapExternalMemoryMipmappedArrayDesc));

		CUDA_CALL(cudaExternalMemoryGetMappedMipmappedArray(&cudaMipmappedImageArrayShadowEdge, cudaExtMemShadowEdgeImageBuffer, &shadowEdgeExternalMemoryMipmappedArrayDesc));

		CUDA_CALL(cudaExternalMemoryGetMappedMipmappedArray(&cudaMipmappedImageArrayPosition, cudaExtMemPositionImageBuffer, &positionExternalMemoryMipmappedArrayDesc));

		CUDA_CALL(cudaExternalMemoryGetMappedMipmappedArray(&cudaMipmappedImageArrayLight, cudaExtMemLightImageBuffer, &lightExternalMemoryMipmappedArrayDesc));

		for (int mipLevelIdx = 0; mipLevelIdx < mipLevels; mipLevelIdx++) {
			cudaArray_t cudaMipLevelArrayShadowEdge, cudaMipLevelArrayLight;
			cudaResourceDesc resourceDesc;

			/// <Shadow Edge>
			CUDA_CALL(cudaGetMipmappedArrayLevel(&cudaMipLevelArrayShadowEdge, cudaMipmappedImageArrayShadowEdge, mipLevelIdx));

			memset(&resourceDesc, 0, sizeof(resourceDesc));
			resourceDesc.resType = cudaResourceTypeArray;
			resourceDesc.res.array.array = cudaMipLevelArrayShadowEdge;

			cudaSurfaceObject_t surfaceObjectShadowEdge;
			CUDA_CALL(cudaCreateSurfaceObject(&surfaceObjectShadowEdge, &resourceDesc));

			surfaceObjectListShadowEdge.push_back(surfaceObjectShadowEdge);
			/// </Shadow Edge>

			/// <Light>
			CUDA_CALL(cudaGetMipmappedArrayLevel(&cudaMipLevelArrayLight, cudaMipmappedImageArrayLight, mipLevelIdx));

			memset(&resourceDesc, 0, sizeof(resourceDesc));
			resourceDesc.resType = cudaResourceTypeArray;
			resourceDesc.res.array.array = cudaMipLevelArrayLight;

			cudaSurfaceObject_t surfaceObjectLight;
			CUDA_CALL(cudaCreateSurfaceObject(&surfaceObjectLight, &resourceDesc));

			surfaceObjectListLight.push_back(surfaceObjectLight);
			/// </Light>
		}

		/// <Shadow Map>
		cudaResourceDesc resDescr;
		memset(&resDescr, 0, sizeof(cudaResourceDesc));

		resDescr.resType = cudaResourceTypeMipmappedArray;
		resDescr.res.mipmap.mipmap = cudaMipmappedImageArrayShadowMap;

		cudaTextureDesc texDescr;
		memset(&texDescr, 0, sizeof(cudaTextureDesc));

		texDescr.normalizedCoords = true;
		texDescr.filterMode = cudaFilterModePoint;
		texDescr.mipmapFilterMode = cudaFilterModePoint;

		texDescr.addressMode[0] = cudaAddressModeWrap;
		texDescr.addressMode[1] = cudaAddressModeWrap;

		texDescr.maxMipmapLevelClamp = float(mipLevels - 1);

		texDescr.readMode = cudaReadModeElementType;

		CUDA_CALL(cudaCreateTextureObject(&textureObjShadowMap, &resDescr, &texDescr, NULL));
		/// </Shadow Map>

		/// <Position>
		resDescr.res.mipmap.mipmap = cudaMipmappedImageArrayPosition;
		CUDA_CALL(cudaCreateTextureObject(&textureObjPosition, &resDescr, &texDescr, NULL));
		/// </Position>

		/// <Shadow Edge>
		CUDA_CALL(cudaMalloc((void**)&d_surfaceObjectListShadowEdge, sizeof(cudaSurfaceObject_t) * mipLevels));
		CUDA_CALL(cudaMemcpy(d_surfaceObjectListShadowEdge, surfaceObjectListShadowEdge.data(), sizeof(cudaSurfaceObject_t) * mipLevels, cudaMemcpyHostToDevice));
		/// </Shadow Edge>

		/// <Light>
		CUDA_CALL(cudaMalloc((void**)&d_surfaceObjectListLight, sizeof(cudaSurfaceObject_t) * mipLevels));
		CUDA_CALL(cudaMemcpy(d_surfaceObjectListLight, surfaceObjectListLight.data(), sizeof(cudaSurfaceObject_t) * mipLevels, cudaMemcpyHostToDevice));
		/// </Light>

		printf("CUDA Kernel Vulkan image buffer\n");
	}

	void createSyncObjectsExt() {
		VkSemaphoreCreateInfo semaphoreInfo = {};
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		memset(&semaphoreInfo, 0, sizeof(semaphoreInfo));
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

#ifdef _WIN64
		WindowsSecurityAttributes winSecurityAttributes;

		VkExportSemaphoreWin32HandleInfoKHR vulkanExportSemaphoreWin32HandleInfoKHR = {};
		vulkanExportSemaphoreWin32HandleInfoKHR.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR;
		vulkanExportSemaphoreWin32HandleInfoKHR.pNext = NULL;
		vulkanExportSemaphoreWin32HandleInfoKHR.pAttributes = &winSecurityAttributes;
		vulkanExportSemaphoreWin32HandleInfoKHR.dwAccess = DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE;
		vulkanExportSemaphoreWin32HandleInfoKHR.name = (LPCWSTR)NULL;
#endif
		VkExportSemaphoreCreateInfoKHR vulkanExportSemaphoreCreateInfo = {};
		vulkanExportSemaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR;
#ifdef _WIN64
		vulkanExportSemaphoreCreateInfo.pNext = IsWindows8OrGreater() ? &vulkanExportSemaphoreWin32HandleInfoKHR : NULL;
		vulkanExportSemaphoreCreateInfo.handleTypes = IsWindows8OrGreater()?
			VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT : VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT;
#else
		vulkanExportSemaphoreCreateInfo.pNext = NULL;
		vulkanExportSemaphoreCreateInfo.handleTypes =
			VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
		semaphoreInfo.pNext = &vulkanExportSemaphoreCreateInfo;

		if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &cudaUpdateDoneVkSemaphore) != VK_SUCCESS ||
			vkCreateSemaphore(device, &semaphoreInfo, nullptr, &geometryDoneVkSemaphore) != VK_SUCCESS || 
			vkCreateSemaphore(device, &semaphoreInfo, nullptr, &shadowDoneVkSemaphore) != VK_SUCCESS) {
			throw std::runtime_error("failed to create synchronization objects for a CUDA-Vulkan!");
		}
	}

	void cudaVkSemaphoreWait() {
		cudaExternalSemaphore_t semaphores[2] = { geometryDoneSemaphore, shadowDoneSemaphore };
		cudaExternalSemaphoreWaitParams extSemaphoreWaitParams[2];

		memset(&extSemaphoreWaitParams, 0, sizeof(extSemaphoreWaitParams));

		extSemaphoreWaitParams[0].params.fence.value = 0;
		extSemaphoreWaitParams[0].flags = 0;

		extSemaphoreWaitParams[1].params.fence.value = 0;
		extSemaphoreWaitParams[1].flags = 0;

		CUDA_CALL(cudaWaitExternalSemaphoresAsync(semaphores, extSemaphoreWaitParams, 2, streamToRun));
	}

	void cudaVkSemaphoreSignal(cudaExternalSemaphore_t& extSemaphore) {
		cudaExternalSemaphoreSignalParams extSemaphoreSignalParams;
		memset(&extSemaphoreSignalParams, 0, sizeof(extSemaphoreSignalParams));

		extSemaphoreSignalParams.params.fence.value = 0;
		extSemaphoreSignalParams.flags = 0;

		CUDA_CALL(cudaSignalExternalSemaphoresAsync(&extSemaphore, &extSemaphoreSignalParams, 1, streamToRun));
	}

	void initCuda() {
		setCudaVkDevice();
		CUDA_CALL(cudaStreamCreate(&streamToRun));
		cudaVkImportImageMem();
		cudaVkImportSemaphore();
	}

	void cudaUpdateVkImage() {
		cudaVkSemaphoreWait();

		sobelFilter(d_surfaceObjectListShadowEdge, d_surfaceObjectListLight, textureObjShadowMap, textureObjPosition, streamToRun, mipLevels, geometryFrameBuf.width, geometryFrameBuf.height);

		cudaVkSemaphoreSignal(cudaUpdateDoneSemaphore);
	}

	int setCudaVkDevice() {
		int current_device = 0;
		int device_count = 0;
		int devices_prohibited = 0;

		cudaDeviceProp deviceProp;
		CUDA_CALL(cudaGetDeviceCount(&device_count));

		if (device_count == 0) {
			fprintf(stderr, "CUDA error: no devices supporting CUDA.\n");
			exit(EXIT_FAILURE);
		}

		// Find the GPU which is selected by Vulkan
		while (current_device < device_count) {
			cudaGetDeviceProperties(&deviceProp, current_device);

			if ((deviceProp.computeMode != cudaComputeModeProhibited)) {
				// Compare the cuda device UUID with vulkan UUID
				int ret = memcmp(&deviceProp.uuid, &vkDeviceUUID, VK_UUID_SIZE);
				if (ret == 0) {
					CUDA_CALL(cudaSetDevice(current_device));
					CUDA_CALL(cudaGetDeviceProperties(&deviceProp, current_device));
					printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n",
						current_device, deviceProp.name, deviceProp.major,
						deviceProp.minor);

					return current_device;
				}

			}
			else {
				devices_prohibited++;
			}

			current_device++;
		}

		if (devices_prohibited == device_count) {
			fprintf(stderr, "CUDA error: No Vulkan-CUDA Interop capable GPU found.\n");
			exit(EXIT_FAILURE);
		}

		return -1;
	}

	/// </External Memory Use>

	void prepare()
	{
		VulkanRTCommon::prepare();

		createShadowmapCommandBuffers();
		createGeometryCommandBuffers();

		loadAssets();

		// Create the acceleration structures used to render the ray traced scene
		createBottomLevelAccelerationStructure();
		createTopLevelAccelerationStructure();

		createStorageImage(swapChain.colorFormat, { width, height, 1 });

		prepareShadowmapFramebuffer();
		prepareGeometryFramebuffer();

		/// <External Memory Use>
		createTextureImage();
		shadowEdgeTextureImageView = createImageView(shadowEdgeTextureImage, VK_FORMAT_R32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT);
		createTextureSampler(shadowEdgeTextureSampler);

		lightTextureImageView = createImageView(lightTextureImage, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT);
		createTextureSampler(lightTextureSampler);

		getKhrExtensionsFn();
		createSyncObjectsExt();
		initCuda();
		/// </External Memory Use>

		createUniformBuffers();
		createDescriptorSets();
		preparePipelines();
		createShaderBindingTables();

		buildShadowmapCommandBuffers();
		buildGeometryCommandBuffer();
		buildCommandBuffers();

		prepared = true;
	}

	void draw()
	{
		VulkanRTBase::prepareFrame();

		VkPipelineStageFlags offscreenWaitStages = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

		submitInfo.pNext = NULL;
		submitInfo.pWaitDstStageMask = &offscreenWaitStages;
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = &semaphores.presentComplete;
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &shadowDoneVkSemaphore;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &shadowmapCmdBuffers[currentBuffer];
		VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));

		submitInfo.pNext = NULL;
		submitInfo.pWaitDstStageMask = &offscreenWaitStages;
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = &shadowDoneVkSemaphore;
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &geometryDoneVkSemaphore;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &geometryCmdBuffers[currentBuffer];
		VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));

		cudaUpdateVkImage();

		VkSemaphore waitSemaphores[] = { shadowDoneVkSemaphore, geometryDoneVkSemaphore, cudaUpdateDoneVkSemaphore };

		VkPipelineStageFlags lightingWaitStages[] = {
			VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
			VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
			VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR
		};

		submitInfo.pNext = NULL;
		submitInfo.pWaitDstStageMask = lightingWaitStages;
		submitInfo.waitSemaphoreCount = 3;
		submitInfo.pWaitSemaphores = waitSemaphores;
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &semaphores.renderComplete;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];
		VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));

		VulkanRTBase::submitFrame();

		currentBuffer = (currentBuffer + 1) % 3;
	}

	virtual void render()
	{
		if (!prepared)
			return;
		updateUniformBufferShadowmap();
		updateUniformBufferOffscreen();
		updateUniformBufferComposition();
		draw();
	}
};

VULKAN_HYBRID_SHADOWMAP()
