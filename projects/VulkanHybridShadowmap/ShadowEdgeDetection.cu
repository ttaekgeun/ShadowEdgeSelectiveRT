#include <stdio.h>
#include <stdlib.h>
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include "ShadowEdgeDetection.cuh"

namespace cg = cooperative_groups;

#define RADIUS 1
extern __shared__ float LocalBlock[];

// This will output the proper CUDA error strings in the event that a CUDA host
// call returns an error
__device__ float ComputeSobel(float ul,  // upper left
	float um,  // upper middle
	float ur,  // upper right
	float ml,  // middle left
	float mm,  // middle (unused)
	float mr,  // middle right
	float ll,  // lower left
	float lm,  // lower middle
	float lr  // lower right
	)
{
	double Horz = ur + 2 * mr + lr - ul - 2 * ml - ll;
	double Vert = ul + 2 * um + ur - ll - 2 * lm - lr;
	double Sum = (double)((fabs((float)Horz) + fabs((float)Vert)));

	if (Sum < 0) {
		return 0;
	}
	else if (Sum > 0xffff) {
		return 0xffff;
	}

	return (float)Sum;
}

__device__ void matMul4x4(float* C, const float* A, const float* B) {
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++) {
			float tmp = 0.0;
			for (int k = 0; k < 4; k++)
				tmp += A[i * 4 + k] * B[4 * k + j];
			C[i * 4 + j] = tmp;
	}
}

__device__ float4 matMul4xVec4(const float* A, float4 B) {
	float4 tmp = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	tmp.x += A[0] * B.x + A[1] * B.y + A[2] * B.z + A[3] * B.w;
	tmp.y += A[4] * B.x + A[5] * B.y + A[6] * B.z + A[7] * B.w;
	tmp.z += A[8] * B.x + A[9] * B.y + A[10] * B.z + A[11] * B.w;
	tmp.w += A[12] * B.x + A[13] * B.y + A[14] * B.z + A[15] * B.w;
	return tmp;
}

__device__ bool textureProj(cudaTextureObject_t shadowMapTexture, float4 shadowCoord, float mipLevelIdx)
{
	bool shadowed = false;

	if (shadowCoord.z > -1.0f && shadowCoord.z < 1.0f)
	{
		float dist = tex2DLod<float>(shadowMapTexture, shadowCoord.x, shadowCoord.y, mipLevelIdx);
		if (shadowCoord.w > 0.0f && dist < shadowCoord.z)
		{
			shadowed = true;
		}
	}
	return shadowed;
}

__device__ float calculateShadow(cudaTextureObject_t positionTexture, cudaTextureObject_t shadowMapTexture, uint32_t mipLevelIdx, unsigned int x, unsigned int y, float px, float py)
{
	float4 pos = tex2DLod<float4>(positionTexture, x * px, y * py, (float)mipLevelIdx);
	pos.w = 1.0f;

	float4 temp = matMul4xVec4(d_depthBiasMVP, pos);
	float4 shadowCoord = matMul4xVec4(d_biasMat, temp);

	shadowCoord = make_float4(shadowCoord.x / shadowCoord.w, shadowCoord.y / shadowCoord.w, shadowCoord.z / shadowCoord.w, 1.0f);

	bool shadowed = textureProj(shadowMapTexture, shadowCoord, (float) mipLevelIdx);

	float shadow = shadowed;

	return shadow;
}

__global__ void sobelFilterKernel(cudaSurfaceObject_t* __restrict shadowEdgeTexture, cudaSurfaceObject_t* __restrict lightTexture, cudaTextureObject_t shadowMapTexture, cudaTextureObject_t positionTexture, size_t mipLevels, int width, int height, int shadowMapSize)
{
	cg::thread_block cta = cg::this_thread_block();

	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	float px = 1.0f / width;
	float py = 1.0f / height;

	for (uint32_t mipLevelIdx = 0; mipLevelIdx < mipLevels; mipLevelIdx++)
	{
		float shadow = calculateShadow(positionTexture, shadowMapTexture, mipLevelIdx, x, y, px, py);
		LocalBlock[(threadIdx.x + 1) + (blockDim.x + 2) * (threadIdx.y + 1)] = shadow;

		float4 temp = make_float4(shadow, 0.0f, 0.0f, 0.0f);
		surf2Dwrite(temp, lightTexture[mipLevelIdx], x * 16, y);

		int side_left = 0, side_right = 0;
		if (threadIdx.x < 1)
		{
			float shadow = calculateShadow(positionTexture, shadowMapTexture, mipLevelIdx, x - 1, y, px, py);
			LocalBlock[(threadIdx.x) + (blockDim.x + 2) * (threadIdx.y + 1)] = shadow;
			side_left = 1;
		}
		else if (threadIdx.x >= blockDim.x - 1)
		{
			float shadow = calculateShadow(positionTexture, shadowMapTexture, mipLevelIdx, x + 1, y, px, py);
			LocalBlock[(threadIdx.x + 2) + (blockDim.x + 2) * (threadIdx.y + 1)] = shadow;
			side_right = 1;
		}

		if (threadIdx.y < 1)
		{
			float shadow = calculateShadow(positionTexture, shadowMapTexture, mipLevelIdx, x, y - 1, px, py);
			LocalBlock[(threadIdx.x + 1) + (blockDim.x + 2) * (threadIdx.y)] = shadow;
			if (side_left == 1) {
				shadow = calculateShadow(positionTexture, shadowMapTexture,  mipLevelIdx, x - 1, y - 1, px, py);
				LocalBlock[(threadIdx.x) + (blockDim.x + 2) * (threadIdx.y)] = shadow;
			}
			if (side_right == 1) {
				shadow = calculateShadow(positionTexture, shadowMapTexture, mipLevelIdx, x + 1, y - 1, px, py);
				LocalBlock[(threadIdx.x + 2) + (blockDim.x + 2) * (threadIdx.y)] = shadow;
			}
		}
		else if (threadIdx.y >= blockDim.y - 1)
		{
			float shadow = calculateShadow(positionTexture, shadowMapTexture, mipLevelIdx, x, y + 1, px, py);
			LocalBlock[(threadIdx.x + 1) + (blockDim.x + 2) * (threadIdx.y + 2)] = shadow;

			if (side_left == 1) {
				shadow = calculateShadow(positionTexture, shadowMapTexture, mipLevelIdx, x - 1, y + 1, px, py);
				LocalBlock[(threadIdx.x) + (blockDim.x + 2) * (threadIdx.y + 2)] = shadow;
			}
			if (side_right == 1) {
				shadow = calculateShadow(positionTexture, shadowMapTexture, mipLevelIdx, x + 1, y + 1, px, py);
				LocalBlock[(threadIdx.x + 2) + (blockDim.x + 2) * (threadIdx.y + 2)] = shadow;
			}
		}
		cg::sync(cta);

		float pix00 = LocalBlock[(threadIdx.x) + (blockDim.x + 2) * (threadIdx.y)];
		float pix01 = LocalBlock[(threadIdx.x + 1) + (blockDim.x + 2) * (threadIdx.y)];
		float pix02 = LocalBlock[(threadIdx.x + 2) + (blockDim.x + 2) * (threadIdx.y)];
		float pix10 = LocalBlock[(threadIdx.x) + (blockDim.x + 2) * (threadIdx.y + 1)];
		float pix11 = LocalBlock[(threadIdx.x + 1) + (blockDim.x + 2) * (threadIdx.y + 1)];
		float pix12 = LocalBlock[(threadIdx.x + 2) + (blockDim.x + 2) * (threadIdx.y + 1)];
		float pix20 = LocalBlock[(threadIdx.x) + (blockDim.x + 2) * (threadIdx.y + 2)];
		float pix21 = LocalBlock[(threadIdx.x + 1) + (blockDim.x + 2) * (threadIdx.y + 2)];
		float pix22 = LocalBlock[(threadIdx.x + 2) + (blockDim.x + 2) * (threadIdx.y + 2)];

		float out = ComputeSobel(pix00, pix01, pix02, pix10, pix11, pix12, pix20, pix21, pix22);

		if (x  < width && y < height) {
			surf2Dwrite(out, shadowEdgeTexture[mipLevelIdx], x * 4, y);
		}
		cg::sync(cta);
	}
}

// Wrapper for the __global__ call that sets up the texture and threads
extern "C" void sobelFilter(cudaSurfaceObject_t* shadowEdgeTexture, cudaSurfaceObject_t* lightTexture, cudaTextureObject_t shadowMapTexture, cudaTextureObject_t positionTexture, cudaStream_t streamToRun, size_t mipLevels, int width, int height, int shadowMapSize, float* projInverseMat, float* viewInverseMat, float* depthBiasMVPMat, float* lightPos) {
	
	CUDA_CALL(cudaMemcpyToSymbol(d_projInverse, projInverseMat, sizeof(float) * 16));
	CUDA_CALL(cudaMemcpyToSymbol(d_viewInverse, viewInverseMat, sizeof(float) * 16));
	CUDA_CALL(cudaMemcpyToSymbol(d_depthBiasMVP, depthBiasMVPMat, sizeof(float) * 16));
	CUDA_CALL(cudaMemcpyToSymbol(d_lightPos, lightPos, sizeof(float) * 4));



	dim3 threads(32, 32);

	dim3 blocks = dim3((width + threads.x - 1) / (threads.x), (height + threads.y - 1) / threads.y);
	int sharedMem = (threads.x + 2) * (threads.y + 2) * sizeof(float);

	sobelFilterKernel << <blocks, threads, sharedMem, streamToRun >> > (shadowEdgeTexture, lightTexture, shadowMapTexture, positionTexture, mipLevels, width, height, shadowMapSize);
}
