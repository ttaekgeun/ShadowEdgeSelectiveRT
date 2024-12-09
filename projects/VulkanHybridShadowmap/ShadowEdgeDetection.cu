#include <stdio.h>
#include <stdlib.h>
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include "ShadowEdgeDetection.cuh"

namespace cg = cooperative_groups;

#define RADIUS 1

//#ifdef FIXED_BLOCKWIDTH
//#define BlockWidth 80
//#define SharedPitch 384
//#endif

// convert floating point rgba color to 16-bit integer
__device__ unsigned short FloatToUShort(float value) {
	//rgba.x = __saturatef(rgba.x);  // clamp to [0.0, 1.0]
	//rgba.y = __saturatef(rgba.y);
	//rgba.z = __saturatef(rgba.z);
	//rgba.w = __saturatef(rgba.w);
	//return ((unsigned int)(rgba.w * 255.0f) << 24) |
	//	((unsigned int)(rgba.z * 255.0f) << 16) |
	//	((unsigned int)(rgba.y * 255.0f) << 8) |
	//	((unsigned int)(rgba.x * 255.0f));
	value = __saturatef(value);
	return (unsigned short)(value * 65535.0f);  // 65535 == 2^16 - 1
}

//// This will output the proper CUDA error strings in the event that a CUDA host
//// call returns an error
//__device__ unsigned char ComputeSobel(unsigned char ul,  // upper left
//	unsigned char um,  // upper middle
//	unsigned char ur,  // upper right
//	unsigned char ml,  // middle left
//	unsigned char mm,  // middle (unused)
//	unsigned char mr,  // middle right
//	unsigned char ll,  // lower left
//	unsigned char lm,  // lower middle
//	unsigned char lr,  // lower right
//	float fScale) {
//	short Horz = ur + 2 * mr + lr - ul - 2 * ml - ll;
//	short Vert = ul + 2 * um + ur - ll - 2 * lm - lr;
//	short Sum = (short)(fScale * (abs((int)Horz) + abs((int)Vert)));
//
//	if (Sum < 0) {
//		return 0;
//	}
//	else if (Sum > 0xff) {
//		return 0xff;
//	}
//
//	return (unsigned char)Sum;
//}
//
//__global__ void SobelShared(uchar4* pSobelOriginal, unsigned short SobelPitch,
//#ifndef FIXED_BLOCKWIDTH
//	short BlockWidth, short SharedPitch,
//#endif
//	short w, short h, float fScale,
//	cudaTextureObject_t tex) {
//	// Handle to thread block group
//	cg::thread_block cta = cg::this_thread_block();
//	short u = 4 * blockIdx.x * BlockWidth;
//	short v = blockIdx.y * blockDim.y + threadIdx.y;
//	short ib;
//
//	int SharedIdx = threadIdx.y * SharedPitch;
//
//	for (ib = threadIdx.x; ib < BlockWidth + 2 * RADIUS; ib += blockDim.x) {
//		LocalBlock[SharedIdx + 4 * ib + 0] = tex2D<unsigned char>(
//			tex, (float)(u + 4 * ib - RADIUS + 0), (float)(v - RADIUS));
//		LocalBlock[SharedIdx + 4 * ib + 1] = tex2D<unsigned char>(
//			tex, (float)(u + 4 * ib - RADIUS + 1), (float)(v - RADIUS));
//		LocalBlock[SharedIdx + 4 * ib + 2] = tex2D<unsigned char>(
//			tex, (float)(u + 4 * ib - RADIUS + 2), (float)(v - RADIUS));
//		LocalBlock[SharedIdx + 4 * ib + 3] = tex2D<unsigned char>(
//			tex, (float)(u + 4 * ib - RADIUS + 3), (float)(v - RADIUS));
//	}
//
//	if (threadIdx.y < RADIUS * 2) {
//		//
//		// copy trailing RADIUS*2 rows of pixels into shared
//		//
//		SharedIdx = (blockDim.y + threadIdx.y) * SharedPitch;
//
//		for (ib = threadIdx.x; ib < BlockWidth + 2 * RADIUS; ib += blockDim.x) {
//			LocalBlock[SharedIdx + 4 * ib + 0] =
//				tex2D<unsigned char>(tex, (float)(u + 4 * ib - RADIUS + 0),
//					(float)(v + blockDim.y - RADIUS));
//			LocalBlock[SharedIdx + 4 * ib + 1] =
//				tex2D<unsigned char>(tex, (float)(u + 4 * ib - RADIUS + 1),
//					(float)(v + blockDim.y - RADIUS));
//			LocalBlock[SharedIdx + 4 * ib + 2] =
//				tex2D<unsigned char>(tex, (float)(u + 4 * ib - RADIUS + 2),
//					(float)(v + blockDim.y - RADIUS));
//			LocalBlock[SharedIdx + 4 * ib + 3] =
//				tex2D<unsigned char>(tex, (float)(u + 4 * ib - RADIUS + 3),
//					(float)(v + blockDim.y - RADIUS));
//		}
//	}
//
//	cg::sync(cta);
//
//	u >>= 2;  // index as uchar4 from here
//	uchar4* pSobel = (uchar4*)(((char*)pSobelOriginal) + v * SobelPitch);
//	SharedIdx = threadIdx.y * SharedPitch;
//
//	for (ib = threadIdx.x; ib < BlockWidth; ib += blockDim.x) {
//		unsigned char pix00 = LocalBlock[SharedIdx + 4 * ib + 0 * SharedPitch + 0];
//		unsigned char pix01 = LocalBlock[SharedIdx + 4 * ib + 0 * SharedPitch + 1];
//		unsigned char pix02 = LocalBlock[SharedIdx + 4 * ib + 0 * SharedPitch + 2];
//		unsigned char pix10 = LocalBlock[SharedIdx + 4 * ib + 1 * SharedPitch + 0];
//		unsigned char pix11 = LocalBlock[SharedIdx + 4 * ib + 1 * SharedPitch + 1];
//		unsigned char pix12 = LocalBlock[SharedIdx + 4 * ib + 1 * SharedPitch + 2];
//		unsigned char pix20 = LocalBlock[SharedIdx + 4 * ib + 2 * SharedPitch + 0];
//		unsigned char pix21 = LocalBlock[SharedIdx + 4 * ib + 2 * SharedPitch + 1];
//		unsigned char pix22 = LocalBlock[SharedIdx + 4 * ib + 2 * SharedPitch + 2];
//
//		uchar4 out;
//
//		out.x = ComputeSobel(pix00, pix01, pix02, pix10, pix11, pix12, pix20, pix21,
//			pix22, fScale);
//
//		pix00 = LocalBlock[SharedIdx + 4 * ib + 0 * SharedPitch + 3];
//		pix10 = LocalBlock[SharedIdx + 4 * ib + 1 * SharedPitch + 3];
//		pix20 = LocalBlock[SharedIdx + 4 * ib + 2 * SharedPitch + 3];
//		out.y = ComputeSobel(pix01, pix02, pix00, pix11, pix12, pix10, pix21, pix22,
//			pix20, fScale);
//
//		pix01 = LocalBlock[SharedIdx + 4 * ib + 0 * SharedPitch + 4];
//		pix11 = LocalBlock[SharedIdx + 4 * ib + 1 * SharedPitch + 4];
//		pix21 = LocalBlock[SharedIdx + 4 * ib + 2 * SharedPitch + 4];
//		out.z = ComputeSobel(pix02, pix00, pix01, pix12, pix10, pix11, pix22, pix20,
//			pix21, fScale);
//
//		pix02 = LocalBlock[SharedIdx + 4 * ib + 0 * SharedPitch + 5];
//		pix12 = LocalBlock[SharedIdx + 4 * ib + 1 * SharedPitch + 5];
//		pix22 = LocalBlock[SharedIdx + 4 * ib + 2 * SharedPitch + 5];
//		out.w = ComputeSobel(pix00, pix01, pix02, pix10, pix11, pix12, pix20, pix21,
//			pix22, fScale);
//
//		if (u + ib < w / 4 && v < h) {
//			pSobel[u + ib] = out;
//		}
//	}
//
//	cg::sync(cta);
//}

__global__ void sobelTest(cudaSurfaceObject_t* dstSurfMipMapArray, cudaTextureObject_t textureMipMapInput, size_t mipLevels, int width, int height)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	//printf("x: %hu y: %hu\n", x, y);
	for (uint32_t mipLevelIdx = 0; mipLevelIdx < mipLevels; mipLevelIdx++)
	{
		if (y < height && x < width) {
			float px = 1.0 / width;
			float py = 1.0 / height;

			float t = tex2DLod<float>(textureMipMapInput, x * px, y * px, (float)mipLevelIdx);
			surf2Dwrite(t, dstSurfMipMapArray[mipLevelIdx], x * 4, y);
		}
	}
}

// Wrapper for the __global__ call that sets up the texture and threads
extern "C" void sobelFilter(cudaSurfaceObject_t* dstSurfMipMapArray, cudaTextureObject_t textureMipMapInput, cudaStream_t streamToRun, size_t mipLevels, int width, int height) {
//		dim3 threads(16, 4);
//#ifndef FIXED_BLOCKWIDTH
//		int BlockWidth = 80;  // must be divisible by 16 for coalescing
//#endif
//		dim3 blocks = dim3(width / (4 * BlockWidth) + (0 != width % (4 * BlockWidth)),
//			height / threads.y + (0 != height % threads.y));
//		int SharedPitch = ~0x3f & (4 * (BlockWidth + 2 * RADIUS) + 0x3f);
//		int sharedMem = SharedPitch * (threads.y + 2 * RADIUS);
//
//		// for the shared kernel, width must be divisible by 4
//		width &= ~3;
		dim3 threadsperBlock(32, 32);
		//dim3 threadsperBlock(16, 16);
		dim3 numBlocks((width + threadsperBlock.x - 1) / threadsperBlock.x,
			(height + threadsperBlock.y - 1) / threadsperBlock.y);

//		SobelShared << <blocks, threads, sharedMem >> > ((uchar4*)odata, iw,
//#ifndef FIXED_BLOCKWIDTH
//			BlockWidth, SharedPitch,
//#endif
//			iw, ih, fScale, texObject);
		//sobelTest << <blocks, threads, 0, streamToRun >> > (dstSurfMipMapArray, textureMipMapInput, mipLevels, width, height);
		sobelTest << <numBlocks, threadsperBlock, 0, streamToRun >> > (dstSurfMipMapArray, textureMipMapInput, mipLevels, width, height);
}
