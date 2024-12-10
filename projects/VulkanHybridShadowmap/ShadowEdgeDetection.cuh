#include <cuda.h>
#include <cuda_runtime_api.h>

#define CUDA_CALL(x) {			\
	const cudaError_t a = (x);	\
	if (a != cudaSuccess)		\
	{							\
		printf("\nCuda Error: %s (err_num=%d) at line:%d\n", cudaGetErrorString(a), a, __LINE__); \
		cudaDeviceReset();		\
		assert(0);				\
	}							\
}

__constant__ float d_viewInverse[4][4];
__constant__ float d_projInverse[4][4];
__constant__ float d_depthBiasMVP[4][4];
__constant__ float d_lightPos[4];
__constant__ float biasMat[4][4] = {	0.5, 0.0, 0.0, 0.0,
										0.0, 0.5, 0.0, 0.0,
										0.0, 0.0, 1.0, 0.0,
										0.5, 0.5, 0.0, 1.0	};

typedef unsigned char Pixel;

extern "C" void sobelFilter(cudaSurfaceObject_t* shadowEdgeTexture, cudaSurfaceObject_t* lightTexture, cudaTextureObject_t shadowMapTexture, cudaTextureObject_t positionTexture, cudaStream_t streamToRun, size_t mipLevels, int width, int height, int shadowMapSize, float* projInverseMat, float* viewInverseMat, float* depthBiasMVPMat, float* lightPos);