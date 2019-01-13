#pragma once
#ifndef TEXTURE_H
#define TEXTURE_H

#include <vector_types.h>
#include <driver_functions.h>
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <fstream>
#include <stdlib.h>
//#include "helper_cuda.h"
#include "helper_math.h"
struct Image {
	void                   *h_data;
	cudaExtent              size;
	cudaResourceType        type;
	cudaArray_t             dataArray;
	cudaMipmappedArray_t    mipmapArray;
	cudaTextureObject_t     textureObject;

	Image() {
		memset(this, 0, sizeof(Image));
	}
};
#ifndef MAX
#define MAX(a,b) ((a > b) ? a : b)
#endif

uint getMipMapLevels(cudaExtent size);
__global__ void
d_mipmap(cudaSurfaceObject_t mipOutput, cudaTextureObject_t mipInput, float4* test, uint imageW, uint imageH, uint threadWidth, uint threadHeight);

void generateMipMaps(cudaMipmappedArray_t mipmapArray, cudaExtent size);

void initImages(Image *images, float *data, int width, int height);

#endif // !