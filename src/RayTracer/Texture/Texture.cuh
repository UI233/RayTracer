#pragma once
#ifndef TEXTURE_H
#define TEXTURE_H

#include <vector_types.h>
#include <driver_functions.h>
#include <cuda_runtime.h>
#include <iostream>
#include <string>
//#include "helper_cuda.h"
#include "helper_math.h"
//#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_IMPLEMENTATION
#include "tool/stb_image.h"
//#define STB_IMAGE_WRITE_IMPLEMENTATION
//#include "tool/stb_image_write.h"

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

#endif // !