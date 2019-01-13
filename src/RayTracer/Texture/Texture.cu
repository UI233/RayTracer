#include "Texture.cuh"
uint getMipMapLevels(cudaExtent size) {
	size_t sz = MAX(MAX(size.width, size.height), size.depth);

	uint levels = 0;

	while (sz) {
		sz /= 2;
		levels++;
	}

	return levels;
}
__global__ void
d_mipmap(cudaSurfaceObject_t mipOutput, cudaTextureObject_t mipInput, float4* test, uint imageW, uint imageH, uint threadWidth, uint threadHeight) {
	uint x = (blockIdx.x * blockDim.x + threadIdx.x)*threadWidth;
	uint y = (blockIdx.y * blockDim.y + threadIdx.y)*threadHeight;

	float px = 1.0 / float(imageW);
	float py = 1.0 / float(imageH);

	for (int i = 0; i < threadHeight; i++) {
		for (int j = 0; j < threadWidth; j++) {
			if ((x + j < imageW) && (y + i < imageH)) {
				// take the average of 4 samples

				// we are using the normalized access to make sure non-power-of-two textures
				// behave well when downsized.
				float4 color =
					(tex2D<float4>(mipInput, (x + j + 0) * px, (y + i + 0) * py)) +
					(tex2D<float4>(mipInput, (x + j + 1) * px, (y + i + 0) * py)) +
					(tex2D<float4>(mipInput, (x + j + 1) * px, (y + i + 1) * py)) +
					(tex2D<float4>(mipInput, (x + j + 0) * px, (y + i + 1) * py));


				color /= 4.0;
				color = fminf(color, make_float4(1.0));
				//printf("%f %f %f %f\n", color.x, color.y, color.z, color.w);
				surf2Dwrite(color, mipOutput, (x + j) * sizeof(float4), y + i);
				int idx = (y + i) * imageW + j + x;
				test[idx] = color;
			}
		}
	}
}


void generateMipMaps(cudaMipmappedArray_t mipmapArray, cudaExtent size) {
	size_t width = size.width;
	size_t height = size.height;

	uint level = 0;

	while (width != 1 || height != 1) {
		width /= 2;
		width = MAX((size_t)1, width);
		height /= 2;
		height = MAX((size_t)1, height);
		std::cout << width << " " << height << std::endl;
		cudaArray_t levelFrom;
		auto error = cudaGetMipmappedArrayLevel(&levelFrom, mipmapArray, level);
		cudaArray_t levelTo;
		error = cudaGetMipmappedArrayLevel(&levelTo, mipmapArray, level + 1);

		cudaExtent  levelToSize;
		error = cudaArrayGetInfo(NULL, &levelToSize, NULL, levelTo);
		levelToSize.width == width;
		levelToSize.height == height;
		levelToSize.depth == 0;

		// generate texture object for reading
		cudaTextureObject_t         texInput;
		cudaResourceDesc            texRes;
		memset(&texRes, 0, sizeof(cudaResourceDesc));

		texRes.resType = cudaResourceTypeArray;
		texRes.res.array.array = levelFrom;

		cudaTextureDesc             texDescr;
		memset(&texDescr, 0, sizeof(cudaTextureDesc));

		texDescr.normalizedCoords = 1;
		texDescr.filterMode = cudaFilterModeLinear;

		texDescr.addressMode[0] = cudaAddressModeClamp;
		texDescr.addressMode[1] = cudaAddressModeClamp;
		texDescr.addressMode[2] = cudaAddressModeClamp;

		texDescr.readMode = cudaReadModeElementType;;
		float4 * test;
		error = cudaMalloc(&test, width*height * sizeof(float4));
		error = cudaCreateTextureObject(&texInput, &texRes, &texDescr, NULL);

		// generate surface object for writing

		cudaSurfaceObject_t surfOutput;
		cudaResourceDesc    surfRes;
		memset(&surfRes, 0, sizeof(cudaResourceDesc));
		surfRes.resType = cudaResourceTypeArray;
		surfRes.res.array.array = levelTo;

		error = cudaCreateSurfaceObject(&surfOutput, &surfRes);

		// run mipmap kernel
		dim3 blockSize(16, 16, 1);
		dim3 gridSize(((uint)width + blockSize.x - 1) / blockSize.x, ((uint)height + blockSize.y - 1) / blockSize.y, 1);

		d_mipmap << <gridSize, blockSize >> > (surfOutput, texInput, test, (uint)width, (uint)height, 1, 1);

		error = cudaDeviceSynchronize();
		float4 *output = (float4 *)malloc(width*height * sizeof(float4));
		error = cudaMemcpy(output, test, width*height * sizeof(float4), cudaMemcpyDeviceToHost);
			std::string a=".ppm";
			a = std::to_string(width)+a;

			std::ofstream fp(a, std::ios::out);
			fp << "P3" << std::endl;
			fp << width << " " << height << std::endl;
			fp << "255" << std::endl;
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					fp << int(output[i*width + j].x*255) << " " << int(output[i*width + j].y*255) << " " << int(output[i*width + j].z*255) << std::endl;
				}
			}
		error = cudaGetLastError();
		fp.close();
		cudaDestroySurfaceObject(surfOutput);

		cudaDestroyTextureObject(texInput);
		free(output);
		level++;
	}
}

void initImages(Image *images, float *data, int width, int height) {
	// create individual textures
	Image &image = *images;
	float highestLod = 0;
	image.h_data = data;
	std::cout << width << height << std::endl;
	image.size = make_cudaExtent(width, height, 0);
	image.size.depth = 0;
	image.type = cudaResourceTypeMipmappedArray;
	// how many mipmaps we need
	uint levels = getMipMapLevels(image.size);
	highestLod = MAX(highestLod, (float)levels - 1);
	cudaChannelFormatDesc desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
	cudaMallocMipmappedArray(&image.mipmapArray, &desc, image.size, levels);
	// upload level 0
	cudaArray_t level0;
	cudaGetMipmappedArrayLevel(&level0, image.mipmapArray, 0);

	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr = make_cudaPitchedPtr(image.h_data, image.size.width * sizeof(float4), image.size.width, image.size.height);
	copyParams.dstArray = level0;
	copyParams.extent = image.size;
	copyParams.extent.depth = 1;
	copyParams.kind = cudaMemcpyHostToDevice;
	cudaMemcpy3D(&copyParams);
	// compute rest of mipmaps based on level 0
	generateMipMaps(image.mipmapArray, image.size);

	// generate bindless texture object

	cudaResourceDesc            resDescr;
	memset(&resDescr, 0, sizeof(cudaResourceDesc));

	resDescr.resType = cudaResourceTypeMipmappedArray;
	resDescr.res.mipmap.mipmap = image.mipmapArray;

	cudaTextureDesc             texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));

	texDescr.normalizedCoords = 1;
	texDescr.filterMode = cudaFilterModeLinear;
	texDescr.mipmapFilterMode = cudaFilterModeLinear;
	texDescr.addressMode[0] = cudaAddressModeClamp;
	texDescr.addressMode[1] = cudaAddressModeClamp;
	texDescr.addressMode[2] = cudaAddressModeClamp;

	texDescr.maxMipmapLevelClamp = float(levels - 1);
	texDescr.readMode = cudaReadModeElementType;

	auto error = cudaCreateTextureObject(&image.textureObject, &resDescr, &texDescr, NULL);

}
