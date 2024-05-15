#pragma once

#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "types.h"
#include <map>

#define BLOCK_SIZE 8
void cuda_preprocess_init(int max_image_size);

void cuda_preprocess_destroy();

void cuda_batch_preprocess(std::vector<cv::Mat> &img_batch, float *dst, int dst_width, int dst_height, cudaStream_t stream);

void resize_cuda(const int& batchSize, unsigned char* src, int srcWidth, int srcHeight,
	float* dst, int dstWidth, int dstHeight,
	float paddingValue, AffineMat matrix);

