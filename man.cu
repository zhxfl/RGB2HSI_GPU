#include <iostream>
#include <iomanip>
#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/gpu/gpu.hpp"
#include <stdio.h>
#include <time.h>
#include <cmath>
#include <stdio.h>
#include <math.h>

using namespace std;
using namespace cv;
using namespace cv::gpu;

#define msg(format, ...) do { fprintf(stderr, format, ##__VA_ARGS__); } while (0)
#define err(format, ...) do { fprintf(stderr, format, ##__VA_ARGS__); exit(1); } while (0)

#define malloc2D(name, xDim, yDim, type) do {               \
    name = (type **)malloc(xDim * sizeof(type *));          \
    assert(name != NULL);                                   \
    name[0] = (type *)malloc(xDim * yDim * sizeof(type));   \
    assert(name[0] != NULL);                                \
    for (size_t i = 1; i < xDim; i++)                       \
        name[i] = name[i-1] + yDim;                         \
} while (0)

#ifdef __CUDACC__
inline void checkCuda(cudaError_t e) {
	if (e != cudaSuccess) {
		// cudaGetErrorString() isn't always very helpful. Look up the error
		// number in the cudaError enum in driver_types.h in the CUDA includes
		// directory for a better explanation.
		err("CUDA Error %d: %s\n", e, cudaGetErrorString(e));
	}
}

inline void checkLastCudaError() {
	checkCuda(cudaGetLastError());
}
#endif

const float pii = asin(1.0) * 2;

int minn(int a, int b, int c) {
	return min(a, min(b, c));
}

int Rgb2Hsi(const IplImage* img, CvMat* dataH, CvMat* dataS, CvMat* dataI) {
	uchar* data;

	// rgb·ÖÁ¿
	int img_r, img_g, img_b;
	int min_rgb;  // rgb·ÖÁ¿ÖÐµÄ×îÐ¡Öµ
	// HSI·ÖÁ¿
	float fHue, fSaturation, fIntensity;

	for (int i = 0; i < img->height; i++) {
		for (int j = 0; j < img->width; j++) {
			//data = cvPtr2D(img, i, j, 0);
			data = (uchar *) img->imageData + i * img->widthStep + j * 3;
			img_b = (uchar) *data;
			data++;
			img_g = (uchar) *data;
			data++;
			img_r = (uchar) *data;

			// Intensity·ÖÁ¿[0, 1]
			fIntensity = (float) ((img_b + img_g + img_r) / 3) / 255;

			// µÃµœRGB·ÖÁ¿ÖÐµÄ×îÐ¡Öµ
			float fTemp = img_r < img_g ? img_r : img_g;
			min_rgb = fTemp < img_b ? fTemp : img_b;
			// Saturation·ÖÁ¿[0, 1]
			fSaturation = 1.0f
					- (float) (3 * min_rgb) / (img_r + img_g + img_b);

			// ŒÆËãthetaœÇ
			float numerator = (img_r - img_g + img_r - img_b) / 2;
			float denominator = sqrt(
					pow((float) (img_r - img_g), 2)
							+ (img_r - img_b) * (img_g - img_b));

			// ŒÆËãHue·ÖÁ¿
			if (denominator != 0) {
				float theta = acos(numerator / denominator) * 180 / 3.14;

				if (img_b <= img_g) {
					fHue = theta;
				} else {
					fHue = 360 - theta;
				}
			} else {
				fHue = 0;
			}

			// ž³Öµ
			//printf("%f %f %f\n",fHue,fSaturation, fIntensity);
			cvmSet(dataH, i, j, fHue);
			cvmSet(dataS, i, j, fSaturation);
			cvmSet(dataI, i, j, fIntensity);
		}
	}

	return 1;

}

int Hsi2Rgb(IplImage* src, CvMat* dataH, CvMat* dataS, CvMat* dataI) {
	uchar iB, iG, iR;
	for (int i = 0; i < src->height; i++) {
		for (int j = 0; j < src->width; j++) {
			// žÃµãµÄÉ«¶ÈH
			double dH = cvmGet(dataH, i, j);
			// žÃµãµÄÉ«±¥ºÍ¶ÈS
			double dS = cvmGet(dataS, i, j);
			// žÃµãµÄÁÁ¶È
			double dI = cvmGet(dataI, i, j);

			double dTempB, dTempG, dTempR;
			// RGÉÈÇø
			if (dH < 120 && dH >= 0) {
				// œ«H×ªÎª»¡¶È±íÊŸ
				dH = dH * pii / 180;
				dTempB = dI * (1 - dS);
				dTempR = dI * (1 + (dS * cos(dH)) / cos(pii / 3 - dH));
				dTempG = (3 * dI - (dTempR + dTempB));
			}
			// GBÉÈÇø
			else if (dH < 240 && dH >= 120) {
				dH -= 120;

				// œ«H×ªÎª»¡¶È±íÊŸ
				dH = dH * pii / 180;

				dTempR = dI * (1 - dS);
				dTempG = dI * (1 + dS * cos(dH) / cos(pii / 3 - dH));
				dTempB = (3 * dI - (dTempR + dTempG));
			}
			// BRÉÈÇø
			else {
				dH -= 240;

				// œ«H×ªÎª»¡¶È±íÊŸ
				dH = dH * pii / 180;

				dTempG = dI * (1 - dS);
				dTempB = dI * (1 + (dS * cos(dH)) / cos(pii / 3 - dH));
				dTempR = (3 * dI - (dTempG + dTempB));
			}

			//printf("%f %f %f\n", dTempB, dTempG, dTempR);
			if (dTempR > 1.0)
				dTempR = 1.0;
			if (dTempR < 0)
				dTempR = 0.0;
			if (dTempG > 1.0)
				dTempG = 1.0;
			if (dTempG < 0)
				dTempG = 0.0;
			if (dTempB > 1.0)
				dTempB = 1.0;
			if (dTempB < 0)
				dTempB = 0.0;
			iB = (uchar) (dTempB * 255);
			iG = (uchar) (dTempG * 255);
			iR = (uchar) (dTempR * 255);

			cvSet2D(src, i, j, cvScalar(iB, iG, iR));
			//offset = src->widthStep * i + j * src->nChannels;
			//src->imageData[offset] = iB;
			//src->imageData[offset+1] = iG;
			//src->imageData[offset+2] = iR;
		}
	}

	return 1;
}

int EqualizeHist(CvMat *pImg) {
	int histogram[256];

	memset(histogram, 0, sizeof(histogram));
	for (int y = 0; y < pImg->rows; y++) {
		for (int x = 0; x < pImg->cols; x++) {
			int t = 255.0 * cvmGet(pImg, y, x) + 0.5;

			if (t > 255)
				t = 255;
			if (t < 0)
				t = 0;
			histogram[(int) t]++;
		}
	}
	int Min = 1000000000;
	for (int i = 1; i < 255; i++) {
		histogram[i] += histogram[i - 1];
		if (histogram[i] < Min)
			Min = histogram[i];
	}

	int num = pImg->height * pImg->width - Min;
	for (int y = 0; y < pImg->rows; y++) {
		for (int x = 0; x < pImg->cols; x++) {
			int t = 255.0 * (cvmGet(pImg, y, x)) + 0.5;
			if (t > 255)
				t = 255;
			if (t < 0)
				t = 0;
			t = histogram[t];
			//printf("%f ",(float)(t - 1) / num);
			cvmSet(pImg, y, x, (float) (1.0 * t - Min) / num);
		}			//printf("\n");
	}
	return true;
}

__global__ void g_rgb2hsi(int *g_rgb, float *g_h, float *g_s, float *g_i,
		int len) {

	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid >= len) {
		return;
	}
	int img_r, img_g, img_b;
	int min_rgb;
	float fHue, fSaturation, fIntensity;

	int data = g_rgb[tid];
	int t = (1 << 8) - 1;
	img_r = data & t;
	data = data >> 8;
	img_g = data & t;
	data = data >> 8;
	img_b = data & t;

	fIntensity = (float) ((img_b + img_g + img_r) / 3) / 255;

	float fTemp = img_r < img_g ? img_r : img_g;
	min_rgb = fTemp < img_b ? fTemp : img_b;
	fSaturation = 1.0f - (float) (3 * min_rgb) / (img_r + img_g + img_b);

	float numerator = (img_r - img_g + img_r - img_b) / 2;
	float denominator = sqrt(
			pow((float) (img_r - img_g), 2)
					+ (img_r - img_b) * (img_g - img_b));

	if (denominator != 0) {
		float theta = acos(numerator / denominator) * 180 / 3.14;
		if (img_b <= img_g) {
			fHue = theta;
		} else {
			fHue = 360 - theta;
		}
	} else {
		fHue = 0;
	}
	g_h[tid] = fHue;
	g_s[tid] = fSaturation;
	g_i[tid] = fIntensity;
}

__global__ void g_hsi2rgb(int *g_rgb, float *g_h, float *g_s, float *g_i,
		int len, double pi) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid >= len) {
		return;
	}
	uchar iB, iG, iR;
	float dH = (float) g_h[tid];
	float dS = (float) g_s[tid];
	float dI = (float) g_i[tid];

	float dTempB, dTempG, dTempR;

	if (dH < 120 && dH >= 0) {
		// œ«H×ªÎª»¡¶È±íÊŸ
		dH = dH * pi / 180;
		dTempB = dI * (1 - dS);
		dTempR = dI * (1 + (dS * cos(dH)) / cos(pi / 3 - dH));
		dTempG = (3 * dI - (dTempR + dTempB));
	}
	// GBÉÈÇø
	else if (dH < 240 && dH >= 120) {
		dH -= 120;

		// œ«H×ªÎª»¡¶È±íÊŸ
		dH = dH * pi / 180;

		dTempR = dI * (1 - dS);
		dTempG = dI * (1 + dS * cos(dH) / cos(pi / 3 - dH));
		dTempB = (3 * dI - (dTempR + dTempG));
	}
	// BRÉÈÇø
	else {
		dH -= 240;

		// œ«H×ªÎª»¡¶È±íÊŸ
		dH = dH * pi / 180;

		dTempG = dI * (1 - dS);
		dTempB = dI * (1 + (dS * cos(dH)) / cos(pi / 3 - dH));
		dTempR = (3 * dI - (dTempG + dTempB));
	}

	//printf("%f %f %f\n", dTempB, dTempG, dTempR);
	if (dTempR > 1.0)
		dTempR = 1.0;
	if (dTempR < 0)
		dTempR = 0.0;
	if (dTempG > 1.0)
		dTempG = 1.0;
	if (dTempG < 0)
		dTempG = 0.0;
	if (dTempB > 1.0)
		dTempB = 1.0;
	if (dTempB < 0)
		dTempB = 0.0;
	iB = (uchar) (dTempB * 255);
	iG = (uchar) (dTempG * 255);
	iR = (uchar) (dTempR * 255);

	int t = 0;
	t |= (iR);
	t |= (iG << 8);
	t |= (iB << 16);
	g_rgb[tid] = t;
}

bool GPU_Rgb2Hsi(const IplImage* img, CvMat* dataH, CvMat* dataS,
		CvMat* dataI) {
	int * d_rgb;
	float *d_h;
	float *d_s;
	float *d_i;

	int * h_rgb;
	float *h_h;
	float *h_s;
	float *h_i;

	int len = img->width * img->height;

	cudaMalloc((void **) &d_rgb, len * sizeof(int));
	cudaMalloc((void **) &d_h, len * sizeof(float));
	cudaMalloc((void **) &d_s, len * sizeof(float));
	cudaMalloc((void **) &d_i, len * sizeof(float));

	h_rgb = (int *) malloc(len * sizeof(int));
	h_h = (float *) malloc(len * sizeof(float));
	h_s = (float *) malloc(len * sizeof(float));
	h_i = (float *) malloc(len * sizeof(float));

	cudaDeviceProp deviceProp;
	int deviceNum;
	cudaGetDevice(&deviceNum);
	cudaGetDeviceProperties(&deviceProp, deviceNum);
	int maxThread = deviceProp.maxThreadsPerBlock;
	dim3 thread = dim3(maxThread <= len ? maxThread : len);
	dim3 block = dim3((len + maxThread - 1) / maxThread);

	// restore the rgb data, r(byte),g(byte),b(byte)store in  32-bit word memory
	for (int i = 0; i < img->height; i++) {
		for (int j = 0; j < img->width; j++) {
			uchar *data = (uchar *) img->imageData + i * img->widthStep + j * 3;

			uchar img_b = (uchar) *data;
			data++;
			uchar img_g = (uchar) *data;
			data++;
			uchar img_r = (uchar) *data;

			int t = 0;
			t |= (img_r);
			t |= (img_g << 8);
			t |= (img_b << 16);
			h_rgb[i * img->width + j] = t;
		}
	}

	cudaMemcpy(d_rgb, h_rgb, sizeof(int) * len, cudaMemcpyHostToDevice);

	g_rgb2hsi<<<block, thread>>>(d_rgb, d_h, d_s, d_i, len);
	cudaDeviceSynchronize();
	checkLastCudaError();

	cudaMemcpy(h_h, d_h, sizeof(float) * len, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_s, d_s, sizeof(float) * len, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_i, d_i, sizeof(float) * len, cudaMemcpyDeviceToHost);

	for (int i = 0; i < img->height; i++) {
		for (int j = 0; j < img->width; j++) {
			int t = i * img->width + j;
			cvmSet(dataH, i, j, h_h[t]);
			cvmSet(dataS, i, j, h_s[t]);
			cvmSet(dataI, i, j, h_i[t]);
		}
	}

	free(h_rgb);
	free(h_h);
	free(h_s);
	free(h_i);
	cudaFree(d_rgb);
	cudaFree(d_h);
	cudaFree(d_s);
	cudaFree(d_i);

	return true;
}

bool GPU_Hsi2Rgb(IplImage* img, CvMat* dataH, CvMat* dataS, CvMat* dataI) {
	int * d_rgb;
	float *d_h;
	float *d_s;
	float *d_i;

	int * h_rgb;
	float *h_h;
	float *h_s;
	float *h_i;

	int len = img->width * img->height;

	cudaMalloc((void **) &d_rgb, len * sizeof(int));
	cudaMalloc((void **) &d_h, len * sizeof(float));
	cudaMalloc((void **) &d_s, len * sizeof(float));
	cudaMalloc((void **) &d_i, len * sizeof(float));

	h_rgb = (int *) malloc(len * sizeof(int));
	h_h = (float *) malloc(len * sizeof(float));
	h_s = (float *) malloc(len * sizeof(float));
	h_i = (float *) malloc(len * sizeof(float));

	cudaDeviceProp deviceProp;
	int deviceNum;
	cudaGetDevice(&deviceNum);
	cudaGetDeviceProperties(&deviceProp, deviceNum);
	int maxThread = deviceProp.maxThreadsPerBlock;
	dim3 thread = dim3(maxThread <= len ? maxThread : len);
	dim3 block = dim3((len + maxThread - 1) / maxThread);
	for (int i = 0; i < img->height; i++) {
		for (int j = 0; j < img->width; j++) {
			float dH = cvmGet(dataH, i, j);
			float dS = cvmGet(dataS, i, j);
			float dI = cvmGet(dataI, i, j);

			int t = i * img->width + j;
			h_h[t] = dH;
			h_s[t] = dS;
			h_i[t] = dI;
		}
	}

	cudaMemcpy(d_h, h_h, sizeof(float) * len, cudaMemcpyHostToDevice);
	cudaMemcpy(d_s, h_s, sizeof(float) * len, cudaMemcpyHostToDevice);
	cudaMemcpy(d_i, h_i, sizeof(float) * len, cudaMemcpyHostToDevice);

	g_hsi2rgb<<<block, thread>>>(d_rgb, d_h, d_s, d_i, len, pii);
	cudaDeviceSynchronize();

	cudaMemcpy(h_rgb, d_rgb, sizeof(int) * len, cudaMemcpyDeviceToHost);

	for (int i = 0; i < img->height; i++) {
		for (int j = 0; j < img->width; j++) {
			int id = i * img->width + j;
			int data = h_rgb[id];
			int t = (1 << 8) - 1;
			uchar img_r = data & t;
			data = data >> 8;
			uchar img_g = data & t;
			data = data >> 8;
			uchar img_b = data & t;
			cvSet2D(img, i, j, cvScalar(img_b, img_g, img_r));
		}
	}

	free(h_rgb);
	free(h_h);
	free(h_s);
	free(h_i);
	cudaFree(d_rgb);
	cudaFree(d_h);
	cudaFree(d_s);
	cudaFree(d_i);

	//printf("%d %d\n",sizeof(uchar), sizeof(char));
	return true;
}

void GPU(char * path) {
	IplImage * Img = cvLoadImage(path, 1);
	CvMat * HImg = cvCreateMat(Img->height, Img->width, CV_32FC1);
	CvMat * SImg = cvCreateMat(Img->height, Img->width, CV_32FC1);
	CvMat * IImg = cvCreateMat(Img->height, Img->width, CV_32FC1);

	float start = clock();

	if (GPU_Rgb2Hsi(Img, HImg, SImg, IImg) == 0)

	{
		printf("Convert Error!\n");
		exit(-1);
	}

	EqualizeHist(IImg);

	if (GPU_Hsi2Rgb(Img, HImg, SImg, IImg) == 0) {
		printf("Convert Error!\n");
		exit(-1);
	}
	float end = clock();
	printf("time = %fms\n", 1000.0f * (end - start) / CLOCKS_PER_SEC);
	cvNamedWindow("GPU_1", 1);
	cvShowImage("GPU_1", Img);
	cvWaitKey(0);
}

__global__ void g_map(float *d_i, int *d_histogram, int len, float num, float Min) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid >= len) {
		return;
	}
	__shared__ int h[256];
	if(threadIdx.x < 256)
	{
		h[threadIdx.x] = d_histogram[threadIdx.x];
	}
	__syncthreads();
	int t = 255.0 * (d_i[tid]) + 0.5;
	if (t > 255)
		t = 255;
	else if (t < 0)
		t = 0;
	t = h[t];
	//printf("%d %d\n", t, h_histogram[t]);
	d_i[tid] = (float) (1.0 * t - Min) / num;

}

void new_GPU(char * path) {
	IplImage * Img = cvLoadImage(path, 1);
	CvMat * HImg = cvCreateMat(Img->height, Img->width, CV_32FC1);
	CvMat * SImg = cvCreateMat(Img->height, Img->width, CV_32FC1);
	CvMat * IImg = cvCreateMat(Img->height, Img->width, CV_32FC1);

	float start = clock();

	int *d_rgb;
	float *d_h;
	float *d_s;
	float *d_i;
	int *d_histogram;

	int *h_rgb;
	float *h_h;
	float *h_s;
	float *h_i;
	int *h_histogram;

	int len = Img->width * Img->height;

	cudaMalloc((void **) &d_rgb, len * sizeof(int));
	cudaMalloc((void **) &d_h, len * sizeof(float));
	cudaMalloc((void **) &d_s, len * sizeof(float));
	cudaMalloc((void **) &d_i, len * sizeof(float));
	cudaMalloc((void **) &d_histogram, 256 * sizeof(float));
	h_rgb = (int *) malloc(len * sizeof(int));
	h_h = (float *) malloc(len * sizeof(float));
	h_s = (float *) malloc(len * sizeof(float));
	h_i = (float *) malloc(len * sizeof(float));
	h_histogram = (int *) malloc(256 * sizeof(int));

	cudaDeviceProp deviceProp;
	int deviceNum;
	cudaGetDevice(&deviceNum);
	cudaGetDeviceProperties(&deviceProp, deviceNum);
	int maxThread = deviceProp.maxThreadsPerBlock;
	dim3 thread = dim3(maxThread <= len ? maxThread : len);
	dim3 block = dim3((len + maxThread - 1) / maxThread);
	// restore the rgb data, r(byte),g(byte),b(byte)store in  32-bit word memory
	for (int i = 0; i < Img->height; i++) {
		for (int j = 0; j < Img->width; j++) {
			uchar *data = (uchar *) Img->imageData + i * Img->widthStep + j * 3;

			uchar img_b = (uchar) *data;
			data++;
			uchar img_g = (uchar) *data;
			data++;
			uchar img_r = (uchar) *data;

			int t = 0;
			t |= (img_r);
			t |= (img_g << 8);
			t |= (img_b << 16);
			h_rgb[i * Img->width + j] = t;
		}
	}

	//TODO rbg2hsi
	cudaMemcpy(d_rgb, h_rgb, sizeof(int) * len, cudaMemcpyHostToDevice);
	g_rgb2hsi<<<block, thread>>>(d_rgb, d_h, d_s, d_i, len);
	cudaDeviceSynchronize();
	checkLastCudaError();

	//TODO EqualizeHist
	cudaMemcpy(h_i, d_i, sizeof(float) * len, cudaMemcpyDeviceToHost);

	memset(h_histogram, 0, sizeof(int) * 256);

	for (int y = 0; y < Img->height; y++) {
		for (int x = 0; x < Img->width; x++) {
			int id = y * Img->width + x;
			int t = 255.0 * h_i[id] + 0.5;

			if (t > 255)
				t = 255;
			else if (t < 0)
				t = 0;
			h_histogram[(int) t]++;
		}
	}

	int Min = 1000000000;
	for (int i = 1; i < 255; i++) {
		h_histogram[i] += h_histogram[i - 1];
		if (h_histogram[i] < Min)
			Min = h_histogram[i];
		//printf("%d\n",h_histogram[i]);
	}

	//udaMemcpy(d_i, h_i, sizeof(float) * len, cudaMemcpyHostToDevice);
	cudaMemcpy(d_histogram, h_histogram, sizeof(int) * 256, cudaMemcpyHostToDevice);
	int num = Img->height * Img->width - Min;
	g_map<<<block,thread>>>(d_i, d_histogram,num, len,Min);

	//TODO hsi2rgb
	g_hsi2rgb<<<block, thread>>>(d_rgb, d_h, d_s, d_i, len, pii);
	cudaDeviceSynchronize();
	checkLastCudaError();
	cudaMemcpy(h_rgb, d_rgb, sizeof(int) * len, cudaMemcpyDeviceToHost);

	for (int i = 0; i < Img->height; i++) {
		for (int j = 0; j < Img->width; j++) {
			int id = i * Img->width + j;
			int data = h_rgb[id];
			int t = (1 << 8) - 1;
			uchar img_r = data & t;
			data = data >> 8;
			uchar img_g = data & t;
			data = data >> 8;
			uchar img_b = data & t;
			cvSet2D(Img, i, j, cvScalar(img_b, img_g, img_r));
		}
	}
	float end = clock();
	free(h_rgb);
	free(h_h);
	free(h_s);
	free(h_i);
	free(h_histogram);
	cudaFree(d_histogram);
	cudaFree(d_rgb);
	cudaFree(d_h);
	cudaFree(d_s);
	cudaFree(d_i);
	cudaFree(d_histogram);
	printf("time = %fms\n", 1000.0f * (end - start) / CLOCKS_PER_SEC);
	cvNamedWindow("GPU_2", 1);
	cvShowImage("GPU_2", Img);
	cvWaitKey(0);
}

void CPU(char * path) {
	IplImage * Img = cvLoadImage(path, 1);
	CvMat * HImg = cvCreateMat(Img->height, Img->width, CV_32FC1);
	CvMat * SImg = cvCreateMat(Img->height, Img->width, CV_32FC1);
	CvMat * IImg = cvCreateMat(Img->height, Img->width, CV_32FC1);

	float start = clock();

	if (Rgb2Hsi(Img, HImg, SImg, IImg) == 0) {
		printf("Convert Error!\n");
		exit(-1);
	}

	EqualizeHist(IImg);

	if (Hsi2Rgb(Img, HImg, SImg, IImg) == 0)

	{
		printf("Convert Error!\n");
		exit(-1);
	}
	float end = clock();
	printf("time = %fms\n", 1000.0f * (end - start) / CLOCKS_PER_SEC);
	cvNamedWindow("CPU", 1);
	cvShowImage("CPU", Img);
	cvWaitKey(0);
}

int main(int argc, char ** argv) {
	CPU("1.jpg");
	GPU("1.jpg");
	new_GPU("1.jpg");
	return 0;
}
