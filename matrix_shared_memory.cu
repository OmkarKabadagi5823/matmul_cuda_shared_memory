
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

#define BLOCK_SIZE 32

typedef struct {
	int width;
	int height;
	int stride;
	float* elements;
} Matrix;

void MatMul(const Matrix, const Matrix, Matrix*);

__device__ float GetElement(Matrix* Mat, int row, int col)
{
	return Mat->elements[row*Mat->stride + col];
}

__device__ void SetElement(Matrix* Mat, int row, int col, float value)
{
	Mat->elements[row*Mat->stride + col] = value;
}

__device__ Matrix GetSubMatrix(Matrix* Mat, int row, int col)
{
	Matrix MatSub;
	MatSub.width = BLOCK_SIZE;
	MatSub.height = BLOCK_SIZE;
	MatSub.stride = Mat->stride;
	MatSub.elements = Mat->elements + row*BLOCK_SIZE*Mat->stride + BLOCK_SIZE*col;

	return MatSub;
}

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;
	Matrix Csub = GetSubMatrix(&C, blockRow, blockCol);

	int row = threadIdx.y;
	int col = threadIdx.x;
	int value = 0;

	for(int m=0; m<(A.width/BLOCK_SIZE); ++m)
	{
		Matrix Asub = GetSubMatrix(&A, blockRow, m);
		Matrix Bsub = GetSubMatrix(&B, m, blockCol);

		__shared__ float Asub_shared[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float Bsub_shared[BLOCK_SIZE][BLOCK_SIZE];

		Asub_shared[row][col] = GetElement(&Asub, row, col);
		Bsub_shared[row][col] = GetElement(&Bsub, row, col);

		__syncthreads();

		for(int k=0; k<BLOCK_SIZE; ++k)
			value += Asub_shared[row][k]*Bsub_shared[k][col];

		__syncthreads();
	}

	SetElement(&Csub, row, col, value);
}

void inputMat(Matrix* Mat)
{
	printf("width: ");
	scanf("%d", &(Mat->width));
	printf("height: ");
	scanf("%d", &(Mat->height));

	Mat->stride = Mat->width;

	Mat->elements = (float*)malloc((Mat->height*Mat->width)*sizeof(float));

	for(int row=0; row<Mat->height; ++row)
		for(int col=0; col<Mat->width; ++col)
			scanf("%f", Mat->elements + row*Mat->width + col);
}

void randomMat(Matrix* Mat, int width, int height, int lower, int upper)
{
	Mat->width = width;
	Mat->height = height;
	Mat->stride = width;

	Mat->elements = (float*)malloc((Mat->height*Mat->width)*sizeof(float));

	srand(time(NULL));
	for(int row=0; row<Mat->height; ++row)
		for(int col=0; col<Mat->width; ++col)
			*(Mat->elements + row*Mat->width + col) = lower + (rand()%(upper-lower));
}

void printMat(Matrix* Mat)
{
	for(int row=0; row<Mat->height; ++row)
	{
		for(int col=0; col<Mat->width; ++col)
		{
			printf("%f ", Mat->elements[row*Mat->width + col]);
		}
		printf("\n");
	}
}

int main()
{
	Matrix A, B, C;

	//printf("Input Matrix A:\n");
	randomMat(&A, 512, 512, 0, 5);
	printf("Generated a random Matrix A of dimensions %d x %d\n", A.height, A.width);
	//printf("\n");

	//printf("Input Matrix B:\n");
	randomMat(&B, 512, 512, 0, 5);
	printf("Generated a random Matrix B of dimensions %d x %d\n", B.height, B.width);
	//printf("\n");

	if(A.width != B.height)
	{
		printf("[MatMul] Mat1.width should be equal to Mat2.height\n");
		return 0;
	}

	//printf("A:\n");
	//printMat(&A);
	//printf("\n");
	//printf("B:\n");
	//printMat(&B);

	C.width = B.width;
	C.height = A.height;
	C.stride = C.width;
	C.elements = (float*)malloc((C.width*C.height)*sizeof(float));

	clock_t start, end;

	start = clock();
	MatMul(A, B, &C);
	end = clock();

	double time_taken = ((double)(end-start))/CLOCKS_PER_SEC;

	printf("\n");
	//printf("C (Output):\n");
	//printMat(&C);

	printf("\nTime Taken to calculate A x B (output dimension: %d x %d): %f ms", A.height, B.width, 1000*time_taken);

	free(A.elements);
	free(B.elements);
	free(C.elements);

	return 0;
}

void MatMul(const Matrix A, const Matrix B, Matrix* C)
{
	Matrix d_A;
	d_A.width = A.width;
	d_A.height = A.height;
	d_A.stride = A.stride;
	size_t size_A = A.width*A.height*sizeof(float);
	cudaMalloc(&d_A.elements, size_A);
	cudaMemcpy(d_A.elements, A.elements, size_A, cudaMemcpyHostToDevice);

	Matrix d_B;
	d_B.width = B.width;
	d_B.height = B.height;
	d_B.stride = B.stride;
	size_t size_B = B.width*B.height*sizeof(float);
	cudaMalloc(&d_B.elements, size_B);
	cudaMemcpy(d_B.elements, B.elements, size_B, cudaMemcpyHostToDevice);

	Matrix d_C;
	d_C.width = C->width;
	d_C.height = C->height;
	d_C.stride = C->stride;
	size_t size_C = C->width*C->height*sizeof(float);
	cudaMalloc(&d_C.elements, size_C);

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(B.width/dimBlock.x, A.height/dimBlock.y);

	MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
	cudaDeviceSynchronize();

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	    printf("Error: %s\n", cudaGetErrorString(err));

	cudaMemcpy(C->elements, d_C.elements, size_C, cudaMemcpyDeviceToHost);

	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
}
