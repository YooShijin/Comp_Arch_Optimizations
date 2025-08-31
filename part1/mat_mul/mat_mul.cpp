/*******************************************************************
 * Author: Asmit, Ashutosh, Hirenmay
 * Date: 21/08/2025
 * File: mat_mul.cpp
 * Description: This file contains implementations of matrix multiplication
 *			    algorithms using various optimization techniques.
 *******************************************************************/

// PA 1: Matrix Multiplication

// includes
#include <iostream>
using namespace std;
#include <stdio.h>
#include <stdlib.h>	   // for malloc, free, atoi
#include <time.h>	   // for time()
#include <chrono>	   // for timing
#include <xmmintrin.h> // for SSE
#include <immintrin.h> // for AVX

#include "helper.h" // for helper functions
#include <unistd.h>

// defines
// NOTE: you can change this value as per your requirement
#define TILE_SIZE 80 // size of the tile for blocking

/**
 * @brief 		Performs matrix multiplication of two matrices.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		size 		dimension of the matrices
 */
void naive_mat_mul(double *A, double *B, double *C, int size)
{

	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			for (int k = 0; k < size; k++)
			{
				C[i * size + j] += A[i * size + k] * B[k * size + j];
			}
		}
	}
}

/**
 * @brief 		Task 1A: Performs matrix multiplication of two matrices using loop optimization.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		size 		dimension of the matrices
 */
void loop_opt_mat_mul(double *A, double *B, double *C, int size, int id)
{
	int n = size;

	switch (id)
	{
	case 1: // Loop Unrolling @2, i-j-k
		for (int i = 0; i < n; i++)
		{
			int idX = i * n;
			for (int j = 0; j < n; j++)
			{
				int k = 0;
				for (k; k < n; k += 2)
				{
					C[idX + j] += A[idX + k] * B[k * n + j];
					C[idX + j] += A[idX + k + 1] * B[(k + 1) * n + j];
				}
				for (; k < n; k++)
				{
					C[idX + j] += A[idX + k] * B[k * n + j];
				}
			}
		}
		break;

	case 2: // Loop Unrolling @4, i-j-k
		for (int i = 0; i < n; i++)
		{
			int idX = i * n;
			for (int j = 0; j < n; j++)
			{
				int k = 0;
				for (k; k < n; k += 4)
				{
					C[idX + j] += A[idX + k] * B[k * n + j];
					C[idX + j] += A[idX + k + 1] * B[(k + 1) * n + j];
					C[idX + j] += A[idX + k + 2] * B[(k + 2) * n + j];
					C[idX + j] += A[idX + k + 3] * B[(k + 3) * n + j];
				}
				for (; k < n; k++)
				{
					C[idX + j] += A[idX + k] * B[k * n + j];
				}
			}
		}
		break;

	case 3: // Loop Unrolling @8, i-j-k
		for (int i = 0; i < n; i++)
		{
			int idX = i * n;
			for (int j = 0; j < n; j++)
			{
				int k = 0;
				for (k; k < n; k += 8)
				{
					C[idX + j] += A[idX + k] * B[k * n + j];
					C[idX + j] += A[idX + k + 1] * B[(k + 1) * n + j];
					C[idX + j] += A[idX + k + 2] * B[(k + 2) * n + j];
					C[idX + j] += A[idX + k + 3] * B[(k + 3) * n + j];
					C[idX + j] += A[idX + k + 4] * B[(k + 4) * n + j];
					C[idX + j] += A[idX + k + 5] * B[(k + 5) * n + j];
					C[idX + j] += A[idX + k + 6] * B[(k + 6) * n + j];
					C[idX + j] += A[idX + k + 7] * B[(k + 7) * n + j];
				}
				for (; k < n; k++)
				{
					C[idX + j] += A[idX + k] * B[k * n + j];
				}
			}
		}
		break;

	case 4: // Loop Unrolling @16, i-j-k
		for (int i = 0; i < n; i++)
		{
			int idX = i * n;
			for (int j = 0; j < n; j++)
			{
				int k = 0;
				for (k; k < n; k += 16)
				{
					C[idX + j] += A[idX + k] * B[(k + 0) * n + j];
					C[idX + j] += A[idX + k + 1] * B[(k + 1) * n + j];
					C[idX + j] += A[idX + k + 2] * B[(k + 2) * n + j];
					C[idX + j] += A[idX + k + 3] * B[(k + 3) * n + j];
					C[idX + j] += A[idX + k + 4] * B[(k + 4) * n + j];
					C[idX + j] += A[idX + k + 5] * B[(k + 5) * n + j];
					C[idX + j] += A[idX + k + 6] * B[(k + 6) * n + j];
					C[idX + j] += A[idX + k + 7] * B[(k + 7) * n + j];
					C[idX + j] += A[idX + k + 8] * B[(k + 8) * n + j];
					C[idX + j] += A[idX + k + 9] * B[(k + 9) * n + j];
					C[idX + j] += A[idX + k + 10] * B[(k + 10) * n + j];
					C[idX + j] += A[idX + k + 11] * B[(k + 11) * n + j];
					C[idX + j] += A[idX + k + 12] * B[(k + 12) * n + j];
					C[idX + j] += A[idX + k + 13] * B[(k + 13) * n + j];
					C[idX + j] += A[idX + k + 14] * B[(k + 14) * n + j];
					C[idX + j] += A[idX + k + 15] * B[(k + 15) * n + j];
				}
				for (; k < n; k++)
				{
					C[idX + j] += A[idX + k] * B[k * n + j];
				}
			}
		}
		break;

	case 5: // Loop Reordering j-k-i
		for (int j = 0; j < n; j++)
		{
			for (int k = 0; k < n; k++)
			{
				for (int i = 0; i < n; i++)
				{
					C[i * n + j] += A[i * n + k] * B[k * n + j];
				}
			}
		}
		break;

	case 6: // Loop Reordering k-i-j
		for (int k = 0; k < n; k++)
		{
			for (int i = 0; i < n; i++)
			{
				int idX = i * n;
				for (int j = 0; j < n; j++)
				{
					C[idX + j] += A[idX + k] * B[k * n + j];
				}
			}
		}
		break;

	case 7: // Loop Reordering i-k-j
		for (int i = 0; i < n; i++)
		{
			int idX = i * n;
			for (int k = 0; k < n; k++)
			{
				for (int j = 0; j < n; j++)
				{
					C[idX + j] += A[idX + k] * B[k * n + j];
				}
			}
		}
		break;

	case 8: // Loop Reordering j-i-k
		for (int j = 0; j < n; j++)
		{
			for (int i = 0; i < n; i++)
			{
				int idX = i * n;
				for (int k = 0; k < n; k++)
				{
					C[idX + j] += A[idX + k] * B[k * n + j];
				}
			}
		}
		break;

	case 9: // Loop Reordering k-j-i
		for (int k = 0; k < n; k++)
		{
			for (int j = 0; j < n; j++)
			{
				for (int i = 0; i < n; i++)
				{
					C[i * n + j] += A[i * n + k] * B[k * n + j];
				}
			}
		}
		break;

	case 10: // Loop Unrolling @2, k-i-j
		for (int k = 0; k < n; k++)
		{
			for (int i = 0; i < n; i++)
			{
				int idX = i * n;
				for (int j = 0; j <= n - 2; j += 2)
				{
					C[idX + j] += A[idX + k] * B[k * n + j];
					C[idX + j + 1] += A[idX + k] * B[k * n + j + 1];
				}
			}
		}
		break;

	case 11: // Loop Unrolling @2, i-k-j
		for (int i = 0; i < n; i++)
		{
			for (int k = 0; k < n; k++)
			{
				int idxA = i * n + k;
				int baseC = i * n;
				int baseB = k * n;
				double r = A[idxA];
				for (int j = 0; j < n; j += 2)
				{
					C[baseC + j] += r * B[baseB + j];
					C[baseC + j + 1] += r * B[baseB + j + 1];
				}
			}
		}
		break;

	case 12: // Loop Unrolling @4, i-k-j
		for (int i = 0; i < n; i++)
		{
			for (int k = 0; k < n; k++)
			{
				int idxA = i * n + k;
				int baseC = i * n;
				int baseB = k * n;
				double r = A[idxA];
				for (int j = 0; j < n; j += 4)
				{
					C[baseC + j] += r * B[baseB + j];
					C[baseC + j + 1] += r * B[baseB + j + 1];
					C[baseC + j + 2] += r * B[baseB + j + 2];
					C[baseC + j + 3] += r * B[baseB + j + 3];
				}
			}
		}
		break;

	default:
		std::cerr << "Invalid loop optimization ID!" << std::endl;
		break;
	}
}

/**
 * @brief 		Task 1B: Performs matrix multiplication of two matrices using tiling.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		size 		dimension of the matrices
 * @param 		tile_size 	size of the tile
 * @note 		The tile size should be a multiple of the dimension of the matrices.
 * 				For example, if the dimension is 1024, then the tile size can be 32, 64, 128, etc.
 * 				You can assume that the matrices are square matrices.
 */
void tile_mat_mul(double *A, double *B, double *C, int size, int tile_size)
{
	int n = size;
	int T = tile_size;

	for (int i = 0; i < n; i += T)
	{
		for (int j = 0; j < n; j += T)
		{
			for (int k = 0; k < n; k += T)
			{
				// Multiply the mini T x T sized tiles
				for (int ii = i; ii < i + T && ii < n; ii++)
				{
					for (int jj = j; jj < j + T && jj < n; jj++)
					{
						double sum = C[ii * n + jj];
						for (int kk = k; kk < k + T && kk < n; kk++)
						{
							sum += A[ii * n + kk] * B[kk * n + jj];
						}
						C[ii * n + jj] = sum;
					}
				}
			}
		}
	}
}

/**
 * @brief 		Task 1C: Performs matrix multiplication of two matrices using SIMD instructions.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		size 		dimension of the matrices
 * @note 		You can assume that the matrices are square matrices.
 */
void simd_mat_mul(double *A, double *B, double *C, int size, int id)
{
	//----------------------------------------------------- Write your code here ----------------------------------------------------------------

	int n = size;

	switch (id)
	{
	case 1:
		// Case 1: SSE 128-bit (processes 2 doubles at a time)
		cout << "Running SSE 128-bit version" << endl;
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < n; j++)
			{
				__m128d sum = _mm_setzero_pd();
				int k = 0;
				for (; k + 1 < n; k += 2)
				{
					__m128d a = _mm_loadu_pd(&A[i * n + k]);
					__m128d b = _mm_set_pd(B[(k + 1) * n + j], B[k * n + j]);
					sum = _mm_add_pd(sum, _mm_mul_pd(a, b));
				}
				// Horizontal sum
				__m128d temp = _mm_hadd_pd(sum, sum);
				double result;
				_mm_store_sd(&result, temp);
				C[i * n + j] += result;

				// Handle remainder
				for (; k < n; k++)
				{
					C[i * n + j] += A[i * n + k] * B[k * n + j];
				}
			}
		}
		break;

	case 2:
		// Case 2: AVX2 256-bit (processes 4 doubles at a time)
		cout << "Running AVX2 256-bit version" << endl;
		for (int i = 0; i < n; ++i)
		{
			for (int j = 0; j < n; ++j)
			{
				__m256d sum = _mm256_setzero_pd();

				int k = 0;
				for (; k + 3 < n; k += 4)
				{
					// Load 4 A elements
					__m256d a = _mm256_loadu_pd(&A[i * n + k]);

					// Load non-contiguous B elements into vector
					__m256d b = _mm256_set_pd(B[(k + 3) * n + j],
											  B[(k + 2) * n + j],
											  B[(k + 1) * n + j],
											  B[(k + 0) * n + j]);

					sum = _mm256_add_pd(sum, _mm256_mul_pd(a, b));
				}

				double temp[4];
				_mm256_storeu_pd(temp, sum);
				double acc = temp[0] + temp[1] + temp[2] + temp[3];

				// Handle leftover elements
				for (; k < n; ++k)
				{
					acc += A[i * n + k] * B[k * n + j];
				}

				C[i * n + j] += acc;
			}
		}
		break;

	case 3:
		// Case 3: AVX2 Fused Multiply-Add (FMA) for better performance
		cout << "Running AVX2 FMA version" << endl;
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < n; j++)
			{
				__m256d sum = _mm256_setzero_pd();

				int k = 0;
				for (; k + 3 < n; k += 4)
				{
					__m256d a = _mm256_loadu_pd(&A[i * n + k]);
					__m256d b = _mm256_set_pd(B[(k + 3) * n + j],
											  B[(k + 2) * n + j],
											  B[(k + 1) * n + j],
											  B[(k + 0) * n + j]);
					sum = _mm256_fmadd_pd(a, b, sum); // FMA: sum += a*b
				}

				double temp[4];
				_mm256_storeu_pd(temp, sum);
				double acc = temp[0] + temp[1] + temp[2] + temp[3];

				for (; k < n; ++k)
				{
					acc += A[i * n + k] * B[k * n + j];
				}

				C[i * n + j] += acc;
			}
		}
		break;

	default:
		cout << "Invalid id! Please choose 1 (SSE128), 2 (AVX256), or 3 (FMA)." << endl;
		break;
	}

	//-------------------------------------------------------------------------------------------------------------------------------------------
}

/**
 * @brief 		Task 1D: Performs matrix multiplication of two matrices using combination of tiling/SIMD/loop optimization.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		size 		dimension of the matrices
 * @param 		tile_size 	size of the tile
 * @note 		The tile size should be a multiple of the dimension of the matrices.
 * @note 		You can assume that the matrices are square matrices.
 */
void combination_mat_mul(double *A, double *B, double *C, int size, int tile_size, int id)
{
	//----------------------------------------------------- Write your code here ----------------------------------------------------------------
	int n = size;
	int T = tile_size;

	switch (id)
	{
	case 1:
		// Case 1: Loop Reordering i-k-j with SIMD (basic vectorization)
		cout << "Case 1: i-k-j with SIMD" << endl;
		for (int i = 0; i < n; ++i)
		{
			for (int k = 0; k < n; ++k)
			{
				__m256d a_bcast = _mm256_set1_pd(A[i * n + k]);
				int j = 0;
				for (; j + 4 <= n; j += 4)
				{
					__m256d c_val = _mm256_loadu_pd(&C[i * n + j]);
					__m256d b_val = _mm256_loadu_pd(&B[k * n + j]);
					c_val = _mm256_fmadd_pd(a_bcast, b_val, c_val);
					_mm256_storeu_pd(&C[i * n + j], c_val);
				}
				for (; j < n; ++j)
				{
					C[i * n + j] += A[i * n + k] * B[k * n + j];
				}
			}
		}
		break;

	case 2:
		// Case 2: Tiling + SIMD for better cache reuse
		cout << "Case 2: Tiling with SIMD" << endl;
		for (int i = 0; i < n; i += T)
		{
			for (int j = 0; j < n; j += T)
			{
				for (int k = 0; k < n; k += T)
				{
					int i_end = min(i + T, n);
					int j_end = min(j + T, n);
					int k_end = min(k + T, n);

					for (int ii = i; ii < i_end; ++ii)
					{
						for (int kk = k; kk < k_end; ++kk)
						{
							__m256d a_bcast = _mm256_set1_pd(A[ii * n + kk]);
							int jj = j;
							for (; jj + 4 <= j_end; jj += 4)
							{
								__m256d c_val = _mm256_loadu_pd(&C[ii * n + jj]);
								__m256d b_val = _mm256_loadu_pd(&B[kk * n + jj]);
								c_val = _mm256_fmadd_pd(a_bcast, b_val, c_val);
								_mm256_storeu_pd(&C[ii * n + jj], c_val);
							}
							for (; jj < j_end; ++jj)
							{
								C[ii * n + jj] += A[ii * n + kk] * B[kk * n + jj];
							}
						}
					}
				}
			}
		}
		break;

	case 3:
		// Case 3: i-k-j with SIMD and loop unrolling @ 2
		cout << "Case 3: i-k-j with unrolling x2" << endl;
		for (int i = 0; i < n; i++)
		{
			for (int k = 0; k < n; k++)
			{
				__m256d a_broadcast = _mm256_set1_pd(A[i * n + k]);
				int j = 0;
				for (j = 0; j + 8 < n; j += 8)
				{
					__m256d c0 = _mm256_loadu_pd(&C[i * n + j]);
					__m256d b0 = _mm256_loadu_pd(&B[k * n + j]);
					c0 = _mm256_fmadd_pd(a_broadcast, b0, c0);
					_mm256_storeu_pd(&C[i * n + j], c0);

					__m256d c1 = _mm256_loadu_pd(&C[i * n + j + 4]);
					__m256d b1 = _mm256_loadu_pd(&B[k * n + j + 4]);
					c1 = _mm256_fmadd_pd(a_broadcast, b1, c1);
					_mm256_storeu_pd(&C[i * n + j + 4], c1);
				}
				for (; j < n; j++)
				{
					C[i * n + j] += A[i * n + k] * B[k * n + j];
				}
			}
		}
		break;

	case 4:
		// Case 4: i-k-j with SIMD and loop unrolling @ 4
		cout << "Case 4: i-k-j with unrolling x4" << endl;
		for (int i = 0; i < n; ++i)
		{
			for (int k = 0; k < n; ++k)
			{
				__m256d a_bcast = _mm256_set1_pd(A[i * n + k]);
				int j = 0;
				for (; j + 16 <= n; j += 16)
				{
					__m256d c0 = _mm256_loadu_pd(&C[i * n + j + 0]);
					__m256d b0 = _mm256_loadu_pd(&B[k * n + j + 0]);
					c0 = _mm256_fmadd_pd(a_bcast, b0, c0);
					_mm256_storeu_pd(&C[i * n + j + 0], c0);

					__m256d c1 = _mm256_loadu_pd(&C[i * n + j + 4]);
					__m256d b1 = _mm256_loadu_pd(&B[k * n + j + 4]);
					c1 = _mm256_fmadd_pd(a_bcast, b1, c1);
					_mm256_storeu_pd(&C[i * n + j + 4], c1);

					__m256d c2 = _mm256_loadu_pd(&C[i * n + j + 8]);
					__m256d b2 = _mm256_loadu_pd(&B[k * n + j + 8]);
					c2 = _mm256_fmadd_pd(a_bcast, b2, c2);
					_mm256_storeu_pd(&C[i * n + j + 8], c2);

					__m256d c3 = _mm256_loadu_pd(&C[i * n + j + 12]);
					__m256d b3 = _mm256_loadu_pd(&B[k * n + j + 12]);
					c3 = _mm256_fmadd_pd(a_bcast, b3, c3);
					_mm256_storeu_pd(&C[i * n + j + 12], c3);
				}
				for (; j + 4 <= n; j += 4)
				{
					__m256d c = _mm256_loadu_pd(&C[i * n + j]);
					__m256d b = _mm256_loadu_pd(&B[k * n + j]);
					c = _mm256_fmadd_pd(a_bcast, b, c);
					_mm256_storeu_pd(&C[i * n + j], c);
				}
				for (; j < n; ++j)
				{
					C[i * n + j] += A[i * n + k] * B[k * n + j];
				}
			}
		}
		break;

	case 5:
		// Case 5: Tiling + i-k-j + SIMD (best cache and SIMD use)
		cout << "Case 5: Tiling + i-k-j + SIMD (combined)" << endl;
		for (int ii = 0; ii < n; ii += T)
		{
			const int i_end = min(ii + T, n);
			for (int kk = 0; kk < n; kk += T)
			{
				const int k_end = min(kk + T, n);
				for (int jj = 0; jj < n; jj += T)
				{
					const int j_end = min(jj + T, n);
					for (int i = ii; i < i_end; ++i)
					{
						const int baseC = i * n;
						const int baseAi = i * n;
						for (int k = kk; k < k_end; ++k)
						{
							const __m256d a_bcast = _mm256_set1_pd(A[baseAi + k]);
							const int baseBk = k * n;
							int j = jj;
							for (; j + 4 <= j_end; j += 4)
							{
								__m256d c = _mm256_loadu_pd(&C[baseC + j]);
								__m256d b = _mm256_loadu_pd(&B[baseBk + j]);
								c = _mm256_fmadd_pd(a_bcast, b, c);
								_mm256_storeu_pd(&C[baseC + j], c);
							}
							for (; j < j_end; ++j)
							{
								C[baseC + j] += A[baseAi + k] * B[baseBk + j];
							}
						}
					}
				}
			}
		}
		break;

	default:
		cout << "Invalid id! Choose between 1 and 5." << endl;
		break;
	}

	//-------------------------------------------------------------------------------------------------------------------------------------------
}

// NOTE: DO NOT CHANGE ANYTHING BELOW THIS LINE
/**
 * @brief 		Main function
 * @param 		argc 		number of command line arguments
 * @param 		argv 		array of command line arguments
 * @return 		0 on success
 * @note 		DO NOT CHANGE THIS FUNCTION
 * 				DO NOT ADD OR REMOVE ANY COMMAND LINE ARGUMENTS
 */
int main(int argc, char **argv)
{

	if (argc <= 1)
	{
		printf("Usage: %s <matrix_dimension>\n", argv[0]);
		return 0;
	}

	else
	{
		int size = atoi(argv[1]);
		int tile_size = TILE_SIZE;
		// int tile_size = atoi(argv[2]);
		int id = atoi(argv[2]);

		double *A = (double *)malloc(size * size * sizeof(double));
		double *B = (double *)malloc(size * size * sizeof(double));
		double *C = (double *)calloc(size * size, sizeof(double));

		// initialize random seed
		srand(time(NULL));

		// initialize matrices A and B with random values
		initialize_matrix(A, size, size);
		initialize_matrix(B, size, size);

		// perform normal matrix multiplication
		auto start = std::chrono::high_resolution_clock::now();
		naive_mat_mul(A, B, C, size);
		auto end = std::chrono::high_resolution_clock::now();
		auto time_naive_mat_mul = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		printf("Normal matrix multiplication took %ld ms to execute \n\n", time_naive_mat_mul);

#ifdef OPTIMIZE_LOOP_OPT
		// Task 1a: perform matrix multiplication with loop optimization

		// initialize result matrix to 0
		initialize_result_matrix(C, size, size);

		start = std::chrono::high_resolution_clock::now();
		loop_opt_mat_mul(A, B, C, size, id);
		end = std::chrono::high_resolution_clock::now();
		auto time_loop_mat_mul = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		printf("Loop optimized matrix multiplication took %ld ms to execute \n", time_loop_mat_mul);
		printf("Normalized performance: %f \n\n", (double)time_naive_mat_mul / time_loop_mat_mul);
#endif

#ifdef OPTIMIZE_TILING
		// Task 1b: perform matrix multiplication with tiling

		// initialize result matrix to 0
		initialize_result_matrix(C, size, size);

		start = std::chrono::high_resolution_clock::now();
		tile_mat_mul(A, B, C, size, tile_size);
		end = std::chrono::high_resolution_clock::now();
		auto time_tiling_mat_mul = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		printf("Tiling matrix multiplication took %ld ms to execute \n", time_tiling_mat_mul);
		printf("Normalized performance: %f \n\n", (double)time_naive_mat_mul / time_tiling_mat_mul);
#endif

#ifdef OPTIMIZE_SIMD
		// Task 1c: perform matrix multiplication with SIMD instructions

		// initialize result matrix to 0
		initialize_result_matrix(C, size, size);

		start = std::chrono::high_resolution_clock::now();
		simd_mat_mul(A, B, C, size, id);
		end = std::chrono::high_resolution_clock::now();
		auto time_simd_mat_mul = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

		printf("SIMD matrix multiplication took %ld ms to execute \n", time_simd_mat_mul);
		printf("Normalized performance: %f \n\n", (double)time_naive_mat_mul / time_simd_mat_mul);
#endif

#ifdef OPTIMIZE_COMBINED
		// Task 1d: perform matrix multiplication with combination of tiling, SIMD and loop optimization

		// initialize result matrix to 0
		initialize_result_matrix(C, size, size);

		start = std::chrono::high_resolution_clock::now();
		combination_mat_mul(A, B, C, size, TILE_SIZE, id);
		end = std::chrono::high_resolution_clock::now();
		auto time_combination = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		printf("Combined optimization matrix multiplication took %ld ms to execute \n", time_combination);
		printf("Normalized performance: %f \n\n", (double)time_naive_mat_mul / time_combination);
#endif

		// free allocated memory
		free(A);
		free(B);
		free(C);

		return 0;
	}
}
