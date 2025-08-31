#include "matrix_operation.h"
#include <immintrin.h>

Matrix MatrixOperation::NaiveMatMul(const Matrix &A, const Matrix &B)
{
	size_t n = A.getRows();
	size_t k = A.getCols();
	size_t m = B.getCols();

	if (k != B.getRows())
	{
		throw std::invalid_argument("Matrix dimensions don't match for multiplication");
	}

	Matrix C(n, m);

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			for (int l = 0; l < k; l++)
			{
				C(i, j) += A(i, l) * B(l, j);
			}
		}
	}

	return C;
}

// Loop reordered matrix multiplication (ikj order for better cache locality)
Matrix MatrixOperation::ReorderedMatMul(const Matrix &A, const Matrix &B)
{
	size_t n = A.getRows();
	size_t k = A.getCols();
	size_t m = B.getCols();

	if (k != B.getRows())
	{
		throw std::invalid_argument("Matrix dimensions don't match for multiplication");
	}

	Matrix C(n, m);

	//----------------------------------------------------- Write your code here ----------------------------------------------------------------
	for (int i = 0; i < n; i++)
	{
		for (int l = 0; l < k; l++)
		{
			for (int j = 0; j < m; j++)
			{
				C(i, j) += A(i, l) * B(l, j);
			}
		}
	}

	//-------------------------------------------------------------------------------------------------------------------------------------------

	return C;
}

// Loop unrolled matrix multiplication
Matrix MatrixOperation::UnrolledMatMul(const Matrix &A, const Matrix &B)
{
	size_t n = A.getRows();
	size_t k = A.getCols();
	size_t m = B.getCols();

	if (k != B.getRows())
	{
		throw std::invalid_argument("Matrix dimensions don't match for multiplication");
	}

	Matrix C(n, m);

	const int UNROLL = 16; 

for (int i = 0; i < n; i++)
{
    for (int j = 0; j < m; j++)
    {
        double sum = 0.0; // register accumulator

        int l = 0;
        // main unrolled loop
        for (; l <= k - UNROLL; l += UNROLL)
        {
            sum += A(i, l)     * B(l, j);
            sum += A(i, l + 1) * B(l + 1, j);
            sum += A(i, l + 2) * B(l + 2, j);
            sum += A(i, l + 3) * B(l + 3, j);
            sum += A(i, l + 4) * B(l + 4, j);
            sum += A(i, l + 5) * B(l + 5, j);
            sum += A(i, l + 6) * B(l + 6, j);
            sum += A(i, l + 7) * B(l + 7, j);
            sum += A(i, l + 8) * B(l + 8, j);
            sum += A(i, l + 9) * B(l + 9, j);
            sum += A(i, l + 10) * B(l + 10, j);
            sum += A(i, l + 11) * B(l + 11, j);
            sum += A(i, l + 12) * B(l + 12, j);
            sum += A(i, l + 13) * B(l + 13, j);
            sum += A(i, l + 14) * B(l + 14, j);
            sum += A(i, l + 15) * B(l + 15, j);
        }

        // handle remaining elements
        for (; l < k; l++)
        {
            sum += A(i, l) * B(l, j);
        }

        C(i, j) = sum;
    }
}

return C;

}

// Tiled (blocked) matrix multiplication for cache efficiency
Matrix MatrixOperation::TiledMatMul(const Matrix &A, const Matrix &B)
{
	size_t n = A.getRows();
	size_t k = A.getCols();
	size_t m = B.getCols();

	if (k != B.getRows())
	{
		throw std::invalid_argument("Matrix dimensions don't match for multiplication");
	}

	Matrix C(n, m);
	const size_t T = 32; // tile size
	size_t i_max = 0;
	size_t k_max = 0;
	size_t j_max = 0;
	//----------------------------------------------------- Write your code here ----------------------------------------------------------------

	for (int i = 0; i < n; i += T)
	{
		i_max = (i + T < n) ? i + T : n;
		for (int j = 0; j < m; j += T)
		{
			j_max = (j + T < m) ? j + T : m;
			for (int l = 0; l < k; l += T)
			{
				k_max = (l + T < k) ? l + T : k;
				{
					for (int ii = i; ii < i_max; ii++)
					{
						for (int jj = j; jj < j_max; jj++)
						{
							for (int ll = l; ll < k_max; ll++)
							{
								C(ii, jj) += A(ii, ll) * B(ll, jj);
							}
						}
					}
				}
			}
		}
	}

	//-------------------------------------------------------------------------------------------------------------------------------------------

	return C;
}

// SIMD vectorized matrix multiplication (using AVX2)
Matrix MatrixOperation::VectorizedMatMul(const Matrix &A, const Matrix &B)
{
	size_t n = A.getRows();
	size_t k = A.getCols();
	size_t m = B.getCols();

	if (k != B.getRows())
	{
		throw std::invalid_argument("Matrix dimensions don't match for multiplication");
	}

	Matrix C(n, m);
	//----------------------------------------------------- Write your code here ----------------------------------------------------------------

	for (size_t i = 0; i < n; i++)
	{
		for (size_t j = 0; j < m; j++)
		{
			__m256d sum = _mm256_setzero_pd();

			size_t l = 0;
			for (; l + 4 <= k; l += 4)
			{
				__m256d a = _mm256_loadu_pd(&A(i, l));

				__m256d b = _mm256_set_pd(
					B(l + 3, j),
					B(l + 2, j),
					B(l + 1, j),
					B(l, j));

				__m256d prod = _mm256_mul_pd(a, b);
				sum = _mm256_add_pd(sum, prod);
			}

			double temp[4];
			_mm256_storeu_pd(temp, sum);
			double result = temp[0] + temp[1] + temp[2] + temp[3];

			for (; l < k; l++)
			{
				result += A(i, l) * B(l, j);
			}

			C(i, j) = result;
		}
	}

	//-------------------------------------------------------------------------------------------------------------------------------------------

	return C;
}

// Optimized matrix transpose
Matrix MatrixOperation::Transpose(const Matrix &A)
{
	size_t rows = A.getRows();
	size_t cols = A.getCols();
	Matrix result(cols, rows);
	const int T = 48;

	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			result(j, i) = A(i, j);
		}
	}

	// Optimized transpose using blocking for better cache performance
	// This is a simple implementation, more advanced techniques can be applied
	// Write your code here and commnent the above code
	//----------------------------------------------------- Write your code here ----------------------------------------------------------------
	int i_max = 0;
	int j_max = 0;
	for (int i = 0; i < rows; i += T)
	{
		i_max = (i + T < rows) ? i + T : rows;
		for (int j = 0; j < cols; j += T)
		{
			j_max = (j + T < cols) ? j + T : cols;
			for (int ii = i; ii < i_max; ii++)
			{
				for (int jj = j; jj < j_max; jj++)
				{
					result(jj, ii) = A(ii, jj);
				}
			}
		}
	}

	//-------------------------------------------------------------------------------------------------------------------------------------------

	return result;
}
