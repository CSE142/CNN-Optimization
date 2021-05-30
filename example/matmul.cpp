#include<iostream>
#include "pin_tags.h"
#include<vector>
#include "archlab.hpp"
#include <omp.h>

class Matrix {
	int size;
	int * data;
public:
	Matrix(int size) : size(size), data(NULL) {
		data = new int[size*size];

		for(int i = 0; i < size*size; i++)
			data[i] = 0;
	}

	inline int & operator()(int row, int column) {
		return data[row*size + column];
	}
	inline const int & operator()(int row, int column) const {
		return data[row*size + column];
	}
	~Matrix() {
		delete [] data;
	}
};

//#define SIZE 2
void simple(int SIZE, int tile_size) {
	int A[SIZE][SIZE];
	int B[SIZE][SIZE];
	int C[SIZE][SIZE];
	//int c = 0;
	std::stringstream ss;
	
	ss << "Size-" << SIZE;
	
	NEW_TRACE(ss.str().c_str());
	START_TRACE();
	for(int i = 0; i < SIZE; i++) {
		for(int j = 0; j < SIZE; j++) {
			for(int k = 0; k < SIZE; k++) {
				//std::cout << c++ << ": C[" << i << ", " << j << "] A[" << i << ", " << k <<"] B[" << k << ", " <<j << "]\n";
				//std::cout << c++ << ": (i, j, k) = (" << i << ", " << j << ", " << k << ")\n";
				C[i][j]	+= A[i][k] * B[k][j];
			}
		}
	}


}

void simple_big(int SIZE, int tile_size) {
        Matrix A(SIZE);
        Matrix B(SIZE);
        Matrix C(SIZE);
	//int c = 0;
	std::stringstream ss;
	
	ss << "Size-" << SIZE;
	
	NEW_TRACE(ss.str().c_str());
	START_TRACE();
	for(int i = 0; i < SIZE; i++) {
		for(int j = 0; j < SIZE; j++) {
			for(int k = 0; k < SIZE; k++) {
				//std::cout << c++ << ": C[" << i << ", " << j << "] A[" << i << ", " << k <<"] B[" << k << ", " <<j << "]\n";
				//std::cout << c++ << ": (i, j, k) = (" << i << ", " << j << ", " << k << ")\n";
				C(i, j)	+= A(i, k) * B(k, j);
			}
		}
	}


}

void simple_simd(int SIZE, int tile_size) {
        Matrix A(SIZE);
        Matrix B(SIZE);
        Matrix C(SIZE);
	//int c = 0;
	std::stringstream ss;
	
	ss << "Size-" << SIZE;
	
	NEW_TRACE(ss.str().c_str());
	START_TRACE();
	for(int i = 0; i < SIZE; i++) {
		for(int j = 0; j < SIZE; j++) {
#pragma omp simd
			for(int k = 0; k < SIZE; k++) {
				//std::cout << c++ << ": C[" << i << ", " << j << "] A[" << i << ", " << k <<"] B[" << k << ", " <<j << "]\n";
				//std::cout << c++ << ": (i, j, k) = (" << i << ", " << j << ", " << k << ")\n";
				C(i, j)	+= A(i, k) * B(k, j);
			}
		}
	}


}

void simple_pfor(int SIZE, int tile_size) {
        Matrix A(SIZE);
        Matrix B(SIZE);
        Matrix C(SIZE);
	//int c = 0;
	std::stringstream ss;
	
	ss << "Size-" << SIZE;
	
	NEW_TRACE(ss.str().c_str());
	START_TRACE();
	for(int i = 0; i < SIZE; i++) {
		for(int j = 0; j < SIZE; j++) {
#pragma omp parallel for
			for(int k = 0; k < SIZE; k++) {
				C(i, j)	+= A(i, k) * B(k, j);
			}
		}
	}


}

void simple_pfor_renest(int SIZE, int tile_size) {
        Matrix A(SIZE);
        Matrix B(SIZE);
        Matrix C(SIZE);
	//int c = 0;
	std::stringstream ss;
	
	ss << "Size-" << SIZE;
	
	NEW_TRACE(ss.str().c_str());
	START_TRACE();
#pragma omp parallel for
	for(int k = 0; k < SIZE; k++) {
		Matrix _C(SIZE);
		for(int i = 0; i < SIZE; i++) {
			for(int j = 0; j < SIZE; j++) {
				_C(i, j) += A(i, k) * B(k, j);
			}
		}
#pragma omp critical
		{
			for(int i = 0; i < SIZE; i++) {
#pragma omp simd
				for(int j = 0; j < SIZE; j++) {
					C(i,j) += _C(i,j);
				}
			}
		}
	}


}

//#undef SIZE
// #define SIZE 16
							//std::cout << "C[" << i << ", " << j << "] A[" << i << ", " << k <<"] B[" << k << ", " <<j << "]\n";

void tiled_big( int SIZE, int tile_size) {
        Matrix A(SIZE);
        Matrix B(SIZE);
        Matrix C(SIZE);

	for(int ii = 0; ii < tile_size; ii += tile_size) {
		for(int jj = 0; jj < tile_size; jj += tile_size) {
			for(int kk = 0; kk < tile_size; kk += tile_size) {

				for(int i = ii; i < (ii + tile_size); i++) {
					for(int j = jj; j < (jj + tile_size); j++) {
						for(int k = kk; k < (kk + tile_size); k++) {
							C(i, j)	+= A(i, k) * B(k, j);
						}
					}
				}
				
			}
		}
	}
}

void tiled( int SIZE, int tile_size) {
	int A[SIZE][SIZE];
	int B[SIZE][SIZE];
	int C[SIZE][SIZE];

	for(int ii = 0; ii < tile_size; ii += tile_size) {
		for(int jj = 0; jj < tile_size; jj += tile_size) {
			for(int kk = 0; kk < tile_size; kk += tile_size) {

				for(int i = ii; i < (ii + tile_size); i++) {
					for(int j = jj; j < (jj + tile_size); j++) {
						for(int k = kk; k < (kk + tile_size); k++) {
							C[i][j]	+= A[i][k] * B[k][j];
						}
					}
				}
				
			}
		}
	}
}
int s =0;
void dotprod( int SIZE, int tile_size) {

	int *A =new int[SIZE];
	int *B =new int[SIZE];
	int _s = 0;
	for(int i = 0; i < SIZE; i++) {
	     	_s += A[i] *B[i];
	}
	s = _s;
}

void dotprod_simd( int SIZE, int tile_size) {

	int *A =new int[SIZE];
	int *B =new int[SIZE];
	int _s = 0;
#pragma omp simd
	for(int i = 0; i < SIZE; i++) {
	     	_s += A[i] *B[i];
	}
	s = _s;
}

int main(int argc, char *argv[]) {
	std::vector<int> omp_threads_values;
	std::vector<int> default_omp_threads_values;
	default_omp_threads_values.push_back(1);
	archlab_add_multi_option<std::vector<int> >("threads",
					      omp_threads_values,
					      default_omp_threads_values,
					      "1",
					      "How many threads use.  Pass multiple values to run with multiple thread counts.");

	std::vector<int> tile_sizes;
	std::vector<int> default_tile_sizes;
	default_tile_sizes.push_back(1);
	archlab_add_multi_option<std::vector<int> >("tile",
					      tile_sizes,
					      default_tile_sizes,
					      "1",
					      "Tile size.  Pass multiple values to run with multiple thread counts.");

	std::vector<int> matrix_sizes;
	std::vector<int> default_matrix_sizes;
	default_matrix_sizes.push_back(128);
	archlab_add_multi_option<std::vector<int> >("size",
					      matrix_sizes,
					      default_matrix_sizes,
					      "128",
					      "Matrix size.  Pass multiple values to run with multiple sizes.");

	std::vector<std::string> impls;
	std::vector<std::string> default_impls;
	default_impls.push_back("simple");
	archlab_add_multi_option<std::vector<std::string>>("impl",
							   impls,
							   default_impls,
							   "simple",
							   "Which versions to run");

	archlab_parse_cmd_line(&argc, argv);


	std::map<const std::string, void(*)(int, int)>
		impl_map =
		{
#define IMPL(n) {#n, n}
			IMPL(tiled_big),
			IMPL(simple_big),
			IMPL(simple_simd),
			IMPL(simple_pfor),
			IMPL(tiled),
			IMPL(simple),
			IMPL(simple_pfor_renest),
			IMPL(dotprod_simd),
			IMPL(dotprod),
		};


	for(auto & tile_size: tile_sizes ) {
		for(auto & impl : impls) {
			for(auto & thread_count: omp_threads_values ) {
				omp_set_num_threads(thread_count);
				for(auto size: matrix_sizes) {
					
					pristine_machine();
					theDataCollector->disable_prefetcher();
					{
						ArchLabTimer timer;
						
						timer.attr("size", size);
						timer.attr("thread_count", thread_count);
						timer.attr("impl", impl);
						timer.attr("tile_size", tile_size);
						timer.go();
						impl_map[impl](size, tile_size);
					}
				}
			}
		}
	}
	
	archlab_write_stats();	
}

