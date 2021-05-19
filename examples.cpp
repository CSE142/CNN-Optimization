#include <cstdlib>
#include "archlab.hpp"
#include"pin_tags.h"
uint *openmp_threads_example(unsigned long int size) {
	uint * array = new uint[size];

	
	for (uint j = 0; j < 3; j++) {
#pragma omp parallel for 
		for(uint i= 0 ; i < size; i++) {
			array[i]+= i*j;
		}
	}
	
	return array;
}
uint *openmp_threads_simd_example(unsigned long int size) {
	uint * array = new uint[size];

	for (uint j = 0; j < 3; j++) {
#pragma omp parallel for  simd
		for(uint i= 0 ; i < size; i++) {
			array[i]+= i*j;
		}
	}
	
	return array;
}
uint *openmp_simd_example(unsigned long int size) {
	uint * array = new uint[size];

	for (uint j = 0; j < 3; j++) {
#pragma omp simd
		for(uint i= 0 ; i < size; i++) {
			array[i] += i*j;
		}
	}
	
	return array;
}
uint *openmp_nothing_example(unsigned long int size) {
	uint * array = new uint[size];

	for (uint j = 0; j < 3; j++) {
		for(uint i= 0 ; i < size; i++) {
			array[i] += i*j;
		}
	}
	
	return array;
}

uint *gcc_simd_example(unsigned long int size) {
	typedef uint v8ui __attribute__ ((vector_size (32)));
	
	uint * array = (uint*)aligned_alloc(32, size*sizeof(uint));
	assert(sizeof(uint)==4);
	for (uint j = 0; j < 3; j++) {
		for(uint i= 0 ; i < size; i+=8) {
			v8ui *v = (v8ui*)&array[i];
			v8ui t = {(i)*j,
				  (i+1)*j,
				  (i+2)*j,
				  (i+3)*j,
				  (i+4)*j,
				  (i+5)*j,
				  (i+6)*j,
				  (i+7)*j};
		 
			*v += t; //array[i]+= i*j;
		}
	}
	
	return array;
}



uint *openmp_threads_gcc_simd_example(unsigned long int size) {
	typedef uint v8ui __attribute__ ((vector_size (32)));
	
	uint * array = (uint*)aligned_alloc(32, size*sizeof(uint));
	assert(sizeof(uint)==4);
	for (uint j = 0; j < 3; j++) {
#pragma omp parallel for 
		for(uint i= 0 ; i < size; i+=8) {
			v8ui *v = (v8ui*)&array[i];
			v8ui t = {(i)*j,
				  (i+1)*j,
				  (i+2)*j,
				  (i+3)*j,
				  (i+4)*j,
				  (i+5)*j,
				  (i+6)*j,
				  (i+7)*j};
		 
			*v += t; //array[i]+= i*j;
		}
	}
	
	return array;
}

#define RUN_EXAMPLE(FUNC)						\
	{								\
		ArchLabTimer timer;					\
		pristine_machine();					\
		set_cpu_clock_frequency(mhz);				\
		theDataCollector->disable_prefetcher();			\
		timer.attr("function", #FUNC);				\
		timer.go();						\
		DUMP_START_ALL(#FUNC, false);					\
		uint * t = FUNC(size);					\
		DUMP_STOP(#FUNC);					\
		delete t;						\
	}								


void run_examples(int mhz, unsigned long int size)
{
		
	
	START_TRACE();
	RUN_EXAMPLE(openmp_threads_example);
	RUN_EXAMPLE(openmp_threads_simd_example);
	RUN_EXAMPLE(openmp_simd_example);
	RUN_EXAMPLE(openmp_nothing_example);
	RUN_EXAMPLE(gcc_simd_example);
	RUN_EXAMPLE(openmp_threads_gcc_simd_example);

}
	

int main(int argc, char *argv[])
{

	load_frequencies(); // this grabs the frequencies we can set.
	int mhz = cpu_frequencies_array[0];
	unsigned long int size;
	archlab_add_option<unsigned long int>("size", size, 1024*1024*1024, "size of the array");
	archlab_parse_cmd_line(&argc, argv);

	run_examples(mhz, size);
	
	archlab_write_stats();
	return 0;
}
