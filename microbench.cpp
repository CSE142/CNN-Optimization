#include <cstdlib>
#include "archlab.hpp"
#include"pin_tags.h"
#include"omp.h"
#include <unordered_set>

uint *serial(unsigned long int size) {
	DUMP_START_ALL("alloc", false);
	uint * array = new uint[size];
	DUMP_STOP("alloc");
	DUMP_START_ALL("clear", false);
	for(uint i = 0; i < size; i++) {
		array[i] = 0;
	}
	DUMP_STOP("clear");
	DUMP_START_ALL("compute", false);
	for (uint j = 0; j < 3; j++) {
		for(uint i= 0 ; i < size; i++) {
			array[i] += i*j;
		}
	}
	DUMP_STOP("compute");
	
	return array;
}





uint *serial_improved(unsigned long int size) {
	DUMP_START_ALL("alloc", false);
	uint * array = new uint[size];
	DUMP_STOP("alloc");
	DUMP_START_ALL("clear", false);
	for(uint i = 0; i < size; i++) {
		array[i] = 0;
	}
	DUMP_STOP("clear");
	DUMP_START_ALL("compute", false);
	for (uint j = 0; j < 3; j++) {
		uint i = 0;
		const uint s = size - (size % 8);
#pragma omp simd
		for(i = 0; i < s; i+=8) {
			array[i] += i*j;
			array[i+1] += (i+1)*j;
			array[i+2] += (i+2)*j;
			array[i+3] += (i+3)*j;
			array[i+4] += (i+4)*j;
			array[i+5] += (i+5)*j;
			array[i+6] += (i+6)*j;
			array[i+7] += (i+7)*j;
		}
		for(; i < size; i++) {
			array[i] += i*j;
		}
	}
	DUMP_STOP("compute");
	
	return array;
}

uint *openmp_threads(unsigned long int size) {
	uint * array = new uint[size];
	for(uint i = 0; i < size; i++) {
		array[i] = 0;
	}
	
	
	for (uint j = 0; j < 3; j++) {
#pragma omp parallel for 
		for(uint i= 0 ; i < size; i++) {
			array[i]+= i*j;
		}
	}
	
	return array;
}

uint *openmp_simd(unsigned long int size) {
	DUMP_START_ALL("alloc", false);
	uint * array = new uint[size];
	DUMP_STOP("alloc");
	DUMP_START_ALL("clear", false);
#pragma omp simd
	for(uint i = 0; i < size; i++) {
		array[i] = 0;
	}
	
	DUMP_STOP("clear");
	DUMP_START_ALL("compute", false);

	for (uint j = 0; j < 3; j++) {
#pragma omp simd
		for(uint i= 0 ; i < size; i++) {
			array[i] += i*j;
		}
	}
	DUMP_STOP("compute");
	
	return array;
}


uint *openmp_threads_simd(unsigned long int size) {
	uint * array = new uint[size];
#pragma omp parallel for  simd
	for(uint i = 0; i < size; i++) {
		array[i] = 0;
	}
	

	for (uint j = 0; j < 3; j++) {
#pragma omp parallel for  simd
		for(uint i= 0 ; i < size; i++) {
			array[i]+= i*j;
		}
	}
	
	return array;
}
uint *gcc_simd(unsigned long int size) {
	typedef uint v8ui __attribute__ ((vector_size (32)));
	
	uint * array = (uint*)aligned_alloc(32, size*sizeof(uint));
	assert(sizeof(uint)==4);

	for(uint i = 0; i < size; i++) {
		array[i] = 0;
	}
	
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



uint *openmp_threads_gcc_simd(unsigned long int size) {
	typedef uint v8ui __attribute__ ((vector_size (32)));
	
	uint * array = (uint*)aligned_alloc(32, size*sizeof(uint));
	assert(sizeof(uint)==4);
	for(uint i = 0; i < size; i++) {
		array[i] = 0;
	}
	

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

uint *malloc_test(unsigned long int size) {
	std::unordered_set<int*> allocs;


	for(unsigned long int i = 0; i < size ;i++) {
		if ((i & 0x3) == 0) {
			if (allocs.size() > 0) {
				auto t = allocs.begin();
				delete *t;
				allocs.erase(t);
			}
		} else {
			allocs.insert(new int);
		}
	}
	return NULL;
}


int main(int argc, char *argv[])
{

	load_frequencies(); // this grabs the frequencies we can set.
	int mhz = cpu_frequencies_array[0];
	std::vector<std::string> impls;
	std::vector<std::string> default_impls;

	default_impls.push_back("serial");
	archlab_add_multi_option<std::vector<std::string>>("impl",
							   impls,
							   default_impls,
							   "serial",
							   "Which versions to run");

	std::vector<int> omp_threads_values;
	std::vector<int> default_omp_threads_values;
	default_omp_threads_values.push_back(1);
	archlab_add_multi_option<std::vector<int> >("threads",
					      omp_threads_values,
					      default_omp_threads_values,
					      "1",
					      "How many threads use.  Pass multiple values to run with multiple thread counts.");

	std::vector<unsigned long int> sizes;
	std::vector<unsigned long int> default_sizes;
	default_sizes.push_back(1024*1024*512);
	archlab_add_multi_option<std::vector<unsigned long int> >("size",
					      sizes,
					      default_sizes,
					      "1024*1024*512",
					      "Size.  Pass multiple values to run with multiple sizes..");
	archlab_parse_cmd_line(&argc, argv);
	std::map<const std::string, uint*(*)(unsigned long int)>
		impl_map =
		{
#define IMPL(n) {#n, n}
			IMPL(openmp_threads),
			IMPL(openmp_threads_simd),
			IMPL(openmp_simd),
			IMPL(serial),
			IMPL(gcc_simd),
			IMPL(openmp_threads_gcc_simd),
			IMPL(malloc_test),
			IMPL(serial_improved)
		};

	for(auto & size: sizes ) {
		for(auto & impl : impls) {
			for(auto & thread_count: omp_threads_values ) {
				omp_set_num_threads(thread_count);
				theDataCollector->register_tag("omp_threads", thread_count);
				theDataCollector->register_tag("size",size);
				START_TRACE();
				{								
					ArchLabTimer timer;					
					pristine_machine();					
					set_cpu_clock_frequency(mhz);				
					NEW_TRACE(impl.c_str());					
					theDataCollector->disable_prefetcher();			
					timer.attr("function", impl.c_str());				
					timer.go();						
					DUMP_START_ALL(impl.c_str(), false);
					if (impl_map.find(impl) == impl_map.end()) {
						std::cerr << "Unknow implementation: " << impl <<"\n";
						exit(1);
					}
					uint * t = impl_map[impl](size);					
					DUMP_STOP(impl.c_str());					
					if (t)
						delete t;						
				}								
			}
		}
	}
				
	archlab_write_stats();
	return 0;
}
