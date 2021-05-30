//#define INCLUDE_TESTS
#define DEBUG_OUTPUT "output/"

#include <iostream>
#include "gtest/gtest.h"
#include <opt_cnn.hpp>
#include <sstream>
#include "parameters.hpp"
#include "archlab.hpp"
#include"omp.h"

namespace Tests {

	class OptimizationTests :  public ::testing::Test {
		
	};

	TEST_F(OptimizationTests, level_0_fc) {
		fc_test_activate<opt_fc_layer_t>   (1,1,1,1,1,1);
		fc_test_calc_grads<opt_fc_layer_t> (1,1,1,1,1,1);
		fc_test_fix_weights<opt_fc_layer_t>(1,1,1,1,1,1);
		fc_test<opt_fc_layer_t>            (1,1,1,1,1,1);
	}	
			  
	TEST_F(OptimizationTests, level_1_fc) {
#define FC_TEST_1(method)						\
		method<opt_fc_layer_t>(4,  4,  4,  4, 4, 1);\
		method<opt_fc_layer_t>(8,  8,  2,  2, 16,1);\
		method<opt_fc_layer_t>(32, 32, 16, 8, 8, 1);
		
		FC_TEST_1(fc_test_activate);
		FC_TEST_1(fc_test_calc_grads);
		FC_TEST_1(fc_test_fix_weights);
		FC_TEST_1(fc_test);
	}
	TEST_F(OptimizationTests, level_2_fc) {
#define FC_TEST_2(method)						\
		method<opt_fc_layer_t>(4,  6,  6,  6,  6,  1);\
		method<opt_fc_layer_t>(12, 12, 3,  2,  3,  1);\
		method<opt_fc_layer_t>(16, 96, 2,  2,  12, 1);
		
		FC_TEST_2(fc_test_activate);
		FC_TEST_2(fc_test_calc_grads);
		FC_TEST_2(fc_test_fix_weights);
		FC_TEST_2(fc_test);
	}

	TEST_F(OptimizationTests, level_3_fc) {
#define FC_TEST_3(method)						\
		method<opt_fc_layer_t>(3,  7,  13, 3, 7,  1);\
		method<opt_fc_layer_t>(31, 29, 5,  5, 13, 1);\
		method<opt_fc_layer_t>(3,  17, 31, 3, 23, 1);
		
		FC_TEST_3(fc_test_activate);
		FC_TEST_3(fc_test_calc_grads);
		FC_TEST_3(fc_test_fix_weights);
		FC_TEST_3(fc_test);
	}

	TEST_F(OptimizationTests, level_4_fc) {
		for (int i = 0; i < 10; i++) {
			srand(i);
			int x = RAND_LARGE(16);
			int y = RAND_LARGE(24);
			int z = RAND_LARGE(24);
			int b = RAND_LARGE(16);
			int out = RAND_LARGE(8);
			
			fc_test_activate<opt_fc_layer_t>(x,y,z,b,out,1);
			fc_test_calc_grads<opt_fc_layer_t>(x,y,z,b,out,1);
			fc_test_fix_weights<opt_fc_layer_t>(x,y,z,b,out,1);
			fc_test<opt_fc_layer_t>(x,y,z,b,out,1);

		}
		
	}


	TEST_F(OptimizationTests, level_0_conv) {
		conv_test_activate<opt_conv_layer_t>   (1,1,1,1,1,1,1,1,1);
		conv_test_calc_grads<opt_conv_layer_t> (1,1,1,1,1,1,1,1,1);
		conv_test_fix_weights<opt_conv_layer_t>(1,1,1,1,1,1,1,1,1);
		conv_test<opt_conv_layer_t>            (1,1,1,1,1,1,1,1,1);
	}	
			  
	TEST_F(OptimizationTests, level_1_conv) {
#define CONV_TEST_1(method)						\
		method<opt_conv_layer_t>(4,  4,  1,  4, 1,  4, 2, 1.0, 1); \
		method<opt_conv_layer_t>(8,  8,  4,  2, 3, 16, 8, 1.0, 1); \
		method<opt_conv_layer_t>(32, 32, 8,  8, 1,  8, 2, 0.0, 1);	
		
		CONV_TEST_1(conv_test_activate);
		CONV_TEST_1(conv_test_calc_grads);
		CONV_TEST_1(conv_test_fix_weights);
		CONV_TEST_1(conv_test);
		
	}
	TEST_F(OptimizationTests, level_2_conv) {
#define CONV_TEST_2(method)						\
		method<opt_conv_layer_t>(4,  6,  6,  1,  2,  6,  6,  1.0, 1);\
		method<opt_conv_layer_t>(12, 12, 3,  6,  3, 8,  12, 1.0, 1);\
		method<opt_conv_layer_t>(16, 96, 2,  14, 10, 10, 14, 0.0, 1);
		
		CONV_TEST_2(conv_test_activate);
		CONV_TEST_2(conv_test_calc_grads);
		CONV_TEST_2(conv_test_fix_weights);
		CONV_TEST_2(conv_test);
	}

	TEST_F(OptimizationTests, level_3_conv) {
#define CONV_TEST_3(method)						\
		method<opt_conv_layer_t>(3,  7,  13, 3, 3,  5,  7, 1.0, 1); \
		method<opt_conv_layer_t>(5,  9,  17, 5, 5,  11, 7, 1.0, 1); \
		method<opt_conv_layer_t>(89, 31, 7,  7, 19, 23, 3, 1.0, 1);

		CONV_TEST_3(conv_test_activate);
		CONV_TEST_3(conv_test_calc_grads);
		CONV_TEST_3(conv_test_fix_weights);
		CONV_TEST_3(conv_test);
		
	}

	TEST_F(OptimizationTests, level_4_conv) {
		for (int i = 0; i < 10; i++) {
			srand(i);
			int x = RAND_LARGE(16);
			int y = RAND_LARGE(24);
			int z = RAND_LARGE(24);
			int b = RAND_LARGE(16);
			int stride = RAND_LARGE(4);
			int kernel_size = RAND_LARGE(8);
			int kernel_count = RAND_LARGE(8);
			
			conv_test_activate<opt_conv_layer_t>(x,y,z,b,stride, kernel_size, kernel_count, 1.0,i);
			conv_test_calc_grads<opt_conv_layer_t>(x,y,z,b,stride, kernel_size, kernel_count, 1.0,i);
			conv_test_fix_weights<opt_conv_layer_t>(x,y,z,b,stride, kernel_size, kernel_count, 1.0,i);
			conv_test<opt_conv_layer_t>(x,y,z,b,stride, kernel_size, kernel_count, 1.0,i);

		}
		
	}


	TEST_F(OptimizationTests, level_0_pool) {
		pool_test_activate<opt_pool_layer_t>   (1,1,1,1,1,1,1,1);
		pool_test_calc_grads<opt_pool_layer_t> (1,1,1,1,1,1,1,1);
		pool_test_fix_weights<opt_pool_layer_t>(1,1,1,1,1,1,1,1);
		pool_test<opt_pool_layer_t>            (1,1,1,1,1,1,1,1);
	}	
			  
	TEST_F(OptimizationTests, level_1_pool) {
#define POOL_TEST_1(method)						\
		method<opt_pool_layer_t>(4,  4,  1,  4, 1,  4, 1.0, 1); \
		method<opt_pool_layer_t>(8,  8,  4,  2, 3, 16, 1.0, 1); \
		method<opt_pool_layer_t>(32, 32, 8,  8, 1,  8, 0.0, 1);	
		
		POOL_TEST_1(pool_test_activate);
		POOL_TEST_1(pool_test_calc_grads);
		POOL_TEST_1(pool_test_fix_weights);
		POOL_TEST_1(pool_test);
		
	}
	TEST_F(OptimizationTests, level_2_pool) {
#define POOL_TEST_2(method)						\
		method<opt_pool_layer_t>(4,  6,  6,  1,  2,  6,   1.0, 1);\
		method<opt_pool_layer_t>(12, 12, 3,  6,  3, 8,    1.0, 1);\
		method<opt_pool_layer_t>(16, 96, 2,  14, 10, 10,  0.0, 1);
		
		POOL_TEST_2(pool_test_activate);
		POOL_TEST_2(pool_test_calc_grads);
		POOL_TEST_2(pool_test_fix_weights);
		POOL_TEST_2(pool_test);
	}

	TEST_F(OptimizationTests, level_3_pool) {
#define POOL_TEST_3(method)						\
		method<opt_pool_layer_t>(3,  7,  13, 3, 3,  5,  1.0, 1); \
		method<opt_pool_layer_t>(5,  9,  17, 5, 5,  11, 1.0, 1); \
		method<opt_pool_layer_t>(89, 31, 7,  7, 19, 23, 1.0, 1);

		POOL_TEST_3(pool_test_activate);
		POOL_TEST_3(pool_test_calc_grads);
		POOL_TEST_3(pool_test_fix_weights);
		POOL_TEST_3(pool_test);
		
	}

        TEST_F(OptimizationTests, level_4_pool) {
		for (int i = 0; i < 10; i++) {
			srand(i);
			int x = RAND_LARGE(16);
			int y = RAND_LARGE(24);
			int z = RAND_LARGE(24);
			int b = RAND_LARGE(16);
			int stride = RAND_LARGE(4);
			int kernel_size = RAND_LARGE(8);
			
			pool_test_activate<opt_pool_layer_t>(x,y,z,b,stride, kernel_size, 1.0,i);
			pool_test_calc_grads<opt_pool_layer_t>(x,y,z,b,stride, kernel_size, 1.0,i);
			pool_test_fix_weights<opt_pool_layer_t>(x,y,z,b,stride, kernel_size, 1.0,i);
			pool_test<opt_pool_layer_t>(x,y,z,b,stride, kernel_size, 1.0,i);

		}
		
	}

	TEST_F(OptimizationTests, level_0_relu) {
		relu_test_activate<opt_relu_layer_t>   (1,1,1,1,1);
		relu_test_calc_grads<opt_relu_layer_t> (1,1,1,1,1);
		relu_test_fix_weights<opt_relu_layer_t>(1,1,1,1,1);
		relu_test<opt_relu_layer_t>            (1,1,1,1,1);
	}	
			  
	TEST_F(OptimizationTests, level_1_relu) {
#define RELU_TEST_1(method)						\
		method<opt_relu_layer_t>(4,  4,  1,  4,  1); \
		method<opt_relu_layer_t>(8,  8,  4,  2,  1); \
		method<opt_relu_layer_t>(32, 32, 8,  8,  1);	
		
		RELU_TEST_1(relu_test_activate);
		RELU_TEST_1(relu_test_calc_grads);
		RELU_TEST_1(relu_test_fix_weights);
		RELU_TEST_1(relu_test);
		
	}
	TEST_F(OptimizationTests, level_2_relu) {
#define RELU_TEST_2(method)						\
		method<opt_relu_layer_t>(4,  6,  6,  1, 1);\
		method<opt_relu_layer_t>(12, 12, 3,  6, 1);\
		method<opt_relu_layer_t>(16, 96, 2,  14,1);
		
		RELU_TEST_2(relu_test_activate);
		RELU_TEST_2(relu_test_calc_grads);
		RELU_TEST_2(relu_test_fix_weights);
		RELU_TEST_2(relu_test);
	}

	TEST_F(OptimizationTests, level_3_relu) {
#define RELU_TEST_3(method)						\
		method<opt_relu_layer_t>(3,  7,  13, 3, 1); \
		method<opt_relu_layer_t>(5,  9,  17, 5, 1); \
		method<opt_relu_layer_t>(89, 31, 7,  7, 1);

		RELU_TEST_3(relu_test_activate);
		RELU_TEST_3(relu_test_calc_grads);
		RELU_TEST_3(relu_test_fix_weights);
		RELU_TEST_3(relu_test);
		
	}

	TEST_F(OptimizationTests, level_4_relu) {
		for (int i = 0; i < 10; i++) {
			srand(i);
			int x = RAND_LARGE(16);
			int y = RAND_LARGE(24);
			int z = RAND_LARGE(24);
			int b = RAND_LARGE(16);
			
			relu_test_activate<opt_relu_layer_t>(x,y,z,b,i);
			relu_test_calc_grads<opt_relu_layer_t>(x,y,z,b,i);
			relu_test_fix_weights<opt_relu_layer_t>(x,y,z,b,i);
			relu_test<opt_relu_layer_t>(x,y,z,b,i);

		}
		
	}

	class LabTests :  public ::testing::Test {
	};
	
#if(1)
	TEST_F(LabTests, test_lab_model) {
		int i = 3;
		conv_test<opt_conv_layer_t>(112, 112, 3, 4, 4, 11, 96, 0, i);
		conv_test<opt_conv_layer_t>(14, 14, 96, 4, 1, 5, 256, 2, i);
		conv_test<opt_conv_layer_t>(7, 7, 256, 4, 1, 3, 384, 1, i);
		conv_test<opt_conv_layer_t>(7, 7, 384, 4, 1, 3, 256, 1, i);
		conv_test<opt_conv_layer_t>(7, 7, 384, 4, 1, 3, 384, 1, i);
		fc_test<opt_fc_layer_t>(4, 4, 256, 4, 4096, i);
		fc_test<opt_fc_layer_t>(4096, 1, 1, 4, 1000, i);
		fc_test<opt_fc_layer_t>(4096, 1, 1, 4, 4096, i);
		pool_test<opt_pool_layer_t>(14, 14, 256, 4, 2, 3, 0, i);
		pool_test<opt_pool_layer_t>(28, 28, 96, 4, 2, 3, 0, i);
		pool_test<opt_pool_layer_t>(7, 7, 256, 4, 2, 3, 0, i);
		relu_test<opt_relu_layer_t>(14, 14, 96, 4, i);
		relu_test<opt_relu_layer_t>(4, 4, 256, 4, i);
		relu_test<opt_relu_layer_t>(7, 7, 256, 4, i);
		relu_test<opt_relu_layer_t>(7, 7, 384, 4, i);
	}
#endif		


}

int main(int argc, char **argv) {
	if (argc >= 2) {
		if (!strcmp(argv[1], "--print-deltas")) {
			tensor_t<double>::diff_prints_deltas = true;
			argc--;
			argv++;
		}
	}
	::testing::InitGoogleTest(&argc, argv);

	std::vector<int> omp_threads_values;
	std::vector<int> default_omp_threads_values;
	default_omp_threads_values.push_back(1);
	archlab_add_multi_option<std::vector<int> >("threads",
					      omp_threads_values,
					      default_omp_threads_values,
					      "1",
					      "How many threads use.  Pass multiple values to run with multiple thread counts.");
	

	int s = 0;
	std::vector<int> param1_values;
	std::vector<int> default_param1_values;
	default_param1_values.push_back(1);
	archlab_add_multi_option<std::vector<int> >("param1",
					      param1_values,
					      default_param1_values,
					      "1",
					      "Parameter 1 value.  Pass multiple values to run with multiple values.");
	std::string param1_name;
	archlab_add_option<std::string>("param1-name",
					param1_name,
					"param1",
					"Name for param 1");

	std::vector<int> param2_values;
	std::vector<int> default_param2_values;
	default_param2_values.push_back(1);
	archlab_add_multi_option<std::vector<int> >("param2",
					      param2_values,
					      default_param2_values,
					      "1",
					      "Parameter 2 value.  Pass multiple values to run with multiple values.");
	std::string param2_name;
	archlab_add_option<std::string>("param2-name",
					param2_name,
					"param2",
					"Name for param 2");
	
	std::vector<int> param3_values;
	std::vector<int> default_param3_values;
	default_param3_values.push_back(1);
	archlab_add_multi_option<std::vector<int> >("param3",
					      param3_values,
					      default_param3_values,
					      "1",
					      "Parameter 3 value.  Pass multiple values to run with multiple values.");
	std::string param3_name;
	archlab_add_option<std::string>("param3-name",
					param3_name,
					"param3",
					"Name for param 3");

	std::vector<int> param4_values;
	std::vector<int> default_param4_values;
	default_param4_values.push_back(1);
	archlab_add_multi_option<std::vector<int> >("param4",
					      param4_values,
					      default_param4_values,
					      "1",
					      "Parameter 4 value.  Pass multiple values to run with multiple values.");
	std::string param4_name;
	archlab_add_option<std::string>("param4-name",
					param4_name,
					"param4",
					"Name for param 4");

	archlab_parse_cmd_line(&argc, argv);

	for(auto & thread_count: omp_threads_values ) {
		theDataCollector->register_tag("omp_threads", thread_count);
		g_thread_count = thread_count;
		omp_set_num_threads(thread_count);
		std::cout << "Setting threadcount to " << thread_count <<"\n";	for(auto & param1_value : param1_values ) {
			g_param1_value = param1_value;
			theDataCollector->register_tag(param1_name, g_param1_value);
			std::cout << "Setting param1_value (" << param1_name << ") = " << g_param1_value <<"\n";
			for(auto & param2_value : param2_values ) {
				g_param2_value = param2_value;
				theDataCollector->register_tag(param2_name, g_param2_value);
				std::cout << "Setting param2_value (" << param2_name << ") = " << g_param2_value <<"\n";
				for(auto & param3_value : param3_values ) {
					g_param3_value = param3_value;
					theDataCollector->register_tag(param3_name, g_param3_value);
					std::cout << "Setting param3_value (" << param3_name << ") = " << g_param3_value <<"\n";
					for(auto & param4_value : param4_values ) {
						g_param4_value = param4_value;
						theDataCollector->register_tag(param4_name, g_param4_value);
						std::cout << "Setting param4_value (" << param4_name << ") = " << g_param4_value <<"\n";
						
						int r  = RUN_ALL_TESTS();
						if (r != 0) {
							std::cout << "Tests failed for this set of parameters:\n";
							std::cout << param1_name << " = " << g_param1_value << "\n";
							std::cout << param2_name << " = " << g_param2_value << "\n";
							std::cout << param3_name << " = " << g_param3_value << "\n";
							std::cout << param4_name << " = " << g_param4_value << "\n";
							std::cout << "thread count" << " = " << thread_count << "\n";
							return r;
						}
					}
				}
			}
		}
	}
	return s;
}
