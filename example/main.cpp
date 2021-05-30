#include "archlab.hpp"
#include <cstdlib>
#include <getopt.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include "CNN/canela.hpp"
#include "math.h"
#include <omp.h>

using namespace std;

void stabilize(const std::string & version, const dataset_t & dataset, int frames, int tile_size, int thread_count);

int main(int argc, char *argv[])
{
	std::vector<std::string> dataset_s;
	std::vector<std::string> default_set;
	default_set.push_back("mnist");
	uint32_t frames;

	archlab_add_option<uint32_t>("frames",  frames   , 64  ,  "images to process");

	archlab_add_option<std::vector<std::string> >("dataset",
						      dataset_s,
						      default_set,
						      "mnist",
						      "Which dataset to use: 'mnist', 'emnist', 'cifar10', 'cifar100', or 'imagenet'. "
						      "Pass it multiple times to run multiple datasets.");

	std::vector<std::string> impls;
	std::vector<std::string> default_impls;
	default_impls.push_back("baseline");
	archlab_add_multi_option<std::vector<std::string>>("impl",
							   impls,
							   default_impls,
							   "baseline",
							   "Which versions to run");

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

	archlab_parse_cmd_line(&argc, argv);


	for(auto & ds: dataset_s) {
		std::cout << "Running " << ds << "\n";
		
		dataset_t *test = new dataset_t;
	
		if (ds == "mnist") {
			*test = dataset_t::read(std::string(std::getenv("CANELA_ROOT")) + "/datasets/mnist/mnist-test.dataset", 4);
		} else if (ds == "emnist") {
			*test = dataset_t::read(std::string(std::getenv("CANELA_ROOT")) + "/datasets/mnist/emnist-byclass-test.dataset", 4);
		} else if (ds == "cifar") {
			*test = dataset_t::read(std::string(std::getenv("CANELA_ROOT")) + "/datasets/cifar/cifar100_train.dataset", 4);
		} else if (ds == "imagenet") {
			*test = dataset_t::read(std::string(std::getenv("CANELA_ROOT")) + "/datasets/imagenet/imagenet.dataset", 4);
		} else {
			std::cerr << "unknown (Or incompatible) data set: " << ds << "\n";
			exit(1);
		}
		for(auto & tile_size: tile_sizes ) {
			for(auto & impl : impls) {
				for(auto & thread_count: omp_threads_values ) {
					theDataCollector->register_tag("omp_threads", thread_count);
					omp_set_num_threads(thread_count);
					
					std::cout << "Setting threadcount to " << thread_count <<"\n";
					stabilize(impl, *test, frames, tile_size, thread_count);
					
				}
			}
		}
	}
	
	archlab_write_stats();
	return 0;
}

