#include <string>
#include "parameters.hpp"
#include "archlab.hpp"
#include <cstdlib>
#include <getopt.h>
#include <stdio.h>
#include <unistd.h>
#include "CNN/canela.hpp"
#include "model.hpp"
#include <algorithm>
#include "reps.hpp"
#include"canary.hpp"
#include"omp.h"
using namespace std;

int main(int argc, char *argv[])
{
	int batch_size = 4;
	int scale_factor = 0;
	bool do_run_canary = false;
	bool describe_model;
	//bool estimate_runtime;

	
	std::vector<int> mhz_s;
	std::vector<int> default_mhz;
	load_frequencies();
	default_mhz.push_back(cpu_frequencies_array[1]);
	std::stringstream clocks;
	for(int i =0; cpu_frequencies_array[i] != 0; i++) {
		clocks << cpu_frequencies_array[i] << " ";
	}
	initiailize_rep_map();
	std::stringstream fastest;

	archlab_add_flag("run-canary", do_run_canary, false ,  "Run the canary.");
	
	fastest << cpu_frequencies_array[0];
	archlab_add_flag("describe-model", describe_model, false ,  "Print out information about the model and exit.");
	//archlab_add_flag("estimate-runtime", estimate_runtime, false ,  "Print out a very rough estimate of long all these tests will run and exit.");
	
	archlab_add_option<int>("scale", scale_factor, 4, "The scale factor.  Bigger (smaller) numbers mean longer (shorter) run times by running more samples.  It is supposed to roughly set the runtime of the test in seconds.  But it's based on a decent implementation.  If you want to run without opts, turn it down.  For Moneta tracing, set it to zero to run the code only once.");
	archlab_add_option<int>("batch-size", batch_size, 4, "Set the batch size.  How many images to process at once");
	archlab_add_option<std::vector<int> >("MHz",
					      mhz_s,
					      default_mhz,
					      fastest.str(),
					      "Which clock rate to run.  Possibilities on this machine are: " + clocks.str());
	std::vector<int> omp_threads_values;
	std::vector<int> default_omp_threads_values;
	default_omp_threads_values.push_back(1);
	archlab_add_multi_option<std::vector<int> >("threads",
					      omp_threads_values,
					      default_omp_threads_values,
					      "1",
					      "How many threads use.  Pass multiple values to run with multiple thread counts.");
	
	std::vector<std::string> dataset_s;
	std::vector<std::string> default_set;
	default_set.push_back("mininet");
	archlab_add_multi_option<std::vector<std::string> >("dataset",
						      dataset_s,
						      default_set,
						      "mininet",
						      "Which dataset to use: 'mnist', 'emnist', 'cifar10', 'cifar100', 'mininet, and 'imagenet'. "
						      "Pass it multiple times to run multiple datasets.");

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

	std::vector<int> layers_to_test;
	std::vector<int> default_layers_to_test;
	layers_to_test.push_back(-1);
	archlab_add_multi_option<std::vector<int> >("test-layer",
					      layers_to_test,
					      default_layers_to_test,
					      "0",
					      "Which layers to test.  Pass an integer corresponding to the layers order.  Pass multiple values to run functions from multiple layers.");


	std::vector<std::string> function_s;
	std::vector<std::string> default_functions;
	default_functions.push_back("train_model");
	archlab_add_multi_option<std::vector<std::string> >("function",
						      function_s,
						      default_functions,
						      "train_model",
						      "Which function to use: 'examples', 'train_model','activate', 'calc_grads', and 'fix_weights'.  Pass multiple values to run multiple functions.  For 'calc_grads' 'fix_weights' and 'activate', it will test it on all the layers specified with '--test-layer'");

       

	archlab_parse_cmd_line(&argc, argv);

	
	int job_count =
		param1_values.size() *
		param2_values.size() *
		param3_values.size() *
		param4_values.size() *
		layers_to_test.size() *
		omp_threads_values.size() *
		function_s.size();
	
	if (job_count > 144 || (job_count > 24 && scale_factor != 0) || (job_count > 10 && scale_factor > 4)) {
		std::cerr << "To limit wait times on the cluster, you cannot run this many experiments at once with '--scale " << scale_factor << "'.  You are trying to run " << job_count << ".  Break up the jobs into smaller groups and then use `merge-csv` to combine the results into one file.  The lab write up lists the size limits.\n";
		exit(1);
	} 
	
	for(auto & ds: dataset_s) {
		std::cout << "Running " << ds << "\n";
		
		dataset_t *train = new dataset_t;

		std::string path;

		// The code in train_model only runs one batch, over and over,
		// so there's no point in loading more than that.  In addition
		// the calculations around scale_factor in the per function
		// tests are based on a particular input data set.  We really
		// shouldn't be running anything but what we set those values
		// for.
		if (ds == "mnist") {
			path ="/datasets/mnist/mnist-train.dataset";
		} else if (ds == "emnist") {
			path ="/datasets/mnist/emnist-byclass-train.dataset";
		} else if (ds == "cifar10") {
			path ="/datasets/cifar/cifar10_data_batch_1.dataset";
		} else if (ds == "cifar100") {
			path ="/datasets/cifar/cifar100_train.dataset";
		} else if (ds == "imagenet") {
			path ="/datasets/imagenet/imagenet.dataset";
		} else if (ds == "mininet") {
			path ="/datasets/imagenet/mininet.dataset";
		} else {
			std::cerr << "Unknown dataset: " << ds << "\n";
			exit(1);
		}
	
	
		*train = dataset_t::read( std::string(std::getenv("CANELA_ROOT")) + path, batch_size);
		
		theDataCollector->register_tag("dataset", ds);
		theDataCollector->register_tag("training_inputs_count", train->test_cases.size());

		int clock_rate = mhz_s[0];

		model_t * model = build_model(*train);
		model->change_batch_size(batch_size);
		if (layers_to_test[0] == -1) {
			layers_to_test.clear();
			for (uint i = 0; i <  model->layers.size(); i++) {
				layers_to_test.push_back(i);
			}
		}

		std::cout << model->geometry() << "\n"; // output a summary of its sturcture and size.
		std::cout << "Regression parameters\n" <<model->regression_code() << std::endl;

		if (describe_model) {
			exit(0);
		}

	
		if (do_run_canary) {
			run_canary(clock_rate);
		}

		for(auto & thread_count: omp_threads_values ) {
			theDataCollector->register_tag("omp_threads", thread_count);
			omp_set_num_threads(thread_count);
			std::cout << "Setting threadcount to " << thread_count <<"\n";
			for(auto & param1_value : param1_values ) {
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
							for(auto & f: function_s) {
								int ran = 0;

								std::cout << "Running function = " << f <<"\n";
								if (f == "train_model"|| f == "all") {
									int reps = calculate_reps (scale_factor, 0, "train_model");
									g_function_name = "train_model";
									std::cout << "Running train_model " << reps << " times.\n";
									train_model(model, *train, reps, batch_size); // has it's own timing code
									ran++;
								}
			
								for (int layer_index: layers_to_test) {
									if (layer_index < 0 || layer_index >= (int)model->layers.size()) {
										std::cerr << "Illegal layer index: " << layer_index << "\n";
										exit(1);
									}

									std::vector<std::string> functions = {"activate", "calc_grads", "fix_weights"};

									for (auto & fname: functions) {
										if (f == fname || f == "all") {
											std::stringstream label;
											g_function_name = fname;
											auto l = model->layers[layer_index];
											int reps = calculate_reps(scale_factor, layer_index, fname);
											label << "layer[" << layer_index << "]->" << fname;
											std::cout << "Running " << label.str() << " " << reps << " times.\n";
											label << "  " << l->kind_str();
											pristine_machine();
											set_cpu_clock_frequency(clock_rate);
											theDataCollector->disable_prefetcher();
											{
												ArchLabTimer timer;
												timer.attr("reps", reps);
												timer.attr("layer_type", l->kind_str());
												timer.attr("full_name", label.str());
												timer.attr("layer", layer_index);
												timer.attr("function", fname);
												tensor_t<double> _in(l->in.size);
												tensor_t<double> _out(l->out.size);
												timer.go();
												if (fname == "activate") {
													for (int i = 0; i < reps; i++)
														l->activate(_in);
												} else if (fname == "calc_grads") {
													for (int i = 0; i < reps; i++)
														l->calc_grads(_out);
												} else if (fname == "fix_weights") {
													for (int i = 0; i < reps; i++)
														l->fix_weights();
												}
												ran++;
											}
										}
									}
								}
								if (ran == 0) {
									std::cerr << "unknown functions: " << f << "\n";
									exit(1);
								}
							}
						}
					}
				}
			}
		}
		delete train;
	}
	archlab_write_stats();
	return 0;
}
