#pragma once
#include "CNN/canela.hpp"
extern model_t * build_model(const dataset_t & ds);
extern void train_model(model_t * model,
		 dataset_t & train,
		 int reps,
		 int batch_size);

