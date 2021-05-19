#include "CNN/canela.hpp"
#include "opt_cnn.hpp"
#include "archlab.hpp"

void train_model(model_t * model,
		 dataset_t & train,
		 int reps,
		 int batch_size) {
	int batch_index = 0;
	tensor_t<double> batch_data(tdsize(train.data_size.x, train.data_size.y, train.data_size.z, batch_size));
	tensor_t<double> batch_label(tdsize(train.label_size.x, train.label_size.y, train.label_size.z, batch_size));
	for (auto& t : train.test_cases ) {
		for (int x = 0; x < t.data.size.x; x += 1)
			for (int y = 0; y < t.data.size.y; y += 1)
				for (int z = 0; z < t.data.size.z; z += 1)
					batch_data(x, y, z, batch_index) = t.data(x, y, z); 		
		for (int x = 0; x < t.label.size.x; x += 1)
			for (int y = 0; y < t.label.size.y; y += 1)
				for (int z = 0; z < t.label.size.z; z += 1)
					batch_label(x, y, z, batch_index) = t.label(x, y, z); 		
		batch_index += 1;
		if (batch_index >= batch_size) {
			batch_index = 0;
			{
				ArchLabTimer timer; // create it.
				pristine_machine();
				theDataCollector->disable_prefetcher();
				timer.attr("function", "train_model");
				timer.attr("reps", reps);
				timer.go();
				for (int i = 0; i < reps; i += 1) {
					model->train(batch_data, batch_label);
				}
				return; // Run one batch, multiple times and the return.
			}
		}
	}
	
}

#if (0)
void train_model_full(model_t * model,
		 dataset_t & train,
		 int epochs,
		 int batch_size) {
	int batch_index = 0;
	tensor_t<double> batch_data(tdsize(train.data_size.x, train.data_size.y, train.data_size.z, batch_size));
	tensor_t<double> batch_label(tdsize(train.label_size.x, train.label_size.y, train.label_size.z, batch_size));
	for (int i = 0; i < epochs; i += 1) {
		for (auto& t : train.test_cases ) {
			for (int x = 0; x < t.data.size.x; x += 1)
				for (int y = 0; y < t.data.size.y; y += 1)
					for (int z = 0; z < t.data.size.z; z += 1)
						batch_data(x, y, z, batch_index) = t.data(x, y, z); 		
			for (int x = 0; x < t.label.size.x; x += 1)
				for (int y = 0; y < t.label.size.y; y += 1)
					for (int z = 0; z < t.label.size.z; z += 1)
						batch_label(x, y, z, batch_index) = t.label(x, y, z); 		
			batch_index += 1;
			if (batch_index >= batch_size) {
				batch_index = 0;
				model->train(batch_data, batch_label);
			}
		}
	}

}


double test_model(model_t * model,
	dataset_t & test,
	int batch_size) {
	int correct = 0;
	int incorrect = 0;
	int batch_index = 0;

	tensor_t<double> batch_data(tdsize(test.data_size.x, test.data_size.y, test.data_size.z, batch_size));
	tensor_t<double> batch_label(tdsize(test.label_size.x, test.label_size.y, test.label_size.z, batch_size));
	for (auto& t : test.test_cases ) {
		for (int x = 0; x < t.data.size.x; x += 1)
			for (int y = 0; y < t.data.size.y; y += 1)
				for (int z = 0; z < t.data.size.z; z += 1)
					batch_data(x, y, z, batch_index) = t.data(x, y, z); 		
		for (int x = 0; x < t.label.size.x; x += 1)
			for (int y = 0; y < t.label.size.y; y += 1)
				for (int z = 0; z < t.label.size.z; z += 1)
					batch_label(x, y, z, batch_index) = t.label(x, y, z); 		
		batch_index += 1;
		if (batch_index >= batch_size) {
			batch_index = 0;
			tensor_t<double>& out = model->apply(batch_data);
			std::vector<tdsize> maxes = out.argmax_b();
			std::vector<tdsize> correct_maxes = batch_label.argmax_b();
			for (int i = 0; i < batch_size; i += 1) {
				if (maxes[i].x == correct_maxes[i].x) {
					correct += 1;
				} else {
					incorrect += 1;
				}
			}

		}
	}
	double total_error = (correct+0.0)/(correct+ incorrect +0.0);
	return total_error;
}
#endif

model_t * build_model(const dataset_t & ds)  {

	// This is a rough approximation of alex net.
	
	model_t * model = new model_t();	

	conv_layer_t*    layer1   = new opt_conv_layer_t    ( 4, 11, 96, 0, ds.data_size);
	pool_layer_t*    layer2   = new opt_pool_layer_t    ( 2, 3, 0, layer1->out.size );	
	relu_layer_t*    layer3   = new opt_relu_layer_t    ( layer2->out.size );
			         		
	conv_layer_t*    layer4   = new opt_conv_layer_t    ( 1, 5, 256, 2, layer3->out.size );
	pool_layer_t*    layer5   = new opt_pool_layer_t    ( 2, 3, 0, layer4->out.size );	
	relu_layer_t*    layer6   = new opt_relu_layer_t    ( layer5->out.size );
			         		
	conv_layer_t*    layer7   = new opt_conv_layer_t    ( 1, 3, 384, 1, layer6->out.size );
	relu_layer_t*    layer8   = new opt_relu_layer_t    ( layer7->out.size );
			         
	conv_layer_t*    layer9   = new opt_conv_layer_t    ( 1, 3, 384, 1, layer8->out.size );
	relu_layer_t*    layer10  = new opt_relu_layer_t    ( layer9->out.size );
			         
	conv_layer_t*    layer9b  = new opt_conv_layer_t    ( 1, 3, 256, 1, layer10->out.size );
	relu_layer_t*    layer10b = new opt_relu_layer_t    ( layer9b->out.size );
			         		
	pool_layer_t*    layer11  = new opt_pool_layer_t    ( 2, 3, 0, layer10b->out.size );	
	relu_layer_t*    layer12  = new opt_relu_layer_t    ( layer11->out.size );
			         
	fc_layer_t*      layer13  = new opt_fc_layer_t        ( layer12->out.size, 4096 );
	//	dropout_layer_t *layer14  = new dropout_layer_t       (layer13->out.size, 0.5);
	fc_layer_t*      layer15  = new opt_fc_layer_t        ( layer13->out.size, 4096 );
	//	dropout_layer_t *layer16  = new dropout_layer_t       (layer15->out.size, 0.5);
	fc_layer_t*      layer17  = new opt_fc_layer_t        ( layer15->out.size, ds.label_size.x );
	//softmax_layer_t layer18(layer17.out.size);
	
	model->add_layer(*layer1 );
	model->add_layer(*layer2 );
	model->add_layer(*layer3 );
	model->add_layer(*layer4 );
	model->add_layer(*layer5 );
	model->add_layer(*layer6 );
	model->add_layer(*layer7 );
	model->add_layer(*layer8 );
	model->add_layer(*layer9 );
	model->add_layer(*layer10 );
	model->add_layer(*layer9b );
	model->add_layer(*layer10b);
	model->add_layer(*layer11 );
	model->add_layer(*layer12 );
	model->add_layer(*layer13 );
	//	model->add_layer(*layer14 );
	model->add_layer(*layer15 );
	//model->add_layer(*layer16 );
	model->add_layer(*layer17 );

	return model;
}
