#include"reps.hpp"
#include<cmath>
typedef std::pair<int, std::string> rep_map_key;
std::map<rep_map_key,float> rep_map;


void initiailize_rep_map() {
	
#define REP_MAP_ENTRY(layer, function, REPS) do {			  \
		rep_map[rep_map_key(layer, function)] = REPS; \
		/*std::cerr << "Setting <" << layer << ", " << function << "> = " << (unity ? 1 : REPS) << "\n";*/ \
	} while(0)
	
	// THis is a table of how many iterations of each function
	// execute in 1 second for the unoptimized version.
	REP_MAP_ENTRY(0, "train_model",12.65);
	REP_MAP_ENTRY(0,"activate",0.1965);
	REP_MAP_ENTRY(0,"calc_grads",0.555);
	REP_MAP_ENTRY(0,"fix_weights",0.000345192307692308);
	REP_MAP_ENTRY(1,"activate",0.00120996441281139);
	REP_MAP_ENTRY(1,"calc_grads",0.00402877697841727);
	REP_MAP_ENTRY(1,"fix_weights",0.00000000172);
	REP_MAP_ENTRY(2,"activate",0.000157057057057057);
	REP_MAP_ENTRY(2,"calc_grads",0.000215);
	REP_MAP_ENTRY(2,"fix_weights",0.0000000016);
	REP_MAP_ENTRY(3,"activate",0.587);
	REP_MAP_ENTRY(3,"calc_grads",2.9);
	REP_MAP_ENTRY(3,"fix_weights",0.00957264957264957);
	REP_MAP_ENTRY(4,"activate",0.000797101449275362);
	REP_MAP_ENTRY(4,"calc_grads",0.00269396551724138);
	REP_MAP_ENTRY(4,"fix_weights",0.00000000329);
	REP_MAP_ENTRY(5,"activate",0.000101883561643836);
	REP_MAP_ENTRY(5,"calc_grads",0.000143);
	REP_MAP_ENTRY(5,"fix_weights",0.0000000016);
	REP_MAP_ENTRY(6,"activate",0.213666666666667);
	REP_MAP_ENTRY(6,"calc_grads",1.04666666666667);
	REP_MAP_ENTRY(6,"fix_weights",0.013452380952381);
	REP_MAP_ENTRY(7,"activate",0.000180901856763926);
	REP_MAP_ENTRY(7,"calc_grads",0.000244);
	REP_MAP_ENTRY(7,"fix_weights",0.00000000191);
	REP_MAP_ENTRY(8,"activate",0.325);
	REP_MAP_ENTRY(8,"calc_grads",1.585);
	REP_MAP_ENTRY(8,"fix_weights",0.0211538461538462);
	REP_MAP_ENTRY(9,"activate",0.00018283378746594);
	REP_MAP_ENTRY(9,"calc_grads",0.000253);
	REP_MAP_ENTRY(9,"fix_weights",0.00000000291);
	REP_MAP_ENTRY(10,"activate",0.213);
	REP_MAP_ENTRY(10,"calc_grads",1.01333333333333);
	REP_MAP_ENTRY(10,"fix_weights",0.01375);
	REP_MAP_ENTRY(11,"activate",0.000103225806451613);
	REP_MAP_ENTRY(11,"calc_grads",0.000123);
	REP_MAP_ENTRY(11,"fix_weights",0.00000000598);
	REP_MAP_ENTRY(12,"activate",0.000238188976377953);
	REP_MAP_ENTRY(12,"calc_grads",0.000564766839378238);
	REP_MAP_ENTRY(12,"fix_weights",0.00000000291);
	REP_MAP_ENTRY(13,"activate",0.0000334);
	REP_MAP_ENTRY(13,"calc_grads",0.0000302);
	REP_MAP_ENTRY(13,"fix_weights",0.00000000348);
	REP_MAP_ENTRY(14,"activate",0.88);
	REP_MAP_ENTRY(14,"calc_grads",0.845);
	REP_MAP_ENTRY(14,"fix_weights",0.1805);
	REP_MAP_ENTRY(15,"activate",0.85);
	REP_MAP_ENTRY(15,"calc_grads",0.835);
	REP_MAP_ENTRY(15,"fix_weights",0.1755);
	REP_MAP_ENTRY(16,"activate",0.0384);
	REP_MAP_ENTRY(16,"calc_grads",0.0406666666666667);
	REP_MAP_ENTRY(16,"fix_weights",0.0289230769230769);

}

int calculate_reps(int scale, int layer_index, const std::string &function) {
	// scale is roughly the number of seconds the function should run for.
	// For very small functions, calculated number of reps gets meaningless so bound it.
	if (scale == 0) {
		return 1;
	}
	int r = std::max(2.0,std::min(std::ceil((scale + 0.0) / rep_map[rep_map_key(layer_index, function)]), 10000.0));
	return r;
}

