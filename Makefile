default: benchmark.csv run_tests.exe regressions.out cnn.csv cnn.exe
OPTIMIZE=-march=x86-64 -O3
CLEANUP=trace_traceme.hdf5 trace_cnn.hdf5
COMPILER=gcc-8
include $(ARCHLAB_ROOT)/cse141.make
$(BUILD)model.s: $(BUILD)opt_cnn.hpp  $(BUILD)opt_cnn.hpp $(BUILD)canary.hpp $(BUILD)model.hpp $(BUILD)reps.hpp $(BUILD)parameters.hpp

MEMOPS?=10000000

ifeq ($(DEVEL_MODE),yes)
BENCHMARK_CMD_LINE=--stat runtime=ARCHLAB_WALL_TIME 
else
BENCHMARK_CMD_LINE=--stat-set L1.cfg
endif

FULL_CMD_LINE_ARGS=$(LAB_COMMAND_LINE_ARGS) $(CMD_LINE_ARGS) $(IMPL_SEL_ARGS)


cnn.exe:  $(BUILD)parameters.o  $(BUILD)model.o  $(BUILD)canary.o  $(BUILD)reps.o
$(BUILD)model.o:  $(BUILD)opt_cnn.hpp

$(BUILD)cnn.o: $(BUILD)opt_cnn.hpp  $(BUILD)opt_cnn.hpp $(BUILD)canary.hpp $(BUILD)model.hpp $(BUILD)reps.hpp $(BUILD)parameters.hpp

benchmark.csv: cnn.exe
	rm -f gmon.out
	./cnn.exe --run-canary --stats-file $@ --scale 4 --batch-size 4 --function train_model $(BENCHMARK_CMD_LINE)
	pretty-csv $@
	if [ -e gmon.out ]; then gprof $< > benchmark.gprof; fi

cnn.csv: cnn.exe
	rm -f gmon.out
	./cnn.exe --stats-file $@ $(FULL_CMD_LINE_ARGS)
	pretty-csv $@
	if [ -e gmon.out ]; then gprof $< > cnn.gprof; fi


custom.exe: $(BUILD)cnn.o $(BUILD)parameters.o  $(BUILD)model.o  $(BUILD)canary.o  $(BUILD)reps.o 
$(BUILD)custom.o: $(BUILD)opt_cnn.hpp $(BUILD)canary.hpp $(BUILD)model.hpp $(BUILD)reps.hpp $(BUILD)parameters.hpp

custom.csv: custom.exe
	rm -f gmon.out
	./custom.exe --stats-file $@ $(CUSTOM_CMD_LINE_ARGS)
	pretty-csv $@
	if [ -e gmon.out ]; then gprof $< > custom.gprof; fi

examples.csv: examples.exe
	./examples.exe --stats-file $@
	pretty-csv $@

run_tests.exe:  $(BUILD)parameters.o 
run_tests.o: $(BUILD)opt_cnn.hpp  $(BUILD)parameters.hpp

.PHONY: regressions.out
regressions.out regressions.json: ./run_tests.exe
	-./run_tests.exe --gtest_output=json:regressions.json > $@ 
	tail -1 $@


activate.mtrace: cnn.exe
activate.mtrace: CMD=./cnn.exe --param1-name impl --param1 0 1 --function activate --scale 0 

.PHONY: %.mtrace
%.mtrace: 
	mtrace --trace $* --memops 100000000 --file-count 10 -- $(CMD)
	merge-csv $**.stats.csv | pretty-csv -

-include solution/solution.make
