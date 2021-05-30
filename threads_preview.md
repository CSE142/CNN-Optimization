# Final Project Part 2: Threads (PREVIEW)

---
# THIS IS A PREVIEW.  IT MIGHT CHANGE.  SO YOU MIGHT HAVE TO REDO IT.
---

But if you want to get started on Part 2 this weekend, you can.
---


You should have completed Part 1 before starting on Part 2.

This portion of the project is about multi-threading.  We will be using [OpenMP](https://en.wikipedia.org/wiki/OpenMP) for threading.  It's great for parallizing loops.

## New Command `--run-by-proxy`

Introduce a brand new option for `runlab`:  `--run-by-proxy`!

What does it do?  Well, it's just like `--run-git-remotely` except you don't have to commit your code!  It just pulls it out of your local directory.

You should consider this a "beta" feature.  If it causes problems, I'll have to turn it off.

## Turning on OpenMP

To turn on OpenMP, you need to set 

```
OPENMP=yes
```

in `config.env`, if it's not set already.  This sets some compiler flags to enable it.

OpenMP uses a thread pool -- a collection of theads that sit around waiting for work to do.  You can set the size of the pool with `--threads` command line argument.  Add it to the `IMPL_SEL_ARGS` because we want to test the code with multiple threads:

```
IMPL_SEL_ARGS=--threads 8        
```

8 is a reasonable choice because there are 8 thread contexts spread across 4 physical cores in our system.  It's conceivable that 4 might give better performance (for reasons you'll hear about in 142 when you discuss SMT).

## Threads Make Everything Harder

Threads and concurrency make programming and program behavior much more complex.  We could easily spend a whole quarter (or year) studying it.  We are using OpenMP because it simplifies things significantly and it's a good match for CNNs.  However, the tools we are using all have some limitations with respect to threads.

![Nondeterminism, mother of bugs, and queen of the seven cores](img/nondeterminism.jpg)

### Non-Determinism

Multithreading introduces the possibility for a new and really annoying bugs.  If you fail to coordinate your threads correctly, it can give rise to non-deterministic bugs that only occur sporadically.

This means that your program can run correctly one time and incorrectly next.  If this occurs, you are not losing your mind (although it may feel that way).

These intermittent failures are often always due to two threads sharing data without using a lock to ensure that only one thread is accessing the data at one time.

### Many Tools Don't Fully Support Threads

First,  `gprof` doesn't work on with multi-threaded programs. 

Second, Moneta's cache model is not multithread-aware, so the color coding for the traces for hits vs. misses and the hit and miss rates are not accurate.  

Moneta will also show more threads that you might be expecting.  OpenMP threads seem to be Thread 0 and 13 and above.  If you see some threads that don't seem to be doing anything, that's not surprising or concerning.

Finally, our performance counting code only collects data for one thread.  For OpenMP code this is ok:  all the threads do basically the same thing.  But you'll notice, for instance, that if a loop runs in 4 threads, the measure instruction count will go down by ~1/4 (assuming multi-threading didn't add a lot of overhead).

## Amdahl's Law for Multi-threaded Code

Gprof's non-support of multi-threaded code presents a particular problem:  You need to know where your code is spending the most time, so you can focus your optimization efforts.  How else will Amdahl be appeased?

For this project, we can get a decent approximation by running running each function in each layer once and adding up the execution time.  This number should approximate the total runtime of `train_model`, and you can use the latencies of each function to guage how important it is.  To make this work, you'll need to pass `--scale 0`, so each function runs exactly once.

## Tasks To Perform

### Parallelize fc_layer_t::calc_grads()

To get you started, we will walk you through the process for parallelizing `fc_layer_t::calc_grads()`.  Please see the slides in the lab repo for more details.  They contain detailed description of how the code works.

Unlike the previous labs, we have provided a baseline implementation  in `opt_cnn.hpp`.

Run `runlab --no-validate`. It should finish and in the output, you'll see

```
...
[  PASSED  ] 21 tests.
```

Which means that your implementation matches the result of the
baseline (which is no surprise because you have not edited baseline).

These tests are your best friend, since they provide a quick and easy
way of telling whether your code is correct.  `runlab` runs the tests
every time, and if you the last line shows any failures, you should
look at `regressions.out` for a full report.

First, replicate the structure in place for `activate()` to let you select among several implementations.  Make two copies of the starter code.  One you'll leave alone.  The other you'll optimize.

Set it up so that `--param1 1` will run the baseline version and `--param 2` will run the new version.

Then change `OMP_NUM_THREADS` to 2 in `config.env`. 

#### Version 1: `nn`-Loop

Modify the code to add multithreading to the `nn` loop and run it again. You can do this by adding `#pragma omp parallel for` on the line before the `nn` for loop.  When your code finishes running, you will notice that you failed multiple regression tests.  This is because by parallelizing the `nn` loop, multiple threads attempt to write to the same location in `grads_out`.

We will fix this in two stages:

**Stage 1** Add a local tensor the same size as `grads_out` at the top of the `nn` for loop. You can see an example of this in `example/stabilize.cpp` line 338. The example creates a tensor of type double with the same size as `output`. You will do the same except the tensor size will be the same size as `grads_out`. Do not forget to clear it just like the example does on line 339. Then, in the inner most loop (the `i` loop), change `grads_out` to be your new local tensor that each thread will create.

This enables each thread to accumulate their results locally. Thereby eliminating the race condition causing errors. However, we now need to combine the results from each thread.

**Stage 2** At the bottom of the `nn` for loop, add a critical section and create two nested for loops to loop through `out.size.b` and `grads_out.size.x`. Notice that we don't loop through `out.size.x` as well. This is because we only need to accumulate the results into `grads_out` and `n` is not used to index into `grads_out`. 

Inside the nested for loop you just created, accumulate the results of each thread (stored in their local tensors you creaded in stage 1) into `grads_out`. This will look very similar to `example/stabilize.cpp` lines 361 - 369. 

Once you have made your changes, run the code locally and verify that you pass all 21 regression tests. If you do not pass, refer back to the lecture slides, discussion slides, example in `exmaple/stabilize.cpp` lines 330 - 372, and help from the staff during office hours or lab hours. Once you have verified that your code is correct and passes the regression tests, submit to the autograder. You will want to save the resulting benchmark.csv file for the worksheet.

#### Version 2: `b`-Loop

Make another copy of the starter code. Rename it, and add it to the `switch` statement so you can select with `--param1 3`.  We are going to apply a different optimization.

Modify the code to add multithreading to the `b` loop and run it again. You can do this by adding `#pragma omp parallel for` on the line before the `b` for loop. You will notice that you passed the regressions test! There is no need to fix any race condition here as each thread is accumulating its result into a different address of grads_out.

#### Version 3: `n`-Loop

Same thing again: make another copy of the baseline code.  This time make it selectable with `--param1 4`.

Modify the code to add multithreading to the `n` loop and run it again. You can do this by adding `#pragma omp parallel for` on the line before the `n` for loop. When your code finishes running, you will notice that you failed multiple regression tests. This is because by parallelizing the `n` loop, multiple threads attempt to write to the same location in `grads_out`.

We will fix this in two stages:

**Stage 1** Add a local tensor the same size as `grads_out` at the top of the `n` for loop. You can see an example of this in `exmaple/stabilize.cpp` line 338. The example creates a tensor of type double with the same size as `output`. You will do the same except the tensor size will be the same size as `grads_out`. Do not forget to clear it just like the example does on line 339. Then, in the inner most loop (the `i` loop), change `grads_out` to be your new local tensor that each thread will create.

This enables each thread to accumulate their results locally. Thereby eliminating the race condition causing errors. However, we now need to combine the results from each thread.

**Stage 2** At the bottom of the `n` for loop, add a critical section and create a for loops to loop through `grads_out.size.x`. Notice that we don't loop through `out.size.x` or `out.size.b` as well. We exclude `out.size.x` because we only need to accumulate the results into `grads_out` and `n` is not used to index into `grads_out`. We exclude `out.size.b` because we only need to accumulate the result that the threads were individually working on, and they were all already working on the same `b` because we multithreaded the `n` loop, which is inside the `b` loop. 

Inside the for loop you just created, accumulate the results of each thread (stored in their local tensors you creaded in stage 1) into `grads_out`. This will look very similar to `exmaple/stabilize.cpp` lines 361 - 369. 

#### Version 5: `i`-Loop

One more time.  This time use `--param1 5`.

Modify the code to add multithreading to the `i` loop and run it again. You can do this by adding `#pragma omp parallel for` on the line before the `i` for loop. You will notice that you passed the regressions test! There is no need to fix any race condition here as each thread is accumulating its result into a different address of grads_out.

#### Compare Performance

Let's see which loop we should parallelize.  You should be able to measure the performance of all five implementations with one call to `--run-git-remotely`.

In `config.env`, set 

```
IMPL_SEL_ARGS=--param1-name impl --param1 1 2 3 4 5 --threads 1
CMD_LINE_ARGS=--test-layer 14 --function calc_grads  --stat-set PE.cfg --stat misses=PAPI_L1_DCM --engine papi --calc TIC=IC*omp_threads --calc Tmisses=misses*omp_threads
```


This will run all 5 implementations of `calc_grads` with 1 thread on layer 14 (the largest `fc_layer_t`).

Starting with `--stat-set PE.cfg`, it also sets up some performance counters to measure the components of the performance equation and cache performance.  The data can be a little confusing, so here's what the columns mean.  The first 8 are our focus:

| Column name | Meaning                               | 
|-------------|---------------------------------------|
| misses	  | the number of L1 cache misses (for one thread)  |
| IC	      | Instruction Count (for one thread)    |
| TIC         | Total instructions (all threads), `IC * omp_thread`.  If you set `--threads` > 1 and don't use OpenMP, this will be inaccurate. |
| Tmisses     | Total missess (all threads).          |
| CT          | The actual cycle time (cycles/ET). This might not be 1/MHz due to power throttling for.     |
| CPI         | Cycles per Instructions               |
| ET          | Execution time                        |
| omp_threads | Number of threads                     |
| cycles      | Actual clock cycles (for one thread)  |
| runtime     | Same as ET                            |
| PAPI_REF_CYC| "Reference clock" cyles.  Ignore this.|
| MHz         | The clock rate we asked for.          |
| IPC         | 1/CPI                                 |


Commit everything.  Then do 

```
runlab --run-git-remotely -- make cnn.csv
```

and you should get something like this:

```
dataset|training_inputs_count|omp_threads|impl|param2|param3|param4|full_name                        |function  |layer|layer_type|reps|
-------|---------------------|-----------|----|------|------|------|---------------------------------|----------|-----|----------|----|
mininet|4.0                  |1.0        |1.0 |1.0   |1.0   |1.0   |layer[14]->calc_grads  fc_layer_t|calc_grads|14.0 |fc_layer_t|5.0 |
mininet|4.0                  |1.0        |2.0 |1.0   |1.0   |1.0   |layer[14]->calc_grads  fc_layer_t|calc_grads|14.0 |fc_layer_t|5.0 |
mininet|4.0                  |1.0        |3.0 |1.0   |1.0   |1.0   |layer[14]->calc_grads  fc_layer_t|calc_grads|14.0 |fc_layer_t|5.0 |
mininet|4.0                  |1.0        |4.0 |1.0   |1.0   |1.0   |layer[14]->calc_grads  fc_layer_t|calc_grads|14.0 |fc_layer_t|5.0 |
mininet|4.0                  |1.0        |5.0 |1.0   |1.0   |1.0   |layer[14]->calc_grads  fc_layer_t|calc_grads|14.0 |fc_layer_t|5.0 |
```

Isn't it glorious!  Just look at all that data! (the above omits the data from the performance counters.  Refer to your own data).


---
# THE QUESTIONS YOU WILL HAVE TO ANSWER FOR THIS LAB HAVE BEEN DELETED.  WE WILL RELEASE THE PDF SOON.
---

## Go forth and optimize!

There are bunch of loops in the code base just waiting to be parallelized.

Check `README.md` for performance targets for the end of this part of the lab.

Recall the note at the top of this file about Amdahl's Law, multi-threaded code, and how `gprof` can't handle threads.

## Tips

* There are many more things to try in this lab than there have been
  in the earlier labs.  This has two implication:

  * Start early.
 
  * "guess and check" is unlikely to get you a good solution in
    reasonable.  Think carefully about the changes you are making.
    Thinking takes time.  Start early.

  * The autograder servers are going to get very busy near the deadline.  Start early.

* Unfortunately, gprof doesn't work on multi-threaded programs.  You
  can comment out `OPENMP=yes` in `config.env` to make gprof work
  properly.

* OpenMP is a big library.  We've covered a bit of it.  You're free to use the rest.

* There's lot of resources on the web about OpenMP.  Many (or most) of
  them are bad.  This one is pretty good:
  http://jakascorner.com/blog/.  Especially these entries:

  * http://jakascorner.com/blog/2016/04/omp-introduction.html
  
  * http://jakascorner.com/blog/2016/05/omp-for.html
  
  * http://jakascorner.com/blog/2016/07/omp-critical.html

  * http://jakascorner.com/blog/2016/06/omp-for-scheduling.html

  * http://jakascorner.com/blog/2016/06/omp-data-sharing-attributes.html

  * http://jakascorner.com/blog/2016/07/omp-default-none-and-const.html

---
# YOU ARE NOT DONE.  THERE IS ANOTHER PART (`README3.pdf`)
---

  
