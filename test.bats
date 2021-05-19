#!/usr/bin/env bats
#-*- shell-script -*-

@test "solution code devel" {
    make clean
    make DEVEL_MODE=yes LAB_SUBMISSION_DIR=solution
    [ -e code.csv ]
    [ -e benchmark.csv ]
}

@test "starter code devel" {
    make clean
    make DEVEL_MODE=yes 
    [ -e code.csv ]
    [ -e benchmark.csv ]
}

@test "starter code" {
    archlab_check --engine papi || skip
    make clean
    make 
    [ -e code.csv ]
    [ -e benchmark.csv ]
}
