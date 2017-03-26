cmake_minimum_required(VERSION 2.8)
#project (cqt_vgg16)

add_definitions("-Wall -g -Iinc")
include_directories(inc)

add_executable(cqt_vgg16
        cqt_vgg16.c
        cqt_gen/cqt_gen.c
        cqt_lib/numpy.c
        )
