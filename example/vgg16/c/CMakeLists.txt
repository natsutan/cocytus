cmake_minimum_required(VERSION 2.8)
#project (cqt_vgg16)
enable_language(C)

add_definitions("-Wall -g -Iinc")
include_directories(inc)


add_executable(cqt_vgg16
        cqt_vgg16.c
        cqt_gen/cqt_gen.c
        cqt_lib/numpy.c
        cqt_lib/cqt_lib.c
        cqt_lib/InputLayer.c
        cqt_lib/Conv2d_same_3x3.c
        cqt_lib/MaxPooling2D.c
        cqt_lib/Flatten.c
        cqt_lib/Dense.c
        )

target_link_libraries(cqt_vgg16 m)
