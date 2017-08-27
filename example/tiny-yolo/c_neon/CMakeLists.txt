cmake_minimum_required(VERSION 3.8)
project(c_neon)

add_definitions("-pg -Wall -g -Iinc -O3")
include_directories(inc)
SET(CMAKE_EXE_LINKER_FLAGS  "-pg")

set(SOURCE_FILES
        cqt_tyolo.c
        yad2k_yolo.c
        cqt_gen/cqt_debug.c
        cqt_gen/cqt_debug.h
        cqt_gen/cqt_gen.c
        cqt_gen/cqt_gen.h
        cqt_lib/BatchNormalization.c
        cqt_lib/Conv2d_same_1x1.c
        cqt_lib/Conv2d_same_3x3.c
        cqt_lib/cqt_lib.c
        cqt_lib/InputLayer.c
        cqt_lib/LeakyReLU.c
        cqt_lib/MaxPooling2D.c
        cqt_lib/numpy.c
        inc/cqt.h
        inc/cqt_keras.h
        inc/cqt_lib.h
        inc/cqt_net.h
        inc/cqt_type.h
        inc/numpy.h
        inc/SSE2NEON.h)


add_executable(c_neon ${SOURCE_FILES})
target_link_libraries(c_neon m)
