cmake_minimum_required(VERSION 3.0)

project(basicpitch.cpp)
enable_testing()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math -flto -fno-signed-zeros -fassociative-math -freciprocal-math -fno-math-errno -fno-rounding-math -funsafe-math-optimizations -fno-trapping-math -fno-rtti -DNDEBUG")

# Set some libremidi flags for wasm and header-only
add_definitions(-DLIBREMIDI_HEADER_ONLY=1)
#add_definitions(-DORT_NO_EXCEPTIONS=1)

# Define the path to the compiled ONNX Runtime static library
set(ONNX_RUNTIME_LIB ${CMAKE_SOURCE_DIR}/../build/build-ort-linux/MinSizeRel/libonnxruntime.so)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../vendor/onnxruntime/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../ort-model/model)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../vendor/eigen)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../vendor/oboe-resampler)

file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/../src/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/../src_cli/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/../ort-model/model/model.ort.c" "${CMAKE_CURRENT_SOURCE_DIR}/../vendor/oboe-resampler/*.cpp")
add_executable(basicpitch ${SOURCES})

# we only need header mode for libremidi
# use target_include_directories to treat it like a system library to  ignore warnings
target_include_directories(basicpitch SYSTEM PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../vendor/libremidi/include)

target_link_libraries(basicpitch ${ONNX_RUNTIME_LIB} libnyquist)
target_compile_definitions(basicpitch PRIVATE LIBREMIDI_HEADER_ONLY=1)

file(GLOB SOURCES_TO_LINT "${CMAKE_CURRENT_SOURCE_DIR}/../src/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/../src/*.hpp" "${CMAKE_CURRENT_SOURCE_DIR}/../src_wasm/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/../src_cli/*.cpp")

# add target to run standard lints and formatters
add_custom_target(lint
    COMMAND clang-format -i ${SOURCES_TO_LINT} --style=file
    # add clang-tidy command
    # add include dirs to clang-tidy
    COMMAND cppcheck -I"src/" -I"cli-apps/" --enable=all --suppress=missingIncludeSystem ${SOURCES_TO_LINT} --std=c++17
    COMMAND scan-build -o ${CMAKE_BINARY_DIR}/scan-build-report make -C ${CMAKE_BINARY_DIR}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
