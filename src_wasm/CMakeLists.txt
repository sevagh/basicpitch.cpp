cmake_minimum_required(VERSION 3.0)

project(basicpitch.cpp)
enable_testing()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -msimd128 -msse4.2 -flto -fno-exceptions -fno-math-errno -fno-trapping-math -fassociative-math -freciprocal-math -fno-rtti -DNDEBUG")

# Set some libremidi flags for wasm and header-only
add_definitions(-DLIBREMIDI_HEADER_ONLY=1 -DLIBREMIDI_EMSCRIPTEN=1)
add_definitions(-DORT_NO_EXCEPTIONS=1)

# Define the path to the compiled ONNX Runtime static library
set(ONNX_RUNTIME_WASM_LIB ${CMAKE_SOURCE_DIR}/../build/build-ort-wasm/MinSizeRel/libonnxruntime_webassembly.a)

if(NOT EMSCRIPTEN)
    message(FATAL_ERROR "Only Emscripten build target is supported")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../vendor/onnxruntime/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../ort-model/model)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../vendor/eigen)

set(COMMON_LINK_FLAGS "-flto -s ALLOW_MEMORY_GROWTH=1 -s MAXIMUM_MEMORY=4GB -s STACK_SIZE=5MB -s MODULARIZE=1 -s 'EXPORTED_RUNTIME_METHODS=[\"FS\"]'")

file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/../src/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/../src_wasm/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/../ort-model/model/model.ort.c")
add_executable(basicpitch ${SOURCES})

# we only need header mode for libremidi
# use target_include_directories to treat it like a system library to  ignore warnings
target_include_directories(basicpitch SYSTEM PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../vendor/libremidi/include)

target_link_libraries(basicpitch ${ONNX_RUNTIME_WASM_LIB})
target_compile_definitions(basicpitch PRIVATE LIBREMIDI_HEADER_ONLY=1 LIBREMIDI_EMSCRIPTEN=1)
set_target_properties(basicpitch PROPERTIES
    LINK_FLAGS "${COMMON_LINK_FLAGS} -s EXPORT_NAME='libbasicpitch' -s EXPORTED_RUNTIME_METHODS=['getValue'] -s EXPORTED_FUNCTIONS=\"['_malloc', '_free', '_convertToMidi']\""
)

# Custom command to copy the basicpitch.js and basicpitch.wasm files to the ./web directory
add_custom_command(TARGET basicpitch POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_BINARY_DIR}/basicpitch.js" "${CMAKE_SOURCE_DIR}/../web/"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_BINARY_DIR}/basicpitch.wasm" "${CMAKE_SOURCE_DIR}/../web/"
    COMMENT "Copying basicpitch.js and basicpitch.wasm to the web directory"
)
