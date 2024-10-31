#!/usr/bin/env bash

# copied from <https://github.com/olilarkin/ort-builder>

ONNX_CONFIG="${1:-./ort-model/model.required_operators_and_types.config}"
CMAKE_BUILD_TYPE=MinSizeRel

build_arch() {
  ONNX_CONFIG="$1"
  ARCH="$2"

  python ./vendor/onnxruntime/tools/ci_build/build.py \
  --build_dir "./build/build-ort-linux" \
  --config=${CMAKE_BUILD_TYPE} \
  --build_shared_lib \
  --parallel \
  --compile_no_warning_as_error \
  --skip_tests \
  --minimal_build \
  --disable_ml_ops \
  --use_preinstalled_eigen \
  --eigen_path=$(realpath "./vendor/eigen") \
  --disable_rtti \
  --disable_exceptions \
  --include_ops_by_config "$ONNX_CONFIG" \
  --enable_reduced_operator_type_support

  BUILD_DIR=./build-ort/${CMAKE_BUILD_TYPE}
}

build_arch "$ONNX_CONFIG" x86_64
