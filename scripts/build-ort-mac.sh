#!/usr/bin/env bash
set -euf -o pipefail

ONNX_CONFIG="${1:-./ort-model/model.required_operators_and_types.config}"
CMAKE_BUILD_TYPE=MinSizeRel

build_arch() {
  ONNX_CONFIG="$1"
  ARCH="$2"
  BUILD_DIR=./build/build-ort-mac/$ARCH

  python ./vendor/onnxruntime/tools/ci_build/build.py \
  --build_dir "${BUILD_DIR}" \
  --config=${CMAKE_BUILD_TYPE} \
  --parallel \
  --minimal_build \
  --apple_deploy_target="10.15" \
  --disable_ml_ops --disable_rtti \
  --include_ops_by_config "$ONNX_CONFIG" \
  --enable_reduced_operator_type_support \
  --cmake_extra_defines CMAKE_OSX_ARCHITECTURES="${ARCH}" \
  --cmake_extra_defines "EIGEN_NO_AUTOMATIC_RESIZING=1;EIGEN_MPL2_ONLY=1" \
  --skip_tests

  libtool -static -o "onnxruntime-macOS_${ARCH}-static-combined.a" \
  "${BUILD_DIR}/${CMAKE_BUILD_TYPE}/libonnx.a" \
  "${BUILD_DIR}/${CMAKE_BUILD_TYPE}/libonnxruntime_graph.a" \
  "${BUILD_DIR}/${CMAKE_BUILD_TYPE}/libonnx_proto.a" \
  "${BUILD_DIR}/${CMAKE_BUILD_TYPE}/libonnxruntime_mlas.a" \
  "${BUILD_DIR}/${CMAKE_BUILD_TYPE}/libonnx_test_data_proto.a" \
  "${BUILD_DIR}/${CMAKE_BUILD_TYPE}/libonnxruntime_optimizer.a" \
  "${BUILD_DIR}/${CMAKE_BUILD_TYPE}/libonnx_test_runner_common.a" \
  "${BUILD_DIR}/${CMAKE_BUILD_TYPE}/libonnxruntime_common.a" \
  "${BUILD_DIR}/${CMAKE_BUILD_TYPE}/libonnxruntime_providers.a" \
  "${BUILD_DIR}/${CMAKE_BUILD_TYPE}/libonnxruntime_session.a" \
  "${BUILD_DIR}/${CMAKE_BUILD_TYPE}/libonnxruntime_flatbuffers.a" \
  "${BUILD_DIR}/${CMAKE_BUILD_TYPE}/libonnxruntime_test_utils.a" \
  "${BUILD_DIR}/${CMAKE_BUILD_TYPE}/libonnxruntime_framework.a" \
  "${BUILD_DIR}/${CMAKE_BUILD_TYPE}/libonnxruntime_util.a" \
  "${BUILD_DIR}/${CMAKE_BUILD_TYPE}/_deps/re2-build/libre2.a" \
  "${BUILD_DIR}/${CMAKE_BUILD_TYPE}/_deps/google_nsync-build/libnsync_cpp.a" \
  "${BUILD_DIR}/${CMAKE_BUILD_TYPE}/_deps/protobuf-build/libprotobuf-lite.a" \
  "${BUILD_DIR}/${CMAKE_BUILD_TYPE}/_deps/abseil_cpp-build/absl/hash/libabsl_hash.a" \
  "${BUILD_DIR}/${CMAKE_BUILD_TYPE}/_deps/abseil_cpp-build/absl/hash/libabsl_city.a" \
  "${BUILD_DIR}/${CMAKE_BUILD_TYPE}/_deps/abseil_cpp-build/absl/hash/libabsl_low_level_hash.a" \
  "${BUILD_DIR}/${CMAKE_BUILD_TYPE}/_deps/abseil_cpp-build/absl/base/libabsl_throw_delegate.a" \
  "${BUILD_DIR}/${CMAKE_BUILD_TYPE}/_deps/abseil_cpp-build/absl/container/libabsl_raw_hash_set.a" \
  "${BUILD_DIR}/${CMAKE_BUILD_TYPE}/_deps/abseil_cpp-build/absl/base/libabsl_raw_logging_internal.a"
}

build_arch "$ONNX_CONFIG" x86_64
build_arch "$ONNX_CONFIG" arm64

OUTPUT_DIR=./build/build-ort-mac/universal/${CMAKE_BUILD_TYPE}

mkdir -p ${OUTPUT_DIR}
lipo -create onnxruntime-macos_x86_64-static-combined.a \
             onnxruntime-macos_arm64-static-combined.a \
     -output "${OUTPUT_DIR}/libonnxruntime.a"
rm onnxruntime-macos_x86_64-static-combined.a
rm onnxruntime-macos_arm64-static-combined.a
