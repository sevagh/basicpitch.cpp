#!/usr/bin/env bash

# copied from https://github.com/olilarkin/ort-builder

python ./vendor/onnxruntime/tools/ci_build/build.py \
--build_dir ./build/build-ort-wasm \
--config=MinSizeRel \
--build_wasm_static_lib \
--parallel \
--minimal_build \
--disable_ml_ops \
--disable_rtti \
--include_ops_by_config ./ort-model/model.required_operators_and_types.config \
--enable_reduced_operator_type_support \
--skip_tests \
--enable_wasm_simd \
--enable_wasm_exception_throwing_override \
--disable_exceptions
