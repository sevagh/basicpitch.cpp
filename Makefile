EMSDK_ENV_PATH?=/home/sevagh/repos/emsdk/emsdk_env.sh

default: cli

cli:
	cmake -S src_cli -B build -DCMAKE_BUILD_TYPE=Release
	cmake --build build -- -j16

cli-debug:
	cmake -S src_cli -B build -DCMAKE_BUILD_TYPE=Debug
	cmake --build build -- -j16

wasm:
	@/bin/bash -c 'source $(EMSDK_ENV_PATH) && \
		emcmake cmake -S src_wasm -B build-wasm -DCMAKE_BUILD_TYPE=Release \
		&& cmake --build build-wasm -- -j16'

clean-cli:
	rm -rf build

clean-wasm:
	rm -rf build-wasm
