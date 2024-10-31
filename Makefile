EMSDK_ENV_PATH?=/home/sevagh/repos/emsdk/emsdk_env.sh

default: cli

cli:
	cmake -S src_cli -B build/build-cli -DCMAKE_BUILD_TYPE=Release
	cmake --build build/build-cli -- -j16

cli-debug:
	cmake -S src_cli -B build/build-cli -DCMAKE_BUILD_TYPE=Debug
	cmake --build build/build-cli -- -j16

wasm:
	@/bin/bash -c 'source $(EMSDK_ENV_PATH) && \
		emcmake cmake -S src_wasm -B build/build-wasm -DCMAKE_BUILD_TYPE=Release \
		&& cmake --build build/build-wasm -- -j16'

clean-all:
	rm -rf build

clean-cli:
	rm -rf build/build-cli
	
clean-wasm:
	rm -rf build/build-wasm
