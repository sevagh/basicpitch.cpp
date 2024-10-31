# basicpitch.cpp

C++20 inference for the [Spotify basic-pitch](https://github.com/spotify/basic-pitch) with ONNXRuntime, Eigen, and libremidi. Demo apps are provided for WebAssembly/Emscripten and a cli app.

I use [ONNXRuntime](https://github.com/microsoft/onnxruntime) and scripts from the excellent [ort-builder](https://github.com/olilarkin/ort-builder) project to implement the neural network inference like so:
* Convert an ONNX model to ORT (onnxruntime)
* Include only the operations and types needed for the specific neural network, cutting down code size
* Compiling the model weights to a .c and .h file

After the neural network inference, I use [libremidi](https://github.com/celtera/libremidi) to replicate the end-to-end MIDI file creation of the real basic-pitch project.

## Library design

* [ort-model](./ort-model) contains the model in ONNX form, ORT form, and the generated h and c file
* [scripts](./scripts) contain the ORT model build scripts
* [src](./src) is the shared inference and MIDI creation code
* [src_wasm](./src_wasm) is the main WASM function, used in the web demo
* [src_cli](./src_cli) is a Linux cli app (for debugging purposes) that uses [libnyquist](https://github.com/ddiakopoulos/libnyquist) to load the audio files
* [vendor](./vendor) contains third-party/vendored libraries
* [web](./web) contains basic HTML/Javascript code to host the WASM demo

## Build 

1. Clone this repo with git submodules
2. You may even need to go into `vendor/` and init and pull all of their submodules


Create a mamba env for development. Needs a new version of cmake for onnxruntime

```
$ mamba create --name basicpitch python=3.11
$ pip install -r ./scripts/requirements.txt
$ mamba install cmake
```

For web testing:
```
$ python -m http.server 8000
```
