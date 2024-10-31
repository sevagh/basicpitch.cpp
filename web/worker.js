let wasmModule;
let loadedModule;

onmessage = function(e) {
    if (e.data.msg === 'LOAD_WASM') {
        loadWASMModule(e.data.scriptName);
    } else if (e.data.msg === 'PROCESS_AUDIO') {
        const inputData = new Float32Array(e.data.inputData); // Convert back from ArrayBuffer
        const length = e.data.length;  // Use the correct length

        console.log('Running inference with WASM...');
        console.log('length:', length);

        // Allocate memory in WASM and copy input data into the WASM memory
        const audioPointer = loadedModule._malloc(inputData.length * inputData.BYTES_PER_ELEMENT);
        const wasmInputArray = new Float32Array(loadedModule.HEAPF32.buffer, audioPointer, inputData.length);
        wasmInputArray.set(inputData);

        // Allocate memory for MIDI data pointer and size
        const midiDataPointer = loadedModule._malloc(4);  // Allocate 4 bytes for the pointer (uint8_t*)
        const midiSizePointer = loadedModule._malloc(4);  // Allocate 4 bytes for the size (int)

        // Call the WASM function with the audio buffer, length, and pointers for the MIDI data and size
        loadedModule._convertToMidi(audioPointer, length, midiDataPointer, midiSizePointer);

        // Retrieve the MIDI data pointer and size from WASM memory
        const midiData = loadedModule.getValue(midiDataPointer, 'i32');  // Get the pointer to MIDI data
        const midiSize = loadedModule.getValue(midiSizePointer, 'i32'); // Get the size of the MIDI data

        // If valid MIDI data was returned
        if (midiData !== 0 && midiSize > 0) {
            // Access the MIDI data from WASM memory
            const midiBytes = new Uint8Array(loadedModule.HEAPU8.buffer, midiData, midiSize);

            // Create a Blob from the MIDI data
            const blob = new Blob([midiBytes], { type: 'audio/midi' });

            // Optionally, create a URL from the Blob
            const blobUrl = URL.createObjectURL(blob);

            // Send the Blob (or the Blob URL) back to the main thread
            postMessage({
                msg: 'PROCESSING_DONE',
                blob: blob,  // Send the Blob directly
                blobUrl: blobUrl // Alternatively, send the Blob URL
            });

            // Free the memory allocated for the MIDI data in WASM
            loadedModule._free(midiData);
        } else {
            console.error('Failed to generate MIDI data.');
            console.log('midiData:', midiData);
            console.log('midiSize:', midiSize);
            postMessage({ msg: 'PROCESSING_FAILED' });
        }

        // Free the memory allocated in WASM for the input audio and the MIDI pointer/size
        loadedModule._free(audioPointer);
        loadedModule._free(midiDataPointer);
        loadedModule._free(midiSizePointer);
    }
};

function loadWASMModule(scriptName) {
    //importScripts(scriptName);  // Load the WASM glue code
    //<script src="basicpitch.js?v=<?= time() ?>"></script>
    importScripts(`${scriptName}?v=${new Date().getTime()}`);  // Load the WASM glue code w/ cache busting

    // Initialize the WASM module (which should set `Module`)
    wasmModule = libbasicpitch(); // Module is created in the glue code

    wasmModule.then((loaded_module) => {
        console.log('WASM module loaded:', loaded_module);

        postMessage({ msg: 'WASM_READY' });

        loadedModule = loaded_module;
    });
}
