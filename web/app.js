const SAMPLE_RATE = 22050;

let audioContext;

// Event listener for user interaction (e.g., unlocking audio context on click)
document.addEventListener('click', function() {
    let context = getAudioContext();
    if (context.state === 'suspended') {
        context.resume();
    }
});

function getAudioContext() {
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)({sampleRate: SAMPLE_RATE});
    }
    return audioContext;
}

// Initialize the worker
const worker = new Worker('worker.js');

// File input element
const fileInput = document.getElementById('audio-upload');
const processBtn = document.getElementById('process-audio');
const outputDiv = document.getElementById('output');

// Function to process the uploaded audio
function processAudio(inputData) {
    const reader = new FileReader();

    reader.onload = function(event) {
        const arrayBuffer = event.target.result;

        audioContext.decodeAudioData(arrayBuffer, function(decodedData) {
            let leftChannel, rightChannel;
            // decodedData is an AudioBuffer
            if (decodedData.numberOfChannels === 1) {
                // Mono case
                leftChannel = decodedData.getChannelData(0);
                rightChannel = decodedData.getChannelData(0);
            } else {
                // Stereo case
                leftChannel = decodedData.getChannelData(0);
                rightChannel = decodedData.getChannelData(1);
            }

            // Now that the audio is decoded, send it to the WASM worker
            sendAudioToWasm(leftChannel, rightChannel);
        });
    };

    reader.readAsArrayBuffer(inputData);
}

// Function to send the audio data to the WASM worker
function sendAudioToWasm(leftChannel, rightChannel) {
    console.log("Sending audio data to WASM for processing...");

    // Downsample to mono by averaging left and right channels
    const monoAudioData = new Float32Array(leftChannel.length);
    for (let i = 0; i < leftChannel.length; i++) {
        monoAudioData[i] = (leftChannel[i] + rightChannel[i]) / 2.0;
    }

    // Post a message to the worker to load the WASM module
    worker.postMessage({
        msg: 'LOAD_WASM',
        scriptName: 'basicpitch.js' // The glue code for basicpitch WASM
    });

    // After the WASM is loaded, process the audio data
    worker.onmessage = function(e) {
        if (e.data.msg === 'WASM_READY') {
            console.log('WASM module loaded and ready.');

            // Send the mono audio data for processing
            worker.postMessage({
                msg: 'PROCESS_AUDIO',
                inputData: monoAudioData.buffer, // Pass mono audio as ArrayBuffer
                length: monoAudioData.length     // Pass the correct length
            });
        } else if (e.data.msg === 'PROCESSING_DONE') {
            document.getElementById('output').innerText = "WASM Processing complete!";
            console.log('WASM processing complete, output:', e.data.blob);

            // Create a Blob from the returned data
            const midiBlob = e.data.blob;
            const blobUrl = URL.createObjectURL(midiBlob);

            // Create a download link for the MIDI file
            const downloadLink = document.createElement('a');
            downloadLink.href = blobUrl;
            downloadLink.download = 'output.mid';
            downloadLink.innerText = 'Download MIDI File';

            // Append the download link to the output div
            const outputDiv = document.getElementById('output');
            outputDiv.appendChild(downloadLink);
        } else if (e.data.msg === 'PROCESSING_FAILED') {
            document.getElementById('output').innerText = "WASM Processing failed.";
            console.error('WASM processing failed');
        }
    };
}

processBtn.addEventListener('click', function() {
    const uploadedFile = fileInput.files[0];
    if (uploadedFile) {
        console.log('Audio file uploaded:', uploadedFile.name);
        processAudio(uploadedFile);
    }
});
