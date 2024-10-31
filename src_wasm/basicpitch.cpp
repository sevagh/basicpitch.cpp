#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <emscripten.h>
#include <iostream>
#include <libremidi/libremidi.hpp>
#include <libremidi/writer.hpp>
#include <map>
#include <numeric>
#include <ranges>
#include <sstream>
#include <tuple>
#include <vector>

#include "basicpitch.hpp"

extern "C"
{

    // Define a JavaScript function using EM_JS to log for debugging
    EM_JS(void, callWriteWasmLog, (const char *str),
          {console.log(UTF8ToString(str))});

    EMSCRIPTEN_KEEPALIVE
    void convertToMidi(const float *mono_audio, int length,
                       uint8_t **midi_data_ptr, int *midi_size)
    {
        callWriteWasmLog("Starting inference...");

        auto inference_result = basic_pitch::ort_inference(mono_audio, length);

        callWriteWasmLog("Inference finished. Now generating MIDI file...");

        // Call the function to convert the output to MIDI
        std::vector<uint8_t> midiBytes =
            basic_pitch::convert_to_midi(inference_result);

        callWriteWasmLog("MIDI file generated. Now saving to blob...");

        // Log the size of the MIDI data
        std::ostringstream log_message;
        log_message << "MIDI data size: " << midiBytes.size();
        callWriteWasmLog(log_message.str().c_str());

        // Allocate memory in WASM for the MIDI data and copy the contents
        *midi_size = midiBytes.size();
        *midi_data_ptr = (uint8_t *)malloc(*midi_size);
        if (*midi_data_ptr == nullptr)
        {
            callWriteWasmLog("Failed to allocate memory for MIDI data.");

            // error occurred, set the output pointers to nullptr
            *midi_data_ptr = nullptr;
            *midi_size = 0;
            return;
        }
    }
}
