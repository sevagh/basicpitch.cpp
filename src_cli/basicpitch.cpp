#include "basicpitch.hpp"
#include "MultiChannelResampler.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <libnyquist/Common.h>
#include <libnyquist/Decoders.h>
#include <libnyquist/Encoders.h>
#include <map>
#include <numeric>
#include <ranges>
#include <sstream>
#include <stddef.h>
#include <tuple>
#include <vector>

using namespace nqr;
using namespace basic_pitch::constants;

static std::vector<float> load_audio_file(std::string filename)
{
    // load a wav file with libnyquist
    std::shared_ptr<AudioData> fileData = std::make_shared<AudioData>();
    NyquistIO loader;
    loader.Load(fileData.get(), filename);

    std::cout << "Input samples: "
              << fileData->samples.size() / fileData->channelCount << std::endl;
    std::cout << "Length in seconds: " << fileData->lengthSeconds << std::endl;
    std::cout << "Number of channels: " << fileData->channelCount << std::endl;

    if (fileData->channelCount != 2 && fileData->channelCount != 1)
    {
        std::cerr
            << "[ERROR] basicpitch.cpp only supports mono and stereo audio"
            << std::endl;
        exit(1);
    }

    // number of samples per channel
    std::size_t N = fileData->samples.size() / fileData->channelCount;

    std::vector<float> mono_audio(N);

    if (fileData->channelCount == 1)
    {
        for (std::size_t i = 0; i < N; ++i)
        {
            mono_audio[i] = fileData->samples[i];
        }
    }
    else
    {
        // Stereo case: downmix to mono
        for (std::size_t i = 0; i < N; ++i)
        {
            mono_audio[i] =
                (fileData->samples[2 * i] + fileData->samples[2 * i + 1]) /
                2.0f;
        }
    }

    // Check if resampling is needed
    if (fileData->sampleRate != SAMPLE_RATE)
    {
        std::cout << "Resampling from " << fileData->sampleRate << " Hz to "
                  << SAMPLE_RATE << " Hz" << std::endl;

        // Resampling using Oboe's resampler module
        aaudio::resampler::MultiChannelResampler *resampler =
            aaudio::resampler::MultiChannelResampler::make(
                1, // Mono (1 channel)
                fileData->sampleRate, SAMPLE_RATE,
                aaudio::resampler::MultiChannelResampler::Quality::Best);

        int numInputFrames =
            N; // Since the audio is mono, numInputFrames is just N
        int numOutputFrames =
            static_cast<int>(static_cast<double>(numInputFrames) * SAMPLE_RATE /
                                 fileData->sampleRate +
                             0.5);

        std::vector<float> resampledAudio(
            numOutputFrames); // Resampled mono audio

        float *inputBuffer = mono_audio.data();
        float *outputBuffer = resampledAudio.data();

        int inputFramesLeft = numInputFrames;
        int numResampledFrames = 0;

        while (inputFramesLeft > 0 && numResampledFrames < numOutputFrames)
        {
            if (resampler->isWriteNeeded())
            {
                resampler->writeNextFrame(inputBuffer);
                inputBuffer++;
                inputFramesLeft--;
            }
            else
            {
                resampler->readNextFrame(outputBuffer);
                outputBuffer++;
                numResampledFrames++;
            }
        }

        while (!resampler->isWriteNeeded() &&
               numResampledFrames < numOutputFrames)
        {
            resampler->readNextFrame(outputBuffer);
            outputBuffer++;
            numResampledFrames++;
        }

        delete resampler;

        return resampledAudio;
    }

    return mono_audio;
}

int main(int argc, const char **argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <wav file> <out dir>"
                  << std::endl;
        exit(1);
    }

    std::cout << "basicpitch.cpp Main driver program" << std::endl;
    // load audio passed as argument
    std::string wav_file = argv[1];

    // output dir passed as argument
    std::string out_dir = argv[2];

    // Check if the output directory exists, and create it if not
    std::filesystem::path output_dir_path(out_dir);
    if (!std::filesystem::exists(output_dir_path))
    {
        std::cerr << "Directory does not exist: " << out_dir << ". Creating it."
                  << std::endl;
        if (!std::filesystem::create_directories(output_dir_path))
        {
            std::cerr << "Error: Unable to create directory: " << out_dir
                      << std::endl;
            return 1;
        }
    }
    else if (!std::filesystem::is_directory(output_dir_path))
    {
        std::cerr << "Error: " << out_dir << " exists but is not a directory!"
                  << std::endl;
        return 1;
    }

    std::cout << "Predicting MIDI for: " << wav_file << std::endl;

    std::vector<float> audio = load_audio_file(wav_file);

    auto inference_result = basic_pitch::ort_inference(audio);

    Eigen::Tensor2dXf unwrapped_notes = inference_result.notes;
    Eigen::Tensor2dXf unwrapped_onsets = inference_result.onsets;
    Eigen::Tensor2dXf unwrapped_contours = inference_result.contours;

    // Call the function to convert the output to MIDI
    std::vector<uint8_t> midiBytes =
        basic_pitch::convert_to_midi(inference_result);

    // Log the size of the MIDI data
    std::ostringstream log_message;
    log_message << "MIDI data size: " << midiBytes.size();

    std::cout << log_message.str() << std::endl;

    // write the midiBytes to a file 'output.mid' in the output directory
    // we dont need to use libremidi itself since the bytes are already correct
    // just write the bytes to a file

    // Generate MIDI output file name with .mid extension
    std::filesystem::path midi_file =
        output_dir_path / std::filesystem::path(wav_file).filename();
    midi_file.replace_extension(".mid");

    std::ofstream midi_stream(midi_file, std::ios::binary);
    midi_stream.write(reinterpret_cast<const char *>(midiBytes.data()),
                      midiBytes.size());

    std::cout << "Wrote MIDI file to: " << midi_file << std::endl;

    return 0;
}
