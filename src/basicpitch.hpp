#ifndef BASIC_PITCH_HPP
#define BASIC_PITCH_HPP

#include <Eigen/Dense>
#include <cmath>
#include <complex>
#include <iostream>
#include <optional>
#include <string>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

namespace Eigen
{
// define some Tensors
typedef Tensor<float, 3> Tensor3dXf;
typedef Tensor<float, 3, Eigen::RowMajor> Tensor3dRowMajorXf;
typedef Tensor<float, 2> Tensor2dXf;
typedef Tensor<float, 1> Tensor1dXf;
typedef Matrix<float, Dynamic, Dynamic> MatrixXf;
} // namespace Eigen

namespace basic_pitch
{
namespace constants
{
const int SAMPLE_RATE = 22050;
const int AUDIO_SAMPLE_RATE = SAMPLE_RATE;
const int FFT_HOP = 256;
constexpr int ANNOTATIONS_FPS = SAMPLE_RATE / FFT_HOP;
const float ONSET_THRESHOLD = 0.5f;
const float FRAME_THRESHOLD = 0.3f;
const float ANNOTATIONS_BASE_FREQUENCY = 27.5f; // lowest key on a piano
const float MAGIC_NUMBER = 0.0018f;
const int CONTOURS_BINS_PER_SEMITONE = 3;
const int AUDIO_WINDOW_LENGTH = 2;
const int MIN_NOTE_LEN = 11;
const int MIDI_OFFSET = 21;
const int MAX_FREQ_IDX = 87;
const int ENERGY_TOL = 11;
constexpr float ANNOT_N_FRAMES = ANNOTATIONS_FPS * AUDIO_WINDOW_LENGTH;
constexpr float AUDIO_N_SAMPLES = SAMPLE_RATE * AUDIO_WINDOW_LENGTH - FFT_HOP;

// tempo constants
const int MIDI_TEMPO_US = 500'000; // 120 BPM
const float MIDI_TEMPO_BPM = 120.0f;
const int TIME_SIGNATURE_NUMERATOR = 4;
const int TIME_SIGNATURE_DENOMINATOR = 4;

// midi constants
const int DEFAULT_TPQN = 220; // ticks per quarter note
};                            // namespace constants

struct InferenceResult
{
    Eigen::Tensor2dXf notes;
    Eigen::Tensor2dXf onsets;
    Eigen::Tensor2dXf contours;
};

InferenceResult ort_inference(const std::vector<float> &mono_audio);
InferenceResult ort_inference(const float *mono_audio, int length);

struct NoteEvent
{
    int start_idx;
    int end_idx;
    int pitch;
    float amplitude;

    std::optional<std::vector<int>> pitch_bends;

    // comparison operator for sorting
    bool operator<(const NoteEvent &other) const
    {
        return std::tie(start_idx, end_idx, pitch, amplitude, pitch_bends) <
               std::tie(other.start_idx, other.end_idx, other.pitch,
                        other.amplitude, other.pitch_bends);
    }
};

std::vector<uint8_t> convert_to_midi(const InferenceResult &inference_result,
                                     const bool use_melodia_trick = true,
                                     const bool include_pitch_bends = true);
} // namespace basic_pitch

#endif // BASIC_PITCH_HPP
