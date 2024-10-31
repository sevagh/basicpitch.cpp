#include <Eigen/Dense>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <unsupported/Eigen/CXX11/Tensor>

// this is the nmp model baked into a header file
#include "basicpitch.hpp"
#include "model.ort.h"

using namespace basic_pitch::constants;

static Eigen::Tensor2dXf
unwrap_output(const Eigen::Tensor3dRowMajorXf &tensor_3d,
              int audio_original_length, int n_overlapping_frames)
{
    int batch_size = tensor_3d.dimension(0); // Number of batches (chunks)
    int n_times_short =
        tensor_3d.dimension(1);           // Number of time steps per chunk
    int n_freqs = tensor_3d.dimension(2); // Frequency bins

    int n_olap = n_overlapping_frames / 2;

    // Remove overlapping frames from both start and end
    Eigen::array<int, 3> offsets = {0, n_olap, 0};
    Eigen::array<int, 3> extents = {batch_size, n_times_short - 2 * n_olap,
                                    n_freqs};
    Eigen::Tensor<float, 3, Eigen::RowMajor> output_sliced =
        tensor_3d.slice(offsets, extents);

    // Flatten the 3D tensor into a 2D tensor
    int total_time_steps = batch_size * (n_times_short - 2 * n_olap);
    Eigen::Tensor<float, 2, Eigen::RowMajor> unwrapped_output =
        output_sliced.reshape(Eigen::array<int, 2>{total_time_steps, n_freqs});

    // Calculate the expected output length
    int n_output_frames_original = static_cast<int>(
        std::floor(audio_original_length *
                   (ANNOTATIONS_FPS / static_cast<float>(AUDIO_SAMPLE_RATE))));
    n_output_frames_original =
        std::min(n_output_frames_original,
                 static_cast<int>(unwrapped_output.dimension(0)));

    // Trim the output tensor to match the original audio length
    Eigen::Tensor<float, 2, Eigen::RowMajor> final_output =
        unwrapped_output.slice(
            Eigen::array<int, 2>{0, 0},
            Eigen::array<int, 2>{n_output_frames_original, n_freqs});

    // Return the final output as a column-major tensor
    return final_output.swap_layout().shuffle(Eigen::array<int, 2>{1, 0});
}

basic_pitch::InferenceResult
basic_pitch::ort_inference(const std::vector<float> &mono_audio)
{
    return ort_inference(mono_audio.data(), mono_audio.size());
}

basic_pitch::InferenceResult basic_pitch::ort_inference(const float *mono_audio,
                                                        int length)
{
    // Initialize ONNX Runtime environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "basic_pitch");

    // Set session options (use defaults)
    Ort::SessionOptions session_options;

    // Create the ONNX Runtime session from the in-memory ORT model
    Ort::Session session(env, model_ort_start, model_ort_size, session_options);

    // Constants for processing; overlap 30 frames
    const int chunk_size = AUDIO_N_SAMPLES;
    int n_overlapping_frames = 30;
    int overlap_len = n_overlapping_frames * FFT_HOP;
    int hop_size = AUDIO_N_SAMPLES - overlap_len;

    // Padding the start of the audio (overlap_len / 2 zeros at the start)
    std::vector<float> padded_audio(overlap_len / 2, 0.0f);
    padded_audio.insert(padded_audio.end(), mono_audio, mono_audio + length);

    // Calculate the new length after padding
    int padded_length = padded_audio.size();
    int num_chunks = (padded_length + hop_size - 1) / hop_size;

    Ort::AllocatorWithDefaultOptions allocator;
    std::array<int64_t, 3> input_shape = {num_chunks, chunk_size, 1};

    // Allocate ONNX tensor up front
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        allocator, input_shape.data(), input_shape.size());

    float *ort_tensor_data = input_tensor.GetTensorMutableData<float>();

    for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx)
    {
        int start_pos = chunk_idx * hop_size;
        int actual_chunk_size = std::min(chunk_size, padded_length - start_pos);

        std::copy(padded_audio.begin() + start_pos,
                  padded_audio.begin() + start_pos + actual_chunk_size,
                  ort_tensor_data + chunk_idx * chunk_size); // ORT tensor

        // Zero-pad the last chunk if it's smaller than chunk_size
        if (actual_chunk_size < chunk_size)
        {
            std::fill(ort_tensor_data + chunk_idx * chunk_size +
                          actual_chunk_size,
                      ort_tensor_data + (chunk_idx + 1) * chunk_size, 0.0f);
        }
    }

    // Input and output names
    const char *input_names[] = {"serving_default_input_2:0"};
    const char *output_names[] = {
        "StatefulPartitionedCall:1", // note
        "StatefulPartitionedCall:2", // onset
        "StatefulPartitionedCall:0"  // contour
    };

    // Run the inference
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names,
                                      &input_tensor, 1, output_names, 3);

    // Retrieve and process shapes for each output
    std::vector<int64_t> note_shape =
        output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    std::vector<int64_t> onset_shape =
        output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();
    std::vector<int64_t> contour_shape =
        output_tensors[2].GetTensorTypeAndShapeInfo().GetShape();

    int batch_size = note_shape[0];
    int n_times_short_notes =
        note_shape[1]; // Number of time steps for notes and onsets
    int n_freqs_notes = note_shape[2]; // 88 for notes and onsets
    int n_times_short_contours = contour_shape[1];
    int n_freqs_contours = contour_shape[2]; // 264 for contours

    // Get the original length of the audio (in samples)
    int audio_original_length = length;

    // Access raw output data
    float *note_data = output_tensors[0].GetTensorMutableData<float>();
    float *onset_data = output_tensors[1].GetTensorMutableData<float>();
    float *contour_data = output_tensors[2].GetTensorMutableData<float>();

    // Use Eigen::TensorMap to map the ONNX Runtime row-major data
    Eigen::TensorMap<Eigen::Tensor3dRowMajorXf> note_tensor(
        note_data, batch_size, n_times_short_notes, n_freqs_notes);
    Eigen::TensorMap<Eigen::Tensor3dRowMajorXf> onset_tensor(
        onset_data, batch_size, n_times_short_notes, n_freqs_notes);
    Eigen::TensorMap<Eigen::Tensor3dRowMajorXf> contour_tensor(
        contour_data, batch_size, n_times_short_contours, n_freqs_contours);

    // Use unwrap_output to unwrap and convert the row-major 3D tensors to
    // col-major 2D tensors
    Eigen::Tensor2dXf unwrapped_notes =
        unwrap_output(note_tensor, audio_original_length, 30);
    Eigen::Tensor2dXf unwrapped_onsets =
        unwrap_output(onset_tensor, audio_original_length, 30);
    Eigen::Tensor2dXf unwrapped_contours =
        unwrap_output(contour_tensor, audio_original_length, 30);

    return InferenceResult{unwrapped_notes, unwrapped_onsets,
                           unwrapped_contours};
}
