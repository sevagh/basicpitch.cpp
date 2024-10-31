#include "basicpitch.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <libremidi/libremidi.hpp>
#include <libremidi/writer.hpp>
#include <map>
#include <numeric>
#include <ranges>
#include <sstream>
#include <tuple>
#include <vector>

using namespace basic_pitch::constants;

static std::vector<std::pair<int, int>>
find_peaks(const Eigen::Tensor2dXf &onsets)
{
    std::vector<std::pair<int, int>> peaks;

    // Get the dimensions of the onsets tensor
    int n_times = onsets.dimension(0); // Number of time steps (rows)
    int n_freqs = onsets.dimension(1); // Number of frequency bins (columns)

    // Loop through the tensor to find peaks
    for (int t = 1; t < n_times - 1; ++t)
    {
        for (int f = 0; f < n_freqs; ++f)
        {
            // Check if the current element is a peak and exceeds the threshold
            if (onsets(t, f) > ONSET_THRESHOLD &&
                onsets(t, f) > onsets(t - 1, f) &&
                onsets(t, f) > onsets(t + 1, f))
            {

                peaks.emplace_back(t, f); // Store the peak (time, frequency)
            }
        }
    }

    return peaks;
}

static std::vector<float> model_frames_to_time(int n_frames)
{
    std::vector<float> times(n_frames);

    float original_time_factor =
        static_cast<float>(FFT_HOP) / static_cast<float>(SAMPLE_RATE);
    float window_factor = 1.0f / static_cast<float>(ANNOT_N_FRAMES);
    float window_offset =
        original_time_factor * (static_cast<float>(ANNOT_N_FRAMES) -
                                (static_cast<float>(AUDIO_N_SAMPLES) /
                                 static_cast<float>(FFT_HOP))) +
        0.0018f;

    for (int i = 0; i < n_frames; ++i)
    {
        float frame_index = static_cast<float>(i);
        float original_time = frame_index * original_time_factor;
        float window_number = frame_index * window_factor;
        times[i] = original_time - (window_offset * window_number);
    }

    return times;
}

static void
apply_melodia_trick(Eigen::MatrixXf &remaining_energy,
                    const Eigen::MatrixXf &frames, float frame_thresh,
                    int energy_tol, int min_note_len,
                    std::vector<basic_pitch::NoteEvent> &note_events)
{

    int n_times = remaining_energy.rows();

    // Continue applying the trick as long as there is energy above the
    // threshold
    while (remaining_energy.maxCoeff() > frame_thresh)
    {
        // Find the time-frequency point with maximum remaining energy
        Eigen::Index i_mid, freq_idx;
        float max_energy = remaining_energy.maxCoeff(&i_mid, &freq_idx);

        // Zero out the max energy point
        remaining_energy(i_mid, freq_idx) = 0.0f;

        // Forward pass to find note end
        int i = i_mid + 1;
        int k = 0;
        while (i < n_times - 1 && k < energy_tol)
        {
            if (remaining_energy(i, freq_idx) < frame_thresh)
            {
                k++;
            }
            else
            {
                k = 0;
            }
            remaining_energy(i, freq_idx) = 0.0f;

            // Zero out neighboring frequencies if applicable
            if (freq_idx < MAX_FREQ_IDX)
                remaining_energy(i, freq_idx + 1) = 0.0f;
            if (freq_idx > 0)
                remaining_energy(i, freq_idx - 1) = 0.0f;

            i++;
        }
        int i_end = i - 1 - k;

        // Backward pass to find note start
        i = i_mid - 1;
        k = 0;
        while (i > 0 && k < energy_tol)
        {
            if (remaining_energy(i, freq_idx) < frame_thresh)
            {
                k++;
            }
            else
            {
                k = 0;
            }
            remaining_energy(i, freq_idx) = 0.0f;

            // Zero out neighboring frequencies if applicable
            if (freq_idx < MAX_FREQ_IDX)
                remaining_energy(i, freq_idx + 1) = 0.0f;
            if (freq_idx > 0)
                remaining_energy(i, freq_idx - 1) = 0.0f;

            i--;
        }
        int i_start = i + 1 + k;

        // Ensure the note is long enough
        if (i_end - i_start <= min_note_len)
        {
            continue; // Skip short notes
        }

        // Calculate amplitude and store the new note event
        float amplitude = 0.0f;
        for (int t = i_start; t < i_end; ++t)
        {
            amplitude += frames(t, freq_idx);
        }
        amplitude /= (i_end - i_start);

        // Store note event (start, end, MIDI pitch, amplitude)
        note_events.emplace_back(i_start, i_end, freq_idx + MIDI_OFFSET,
                                 amplitude, std::nullopt);
    }
}

// Convert MIDI pitch to frequency (Hz)
static float midi_to_hz(float pitch_midi)
{
    return 440.0f *
           std::pow(2.0f, (pitch_midi - 69.0f) / 12.0f); // A4 is 440Hz, MIDI 69
}

static float midi_pitch_to_contour_bin(float pitch_midi)
{
    // Convert MIDI pitch to frequency (Hz)
    float pitch_hz = midi_to_hz(pitch_midi);

    // Calculate the corresponding bin in the contour matrix
    return 12.0f * CONTOURS_BINS_PER_SEMITONE *
           std::log2(pitch_hz / ANNOTATIONS_BASE_FREQUENCY);
}

static void add_pitch_bends(const Eigen::Tensor2dXf &contours,
                            std::vector<basic_pitch::NoteEvent> &note_events,
                            int n_bins_tolerance = 25)
{
    int n_freqs_contours = contours.dimension(1);
    const int window_length = n_bins_tolerance * 2 + 1;

    // Create Gaussian window similar to scipy.signal.windows.gaussian
    std::vector<float> freq_gaussian(window_length);
    float sigma = 5.0f;
    for (int i = 0; i < window_length; ++i)
    {
        float x = static_cast<float>(i - n_bins_tolerance);
        freq_gaussian[i] = std::exp(-(x * x) / (2 * sigma * sigma));
    }

    for (auto &note_event : note_events)
    {
        auto &[start_idx, end_idx, pitch_midi, amplitude, pitch_bends] =
            note_event;

        float bin_float =
            midi_pitch_to_contour_bin(static_cast<float>(pitch_midi));
        int freq_idx = static_cast<int>(std::round(bin_float));

        // Ensure frequency indices are within valid bounds
        int freq_start_idx = std::max(0, freq_idx - n_bins_tolerance);
        int freq_end_idx =
            std::min(n_freqs_contours, freq_idx + n_bins_tolerance + 1);

        // Adjust Gaussian window bounds to handle boundary conditions
        int gaussian_start = std::max(0, n_bins_tolerance - freq_idx);
        int gaussian_end =
            window_length -
            std::max(0, freq_idx - (n_freqs_contours - n_bins_tolerance - 1));

        // Shift factor for calculating relative bends
        int pb_shift =
            n_bins_tolerance - std::max(0, n_bins_tolerance - freq_idx);

        // Initialize pitch_bends only if needed
        pitch_bends
            .emplace(); // Initializes the std::optional with an empty vector
        pitch_bends->resize(end_idx -
                            start_idx); // Resize the vector within the optional

        // Apply Gaussian window and extract submatrix, performing element-wise
        // multiplication
        for (int t = start_idx; t < end_idx; ++t)
        {
            float max_val = -std::numeric_limits<float>::infinity();
            int max_idx = 0;

            // Loop within the Gaussian window limits: from `gaussian_start` to
            // `gaussian_end`
            for (int f = freq_start_idx, g = gaussian_start;
                 f < freq_end_idx && g < gaussian_end; ++f, ++g)
            {
                // Apply Gaussian window to each frequency bin in the current
                // frame
                float weighted_value = contours(t, f) * freq_gaussian[g];
                if (weighted_value > max_val)
                {
                    max_val = weighted_value;
                    max_idx = f; // Store the actual frequency bin index
                }
            }

            // Normalize the max index relative to the Gaussian window center
            (*pitch_bends)[t - start_idx] =
                (max_idx - freq_start_idx) - pb_shift;
        }
    }
}

// Function to drop pitch bends from overlapping notes
static void
drop_overlapping_pitch_bends(std::vector<basic_pitch::NoteEvent> &note_events)
{
    // Sort by start time
    std::sort(note_events.begin(), note_events.end());

    for (size_t i = 0; i < note_events.size() - 1; ++i)
    {
        for (size_t j = i + 1; j < note_events.size(); ++j)
        {
            // Check if start time of note j is after end time of note i
            if ((note_events[j].start_idx) >= (note_events[i].end_idx))
            {
                break; // No overlap
            }

            // Remove pitch bends from overlapping notes
            note_events[i].pitch_bends = std::nullopt;
            note_events[j].pitch_bends = std::nullopt;
        }
    }
}

// Main function to convert frames and onsets to note events
static std::vector<basic_pitch::NoteEvent>
output_to_notes_polyphonic(const basic_pitch::InferenceResult &inference_result,
                           const bool use_melodia_trick,
                           const bool include_pitch_bends)
{

    int n_times_onsets = inference_result.onsets.dimension(0);

    Eigen::Tensor2dXf frames = inference_result.notes;

    Eigen::Tensor2dXf remaining_energy =
        frames; // Clone frames as we will modify this in-place
    std::vector<basic_pitch::NoteEvent> note_events;

    // Find peaks in the onsets
    auto peaks = find_peaks(inference_result.onsets);

    // reverse sort the peaks by onset value
    // std::sort(filtered_peaks.begin(), filtered_peaks.end(),
    // std::greater<>());
    std::reverse(peaks.begin(), peaks.end());

    // Process peaks to generate note events
    for (const auto &[note_start_idx, freq_idx] : peaks)
    {
        int i = note_start_idx + 1;
        int k = 0;

        // Find the point where the note energy drops below the threshold
        while (i < n_times_onsets - 1 && k < ENERGY_TOL)
        {
            if (remaining_energy(i, freq_idx) < FRAME_THRESHOLD)
            {
                k++;
            }
            else
            {
                k = 0;
            }
            i++;
        }
        i -= k; // Adjust index

        if (i - note_start_idx <= MIN_NOTE_LEN)
            continue; // Skip short notes

        // Clear energy in the current frequency band
        for (int t = note_start_idx; t < i; ++t)
        {
            remaining_energy(t, freq_idx) = 0.0f;
            if (freq_idx > 0)
                remaining_energy(t, freq_idx - 1) = 0.0f;
            if (freq_idx < MAX_FREQ_IDX)
                remaining_energy(t, freq_idx + 1) = 0.0f;
        }

        // Calculate amplitude and store note event
        float amplitude = 0.0f;
        for (int t = note_start_idx; t < i; ++t)
        {
            amplitude += frames(t, freq_idx);
        }
        amplitude /= (i - note_start_idx);

        note_events.emplace_back(note_start_idx, i, freq_idx + MIDI_OFFSET,
                                 amplitude, std::nullopt);
    }

    if (use_melodia_trick)
    {
        // get matrixxf of remaining_energy, frames to apply melodia trick
        Eigen::MatrixXf remaining_energy_mat = Eigen::Map<Eigen::MatrixXf>(
            remaining_energy.data(), remaining_energy.dimension(0),
            remaining_energy.dimension(1));
        Eigen::MatrixXf frames_mat = Eigen::Map<Eigen::MatrixXf>(
            frames.data(), frames.dimension(0), frames.dimension(1));
        apply_melodia_trick(remaining_energy_mat, frames_mat, FRAME_THRESHOLD,
                            ENERGY_TOL, MIN_NOTE_LEN, note_events);
    }

    if (include_pitch_bends)
    {
        add_pitch_bends(inference_result.contours, note_events);
    }
    return note_events;
}

static uint32_t time_to_ticks(float time_seconds, int tempo_us,
                              int tpqn = DEFAULT_TPQN)
{
    return static_cast<uint32_t>(
        std::round((time_seconds * tpqn * 1'000'000) / tempo_us));
}

static libremidi::writer
note_events_to_midi(const std::vector<basic_pitch::NoteEvent> &note_events,
                    int n_times_onsets)
{

    libremidi::writer midi_writer;
    midi_writer.ticksPerQuarterNote = DEFAULT_TPQN;

    // Track with tempo and time signature
    libremidi::midi_track meta_track;
    meta_track.emplace_back(0, 0, libremidi::meta_events::tempo(MIDI_TEMPO_US));
    meta_track.emplace_back(
        0, 0,
        libremidi::meta_events::time_signature(TIME_SIGNATURE_NUMERATOR,
                                               TIME_SIGNATURE_DENOMINATOR));
    midi_writer.tracks.push_back(meta_track);

    // Calculate frame times for each note onset
    std::vector<float> frame_times = model_frames_to_time(n_times_onsets);

    // Create a vector to hold all events with their absolute tick times
    struct MidiEvent
    {
        uint32_t tick; // Absolute tick time
        libremidi::message message;
    };

    std::vector<MidiEvent> midi_events;

    std::cout << "Before iterating over note events" << std::endl;

    // Iterate over note events
    for (const auto &[start_idx, end_idx, pitch, amplitude, pitch_bend_opt] :
         note_events)
    {
        float start_time = frame_times[start_idx];
        float end_time = frame_times[end_idx];
        uint32_t start_tick = time_to_ticks(start_time, MIDI_TEMPO_US);
        uint32_t end_tick = time_to_ticks(end_time, MIDI_TEMPO_US);
        int velocity = static_cast<int>(amplitude * 127);

        // Add `Note_on` event at start_tick
        midi_events.push_back({start_tick, libremidi::channel_events::note_on(
                                               0, pitch, velocity)});

        // Process pitch bends directly without allocating a sub-vector
        if (pitch_bend_opt.has_value())
        {
            const std::vector<int> &pitch_bend = pitch_bend_opt.value();
            int num_bends = pitch_bend.size();

            if (num_bends > 1)
            {
                float time_increment =
                    (end_time - start_time) / (num_bends - 1);

                for (int i = 0; i < num_bends; ++i)
                {
                    float bend_time = start_time + i * time_increment;
                    uint32_t bend_tick =
                        time_to_ticks(bend_time, MIDI_TEMPO_US);

                    // Ensure bend_tick does not exceed end_tick
                    bend_tick = std::min(bend_tick, end_tick);

                    int bend_value =
                        pitch_bend[i] * (4096 / CONTOURS_BINS_PER_SEMITONE) +
                        8192;
                    bend_value = std::clamp(bend_value, 0, 16383);

                    // Add pitch bend event at bend_tick
                    midi_events.push_back(
                        {bend_tick,
                         libremidi::channel_events::pitch_bend(0, bend_value)});
                }
            }
            else
            {
                // Single pitch bend case
                uint32_t bend_tick = start_tick;
                int bend_value =
                    pitch_bend[0] * (4096 / CONTOURS_BINS_PER_SEMITONE) + 8192;
                bend_value = std::clamp(bend_value, 0, 16383);

                // Add pitch bend event at start_tick
                midi_events.push_back(
                    {bend_tick,
                     libremidi::channel_events::pitch_bend(0, bend_value)});
            }
        }

        // Add `Note_off` event at end_tick
        midi_events.push_back(
            {end_tick, libremidi::channel_events::note_off(0, pitch, 0)});
    }

    std::cout << "After iterating over note events" << std::endl;

    // Sort all events by their absolute tick times
    std::sort(midi_events.begin(), midi_events.end(),
              [](const MidiEvent &a, const MidiEvent &b)
              {
                  if (a.tick != b.tick)
                  {
                      return a.tick < b.tick; // Primary sorting by tick
                  }
                  else
                  {
                      // Secondary sorting by message type
                      return a.message.get_message_type() <
                             b.message.get_message_type();
                  }
              });

    std::cout << "Now creating instrument track" << std::endl;

    // Now, compute delta times and add events to the instrument track
    libremidi::midi_track instrument_track;
    instrument_track.emplace_back(0, 0,
                                  libremidi::channel_events::program_change(
                                      0, 4)); // Set program to Electric Piano

    uint32_t last_tick = 0;
    for (const auto &event : midi_events)
    {
        uint32_t delta_ticks = event.tick - last_tick;
        instrument_track.emplace_back(delta_ticks, 0, event.message);
        last_tick = event.tick;
    }

    midi_writer.tracks.push_back(instrument_track);
    return midi_writer;
}

std::vector<uint8_t> basic_pitch::convert_to_midi(
    const basic_pitch::InferenceResult &inference_result,
    const bool use_melodia_trick,  // defaults to true
    const bool include_pitch_bends // defaults to false
)
{
    // Process the unwrapped notes and onsets to detect note events

    std::cout << "output_to_notes_polyphonic" << std::endl;

    std::vector<basic_pitch::NoteEvent> note_events =
        output_to_notes_polyphonic(inference_result, use_melodia_trick,
                                   include_pitch_bends);

    if (include_pitch_bends)
    {
        // Drop pitch bends from overlapping notes
        drop_overlapping_pitch_bends(note_events);
    }

    int n_times_notes = inference_result.notes.dimension(0);

    std::cout << "note_events_to_midi" << std::endl;

    // Convert the detected note events to a MIDI writer object
    libremidi::writer midi_writer =
        note_events_to_midi(note_events, n_times_notes);

    std::cout << "done!" << std::endl;

    // Use ostringstream to write MIDI data to a byte stream
    std::ostringstream output(std::ios::binary);
    midi_writer.write(output);

    // Convert the byte stream to a string
    std::string midi_string = output.str();

    // Create a vector to hold the MIDI data
    std::vector<uint8_t> midi_data(midi_string.begin(), midi_string.end());

    // Return the MIDI data as bytes
    return midi_data;
}
