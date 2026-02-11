//
//
// MIT License
//
// Copyright (c) 2025 Daniel Lopez
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
//

#if __cplusplus >= 202302L // Sorry this uses c++-23 features

#ifndef WAVE_T_HPP
#define WAVE_T_HPP

#include <bit>
#include <bitset>
#include <cfenv>
#include <climits>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <format>
#include <functional>
#include <future>
#include <generator>
#include <ios>
#include <latch>
#include <limits>
#include <numbers>
#include <optional>
#include <sstream>
#include <stdarg.h>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std::literals::complex_literals;

constexpr auto RIFF_ASCII = std::byteswap(0x52494646);
constexpr auto WAVE_ASCII = std::byteswap(0x57415645);
constexpr auto FMT_ASCII = std::byteswap(0x666d7420);
constexpr auto DATA_ASCII = std::byteswap(0x64617461);
constexpr auto PCM{1};
constexpr size_t EXTRA_PARAM_SIZE_OFFSET{35};
constexpr auto _8_BITS_PER_SAMPLE{8};
constexpr auto _16_BITS_PER_SAMPLE{16};
constexpr auto _24_BITS_PER_SAMPLE{24};
constexpr auto _32_BITS_PER_SAMPLE{32};
constexpr auto INT24_MAX{8388607};
constexpr auto BITS_PER_BYTE{8};
constexpr auto DEFAULT_SUB_CHUNK_1_SIZE{16};
constexpr auto DEFAULT_RESERVE_VALUE{44100 * 60 * 5};
constexpr auto MAX_OSC_SUPPORT{7}; // Change this if we are going to support
                                   // more or less (but hopefully more)

enum wave_type_t : uint8_t {
  linear = 0,
  sine = 1,
  triangle = 2,
  square = 4,
  sawtooth = 8
};

// Provides functions that generate signals and dft and invesre dft function
namespace helper {

constexpr double set_volume(double percent, size_t max_sample_value) {
  if (percent > 1.0) {
    percent = 1.0;
  }
  if (percent < 0.0) {
    percent = 0.0;
  }
  return static_cast<double>(max_sample_value - 1) * percent;
}

// Source: https://en.wikipedia.org/wiki/Sign_function
constexpr double sign(double x) {
  if (x > 0) {
    return 1.0;
  } else if (x < 0) {
    return -1.0;
  } else {
    return 0.0;
  }
}

// Source: https://en.wikipedia.org/wiki/Sine_wave
constexpr int32_t pcm_sine(double frequency, double time, double amplititude,
                           double phase) {
  return static_cast<int32_t>(
      amplititude * sin((2.0 * std::numbers::pi * frequency * time) + phase));
}

// Source: https://en.wikipedia.org/wiki/Triangle_wave
constexpr int32_t pcm_triangle(double time, double amplititude,
                               double frequency) {
  const double period = (1.0 / frequency);
  return static_cast<int32_t>(
      fabs(((2.0 * amplititude) / std::numbers::pi) *
           asin(sin(2.0 * time * (std::numbers::pi / period)))));
};

// Source: https://en.wikipedia.org/wiki/Square_wave_(waveform)
constexpr int32_t pcm_square(double time, double amplititude,
                             double frequency) {
  return static_cast<int32_t>(
      amplititude * sign(sin(2 * std::numbers::pi * frequency * time)));
}

// Source: https://en.wikipedia.org/wiki/Sawtooth_wave
constexpr int32_t pcm_saw_tooth(double time, double amplititude,
                                double frequency) {
  const double period = (1.0 / frequency);
  return static_cast<int32_t>(2.0 * amplititude *
                              ((time / period) - floor(0.5 + (time / period))));
}

void inverse_discrete_fourier_transform_async(
    size_t sample_size, std::vector<double> &time_domain,
    const std::vector<std::complex<double>> &frequency_domain) {
  std::vector<std::future<double>> futures;
  for (size_t frequency = 0; frequency < sample_size; frequency++) {
    auto future = std::async(
        std::launch::async, [&sample_size, frequency, &frequency_domain]() {
          double real = 0.0f;
          for (size_t n = 0; n < sample_size; n++) {
            const double x = std::norm(frequency_domain[n]);
            double ratio =
                (static_cast<double>(n) / static_cast<double>(sample_size));
            double z =
                2.0 * std::numbers::pi * static_cast<double>(frequency) * ratio;
            real += x * cos(z);
          }
          return (real / static_cast<double>(sample_size));
        });
    futures.push_back(std::move(future));
  }
  for (auto &future : futures) {
    time_domain.push_back(future.get());
  }
}

void inverse_discrete_fourier_transform(
    size_t sample_size, std::vector<double> &time_domain,
    const std::vector<std::complex<double>> &frequency_domain) {
  for (size_t frequency = 0; frequency < sample_size; frequency++) {
    double real = 0.0f;
    for (size_t n = 0; n < sample_size; n++) {
      const double x = std::norm(frequency_domain[n]);
      double ratio =
          (static_cast<double>(n) / static_cast<double>(sample_size));
      double z =
          2.0 * std::numbers::pi * static_cast<double>(frequency) * ratio;
      real += x * cos(z);
    }
    time_domain.push_back(real / static_cast<double>(sample_size));
  }
}

void discrete_fourier_transform(
    size_t sample_size, const std::vector<double> &time_domain,
    std::vector<std::complex<double>> &frequency_domain) {
  // O(n^2)
  for (size_t frequency = 0; frequency < sample_size; frequency++) {
    std::complex<double> result = 0.0f + 0.0if;
    for (size_t n = 0; n < sample_size; n++) {
      const double &x = time_domain[n];
      double ratio =
          (static_cast<double>(n) / static_cast<double>(sample_size));
      double z =
          2.0 * std::numbers::pi * static_cast<double>(frequency) * ratio;
      result += std::complex<double>((x * cos(z)), -(x * sin(z)));
    }
    frequency_domain.push_back(result);
  }
}

void discrete_fourier_transform_async(
    size_t sample_size, const std::vector<double> &time_domain,
    std::vector<std::complex<double>> &frequency_domain) {
  std::vector<std::future<std::complex<double>>> futures;
  for (size_t frequency = 0; frequency < sample_size; frequency++) {
    auto future = std::async(
        std::launch::async, [&sample_size, &time_domain, frequency]() {
          std::complex<double> result = 0.0f + 0.0if;
          for (size_t n = 0; n < sample_size; n++) {
            const double &x = time_domain[n];
            const double ratio =
                (static_cast<double>(n) / static_cast<double>(sample_size));
            const double z =
                2.0 * std::numbers::pi * static_cast<double>(frequency) * ratio;
            result += std::complex<double>((x * cos(z)), -(x * sin(z)));
          }
          return result;
        });
    futures.push_back(std::move(future));
  }
  for (auto &future : futures) {
    frequency_domain.push_back(future.get());
  }
}

// Converts a PCM sample (8-bit, 16-bit, 24-bit, etc.) to its decibel full scale
// value where 0 dB is like the max and beyond that is essenitally clipping
// Refer to: https://en.wikipedia.org/wiki/DBFS
constexpr double
get_decibel_fullscale_from_sample(const int64_t sample,
                                  const int32_t max_sample_value) {
  auto ratio = std::abs(static_cast<double>(sample)) /
               static_cast<double>(max_sample_value);
  auto original_rounding_method = std::fegetround();
  std::fesetround(FE_TOWARDZERO);
  auto dbfs =
      ratio > 0.98
          ? (std::floor((100.0 * std::log10(ratio) * 20.0) / 2.0) / 100.0)
          : (std::log10(ratio) * 20.0);
  std::fesetround(original_rounding_method);
  return dbfs;
}

} // namespace helper

enum oscillator_selection_t : uint8_t {
  none_selected = 0,
  oscillator_a = 1,
  oscillator_b = 2,
  oscillator_c = 3,
  oscillator_d = 4,
  oscillator_e = 5,
  oscillator_f = 6,
  oscillator_g = 7
};

enum oscillator_type_t : uint8_t {
  empty = 0,
  carrier = 1,
  frequency_modulation = 2,
  phase_modulation = 3,
  amplitude_modulation = 4,
  ring_modulation = 5
};

struct oscillator_config_t {
  oscillator_type_t operator_type;
  uint8_t wave_type; // Can be a combination of multiple waves so this
                     // oscilator can be already a saw and sqaure for example
  double frequency;
  oscillator_selection_t osc_to_modulate;
  double modulation_amplitude; // Used by modulation to control strength or
                               // amplitutde of the modulation signal
  double initial_phase_offset;
};

enum effects_type_t : uint8_t {
  bitcrusher_wet_percentage = 0,
  bitcrusher_gain_value = 1
};

struct lfo_config_t {
  double frequency; // Should be low frequency but we won't cap it the value so
                    // you could make it a hfo if that exists :)
  wave_type_t wave_type;
  effects_type_t effect_to_modulate;
};

struct synth_config_t {
  oscillator_config_t oscillator_a;
  oscillator_config_t oscillator_b;
  oscillator_config_t oscillator_c;
  oscillator_config_t oscillator_d;
  oscillator_config_t oscillator_e;
  oscillator_config_t oscillator_f;
  oscillator_config_t oscillator_g;
  lfo_config_t lfo;
  bool apply_bitcrusher_effect;
  double bitcrusher_gain_value;
  double bitcrusher_wet_percentage;
  bool empty(void) const {
    return oscillator_a.operator_type == oscillator_type_t::empty &&
           oscillator_b.operator_type == oscillator_type_t::empty &&
           oscillator_c.operator_type == oscillator_type_t::empty &&
           oscillator_d.operator_type == oscillator_type_t::empty &&
           oscillator_e.operator_type == oscillator_type_t::empty &&
           oscillator_f.operator_type == oscillator_type_t::empty &&
           oscillator_g.operator_type == oscillator_type_t::empty;
  }
};

// Internal for synth making -- not recommended to call these directly
namespace processing_functions {

std::vector<int32_t> oscillator_processing_callback(
    const oscillator_selection_t &osc_to_process, const size_t &sample_size,
    const double &volume, const bool &is_stereo, const uint32_t &sample_rate,
    synth_config_t &configuration,
    std::initializer_list<oscillator_config_t *> selected_oscs); // See implementation below

} // namespace processing_functions

struct wave_header_t {
  uint32_t chunk_id;
  uint32_t chunk_size;
  uint32_t format;
  uint32_t sub_chunk_1_id;
  uint32_t sub_chunk_1_size;
  uint16_t audio_format;
  uint16_t number_of_channels;
  uint32_t sample_rate;
  uint32_t byte_rate;
  uint16_t block_align;
  uint16_t bits_per_sample;
  uint32_t sub_chunk_2_id;
  uint32_t sub_chunk_2_size;
};

class wave_file_t {
public:
  wave_file_t(void) {
    memset(&m_header, 0, sizeof(m_header));
    m_header.chunk_id = RIFF_ASCII;
    m_header.format = WAVE_ASCII;
    m_header.sub_chunk_1_id = FMT_ASCII;
    m_header.sub_chunk_2_id = DATA_ASCII;
    m_header.audio_format = PCM;
    m_header.sub_chunk_1_size = DEFAULT_SUB_CHUNK_1_SIZE;
    m_samples.reserve(DEFAULT_RESERVE_VALUE);
  }

  // Takes a frequency domain and constructs wave samples for playback --
  // supports async loading of samples or not
  explicit wave_file_t(
      const size_t sample_size,
      const std::vector<std::complex<double>> &frequency_domain,
      bool async = true)
      : wave_file_t() {
    std::vector<double> time_domain;
    if (sample_size > DEFAULT_RESERVE_VALUE) {
      m_samples.reserve(sample_size);
    }
    time_domain.reserve(sample_size);
    if (async) {
      helper::inverse_discrete_fourier_transform_async(sample_size, time_domain,
                                                       frequency_domain);
    } else {
      helper::inverse_discrete_fourier_transform(sample_size, time_domain,
                                                 frequency_domain);
    }
    for (auto &time_domain_sample : time_domain) {
      m_samples.push_back(time_domain_sample);
    }
  }

  wave_file_t(const std::string &wav_file_path) {
    memset(&m_header, 0, sizeof(m_header));
    auto wave_file_handle = fopen(wav_file_path.c_str(), "rb");
    if (wave_file_handle) {
      const size_t file_size = std::filesystem::file_size(
          std::filesystem::absolute(wav_file_path.c_str()));
      uint8_t *temp_buffer = new uint8_t[file_size];
      memset(temp_buffer, 0, sizeof(temp_buffer));
      size_t bytes_read =
          fread(temp_buffer, sizeof(uint8_t), file_size, wave_file_handle);
      if (bytes_read == file_size) {
        const wave_header_t *unchecked_header =
            reinterpret_cast<wave_header_t *>(temp_buffer);
        if (!is_wave_header_valid(unchecked_header)) {
          size_t index{};
          std::vector<uint8_t> no_junk_buffer;
          for (index; index < EXTRA_PARAM_SIZE_OFFSET; index++) {
            no_junk_buffer.push_back(temp_buffer[index]);
          }
          no_junk_buffer.push_back(temp_buffer[EXTRA_PARAM_SIZE_OFFSET]);
          for (index; index < file_size; index++) {
            bool found = 0 == strncmp((char *)&temp_buffer[index], "data",
                                      strlen("data"));
            if (found) {
              break;
            }
          }
          for (index; index < file_size; index++) {
            no_junk_buffer.push_back(temp_buffer[index]);
          }
          delete[] temp_buffer; // We are going to delete this to create a new
                                // buffer the next time we delete temp_buffer it
                                // will be different
          temp_buffer = new uint8_t[no_junk_buffer.size()];
          memcpy(temp_buffer, no_junk_buffer.data(),
                 sizeof(uint8_t) * no_junk_buffer.size());
        }

        memcpy(&m_header, temp_buffer, sizeof(m_header));

        const size_t sample_size =
            m_header.sub_chunk_2_size / m_header.block_align;

        switch (m_header.bits_per_sample) {
        case _8_BITS_PER_SAMPLE: {
          int8_t *samples =
              reinterpret_cast<int8_t *>(&temp_buffer[sizeof(wave_header_t)]);
          for (size_t index = 0; index < sample_size; index++) {
            m_samples.push_back(static_cast<int32_t>(samples[index]));
          }
          break;
        }
        case _16_BITS_PER_SAMPLE: {
          const int16_t *samples =
              reinterpret_cast<int16_t *>(&temp_buffer[sizeof(wave_header_t)]);
          for (size_t index = 0; index < sample_size; index++) {
            m_samples.push_back(static_cast<int32_t>(samples[index]));
          }
          break;
        }
        case _24_BITS_PER_SAMPLE: {
          const size_t bytes_size = m_header.sub_chunk_2_size;
          const uint8_t *bytes = &temp_buffer[sizeof(wave_header_t)];
          int8_t count = 0;
          int64_t constructed_24_bit_sample{};
          for (size_t index = 0; index < bytes_size; index++) {
            if (count == 0) {
              constructed_24_bit_sample |= bytes[index];
            } else {
              constructed_24_bit_sample |= bytes[index]
                                           << ((BITS_PER_BYTE * count));
            }
            if (count == 2) {
              count = 0;
              m_samples.push_back(std::byteswap(constructed_24_bit_sample));
              constructed_24_bit_sample = 0;
              continue;
            }
            count += 1;
          }
          break;
        }
        case _32_BITS_PER_SAMPLE: {
          const int32_t *samples =
              reinterpret_cast<int32_t *>(&temp_buffer[sizeof(wave_header_t)]);
          for (size_t index = 0; index < sample_size; index++) {
            m_samples.push_back(samples[index]);
          }
          break;
        }
        default:
          break;
        }
      }
      delete[] temp_buffer;
    }
  }

  static constexpr bool is_wave_header_valid(
      const wave_header_t
          *header) { // Helper function to validate a wav file header
    return header->chunk_id == RIFF_ASCII && header->format == WAVE_ASCII &&
           header->sub_chunk_1_id == FMT_ASCII &&
           header->sub_chunk_2_id == DATA_ASCII &&
           (header->audio_format != 0) && (header->bits_per_sample > 0) &&
           (header->number_of_channels > 0) && (header->sample_rate > 0) &&
           (header->byte_rate > 0) && (header->chunk_size > 0) &&
           (header->sub_chunk_1_size == DEFAULT_SUB_CHUNK_1_SIZE) &&
           (header->sub_chunk_2_size > 0);
  }

  operator bool() { return is_wave_header_valid(&m_header); }

  constexpr uint32_t get_nyquist_frequency(void) const {
    if (0 == m_header.sample_rate) {
      throw std::invalid_argument("Sample rate of 0 is invalid and thus no "
                                  "nyquist frequency for you!!");
    }
    return m_header.sample_rate / 2;
  }

  constexpr std::optional<int32_t> operator[](size_t index) const {
    if (index < m_samples.size()) {
      return std::make_optional<int32_t>(m_samples[index]);
    }
    return std::nullopt;
  }

  // Indexes and gets the value as a float regardless if the wav file isn't
  // saved as floating point PCM data
  constexpr std::optional<float> index_as_float(const size_t &index) const {
    if (index < m_samples.size()) {
      const float max_sample_value =
          static_cast<float>(get_maximum_sample_value());
      return std::make_optional<float>(
          (static_cast<float>(m_samples[index]) / max_sample_value));
    }
    return std::nullopt;
  }

  // Same as index_as_float but returns doubles
  constexpr std::optional<double> index_as_double(const size_t &index) const {
    if (index < m_samples.size()) {
      const double max_sample_value =
          static_cast<double>(get_maximum_sample_value());
      return std::make_optional<double>(
          (static_cast<double>(m_samples[index]) / max_sample_value));
    }
    return std::nullopt;
  }

  // Same as index_as_float but returns the samples dB full scale
  constexpr std::optional<double> index_as_dBFS(const size_t &index) const {
    if (index < m_samples.size()) {
      const double max_sample_value =
          static_cast<double>(get_maximum_sample_value());
      return std::make_optional<double>(
          helper::get_decibel_fullscale_from_sample(m_samples[index],
                                                    max_sample_value));
    }
    return std::nullopt;
  }

  // Provides a safe way to generate the next sample and wrap around back to the
  // beginning -- maybe a non circular option can also be an option?
  int32_t pcm_sink(void) {
    if (m_samples.empty()) {
      return 0;
    }
    auto &next_value = m_samples[m_sink_index];
    ++m_sink_index %= m_samples.size();
    return next_value;
  }

  // Same behavior as pcm_sink but returns PCM float values in the range of
  // [-1.0, 1.0]
  float pcm_float_sink(void) {
    if (m_samples.empty()) {
      return 0;
    }
    const float max_sample_value =
        static_cast<float>(get_maximum_sample_value());
    float next_value =
        static_cast<float>(m_samples[m_sink_index]) / max_sample_value;
    ++m_sink_index %= m_samples.size();
    return next_value;
  }

  // Same behavior as pcm_sink but returns PCM double values in the range of
  // [-1.0, 1.0]
  double pcm_double_sink(void) {
    if (m_samples.empty()) {
      return 0;
    }
    const double max_sample_value =
        static_cast<double>(get_maximum_sample_value());
    double next_value =
        static_cast<double>(m_samples[m_sink_index]) / max_sample_value;
    ++m_sink_index %= m_samples.size();
    return next_value;
  }

  // Just looks at the max PCM sample and converts it to dB FS for example peak
  // of wav file could be -20 db FS (pretty low sounding relative to a 0 dB FS
  // signal)
  double get_peak_decibel_fullscale_of_signal(void) {
    int64_t peak_sample = INT64_MIN;
    const int32_t max_sample_value =
        static_cast<int32_t>(get_maximum_sample_value());
    for (auto &sample : m_samples) {
      if (sample > peak_sample) {
        peak_sample = sample;
      }
    }
    return helper::get_decibel_fullscale_from_sample(peak_sample,
                                                     max_sample_value);
  }

  size_t sample_size(void) const { return m_samples.size(); }

  void set_sample_rate(const int32_t sample_rate) {
    m_header.sample_rate = sample_rate;
  }
  void set_number_of_channels(const int16_t channel_count) {
    m_header.number_of_channels = channel_count;
  }
  void set_bits_per_sample(const int16_t bits_per_sample) {
    m_header.bits_per_sample = bits_per_sample;
  }
  const wave_header_t &get_header(void) const { return m_header; }

  bool add_24_32_bits_sample(const int64_t sample, ...) {
    va_list args;
    if (m_header.number_of_channels > 1) {
      va_start(args, sample);
    }
    m_samples.push_back(sample);
    for (size_t _ = 1;
         m_header.number_of_channels > 1 && _ < m_header.number_of_channels;
         _++) {
      m_samples.push_back(va_arg(args, const int64_t));
    }
    if (m_header.number_of_channels > 1) {
      va_end(args);
    }
    return true;
  }

  bool add_16_bits_sample(const int16_t sample, ...) {
    va_list args;
    if (m_header.number_of_channels > 1) {
      va_start(args, sample);
    }
    m_samples.push_back(static_cast<int32_t>(sample));
    for (size_t _ = 1;
         m_header.number_of_channels > 1 && _ < m_header.number_of_channels;
         _++) {
      m_samples.push_back(static_cast<int32_t>(va_arg(args, const int16_t)));
    }
    if (m_header.number_of_channels > 1) {
      va_end(args);
    }
    return true;
  }

  bool add_8_bits_sample(const int8_t sample, ...) {
    va_list args;
    if (m_header.number_of_channels > 1) {
      va_start(args, sample);
    }
    m_samples.push_back(static_cast<int32_t>(sample));
    for (size_t _ = 1;
         m_header.number_of_channels > 1 && _ < m_header.number_of_channels;
         _++) {
      m_samples.push_back(static_cast<int32_t>(va_arg(args, const int8_t)));
    }
    if (m_header.number_of_channels > 1) {
      va_end(args);
    }
    return true;
  }

  bool generate_synth(const size_t sample_size, const double volume_percent,
                      synth_config_t &configuration) {

    if (m_header.sample_rate == 0) {
      m_lfo_terminate.store(true);
      return false;
    }

    if (configuration.empty()) {
      m_lfo_terminate.store(true);
      return false;
    }

    const size_t max_sample_value = get_maximum_sample_value();

    const double volume = helper::set_volume(volume_percent, max_sample_value);

    const bool is_stereo = m_header.number_of_channels == 2;

    const uint32_t sample_rate = m_header.sample_rate;

    std::vector<int32_t> output;
    output.reserve(sample_size);

    const auto selected_oscs_a = {
        &configuration.oscillator_b, &configuration.oscillator_c,
        &configuration.oscillator_d, &configuration.oscillator_e,
        &configuration.oscillator_f, &configuration.oscillator_g};
    const auto selected_oscs_b = {
        &configuration.oscillator_a, &configuration.oscillator_c,
        &configuration.oscillator_d, &configuration.oscillator_e,
        &configuration.oscillator_f, &configuration.oscillator_g};
    const auto selected_oscs_c = {
        &configuration.oscillator_a, &configuration.oscillator_b,
        &configuration.oscillator_d, &configuration.oscillator_e,
        &configuration.oscillator_f, &configuration.oscillator_g};
    const auto selected_oscs_d = {
        &configuration.oscillator_a, &configuration.oscillator_c,
        &configuration.oscillator_b, &configuration.oscillator_e,
        &configuration.oscillator_f, &configuration.oscillator_g};
    const auto selected_oscs_e = {
        &configuration.oscillator_a, &configuration.oscillator_c,
        &configuration.oscillator_d, &configuration.oscillator_b,
        &configuration.oscillator_f, &configuration.oscillator_g};
    const auto selected_oscs_f = {
        &configuration.oscillator_a, &configuration.oscillator_c,
        &configuration.oscillator_d, &configuration.oscillator_e,
        &configuration.oscillator_b, &configuration.oscillator_g};
    const auto selected_oscs_g = {
        &configuration.oscillator_a, &configuration.oscillator_c,
        &configuration.oscillator_d, &configuration.oscillator_e,
        &configuration.oscillator_f, &configuration.oscillator_b};

    auto osc_a_carrier_future =
        std::async(std::launch::async,
                   &processing_functions::oscillator_processing_callback,
                   oscillator_selection_t::oscillator_a, std::ref(sample_size),
                   std::ref(volume), std::ref(is_stereo), std::ref(sample_rate),
                   std::ref(configuration), selected_oscs_a);
    
    auto osc_b_carrier_future =
        std::async(std::launch::async,
                   &processing_functions::oscillator_processing_callback,
                   oscillator_selection_t::oscillator_b, std::ref(sample_size),
                   std::ref(volume), std::ref(is_stereo), std::ref(sample_rate),
                   std::ref(configuration), selected_oscs_b);
    
    auto osc_c_carrier_future =
        std::async(std::launch::async,
                   &processing_functions::oscillator_processing_callback,
                  oscillator_selection_t::oscillator_c, std::ref(sample_size),
                   std::ref(volume), std::ref(is_stereo), std::ref(sample_rate),
                   std::ref(configuration), selected_oscs_c);
    
    auto osc_d_carrier_future =
        std::async(std::launch::async,
                   &processing_functions::oscillator_processing_callback,
                   oscillator_selection_t::oscillator_d, std::ref(sample_size),
                   std::ref(volume), std::ref(is_stereo), std::ref(sample_rate),
                   std::ref(configuration), selected_oscs_d);
    
    auto osc_e_carrier_future =
        std::async(std::launch::async,
                   &processing_functions::oscillator_processing_callback,
                   oscillator_selection_t::oscillator_e, std::ref(sample_size),
                   std::ref(volume), std::ref(is_stereo), std::ref(sample_rate),
                   std::ref(configuration), selected_oscs_e);
    
    auto osc_f_carrier_future =
        std::async(std::launch::async,
                   &processing_functions::oscillator_processing_callback,
                   oscillator_selection_t::oscillator_f, std::ref(sample_size),
                   std::ref(volume), std::ref(is_stereo), std::ref(sample_rate),
                   std::ref(configuration), selected_oscs_f);
    
    auto osc_g_carrier_future =
        std::async(std::launch::async,
                   &processing_functions::oscillator_processing_callback,
                   oscillator_selection_t::oscillator_g, std::ref(sample_size),
                   std::ref(volume), std::ref(is_stereo), std::ref(sample_rate),
                   std::ref(configuration), selected_oscs_g);

    auto wave_a = osc_a_carrier_future.get();
    auto wave_b = osc_b_carrier_future.get();
    auto wave_c = osc_c_carrier_future.get();
    auto wave_d = osc_d_carrier_future.get();
    auto wave_e = osc_e_carrier_future.get();
    auto wave_f = osc_f_carrier_future.get();
    auto wave_g = osc_g_carrier_future.get();

    if (configuration.apply_bitcrusher_effect) {
      m_apply_bitcrusher_effect = true;
      m_bitcrusher_gain_value = configuration.bitcrusher_gain_value;
      if (m_bitcrusher_gain_value < 0.0 || m_bitcrusher_gain_value > 1.0) {
        m_bitcrusher_gain_value = 1.0;
      }
      m_bitcrusher_wet_percent = configuration.bitcrusher_wet_percentage;
    } else {
      m_apply_bitcrusher_effect = false;
      m_bitcrusher_wet_percent = 1.0;
    }

    if (m_apply_bitcrusher_effect) {
      const auto &lfo = configuration.lfo;
      if (effects_type_t::bitcrusher_wet_percentage == lfo.effect_to_modulate) {
        const double &frequency = lfo.frequency;
        const wave_type_t &wave_type = lfo.wave_type;
        const bool is_lfo = frequency <= 20.0; // 20hz or less is considered lfo
        apply_osc_to_bitcrusher_wet_value(frequency, wave_type, is_lfo);
      } else if (effects_type_t::bitcrusher_gain_value ==
                 lfo.effect_to_modulate) {
        const double &frequency = lfo.frequency;
        const wave_type_t &wave_type = lfo.wave_type;
        const bool is_lfo = frequency <= 20.0; // 20hz or less is considered lfo
        apply_osc_to_bitcrusher_gain_value(frequency, wave_type, is_lfo);
      }
      m_start_lfo_latch.count_down();
    }

    for (size_t index{}; index < sample_size; index++) {
      int64_t sample{};
      double frequency_offset = 0.0;

      if (index < wave_a.size()) {
        sample += wave_a[index];
      }
      if (index < wave_b.size()) {
        sample += wave_b[index];
      }
      if (index < wave_c.size()) {
        sample += wave_c[index];
      }
      if (index < wave_d.size()) {
        sample += wave_d[index];
      }
      if (index < wave_e.size()) {
        sample += wave_e[index];
      }
      if (index < wave_f.size()) {
        sample += wave_f[index];
      }
      if (index < wave_g.size()) {
        sample += wave_g[index];
      }

      switch (m_header.bits_per_sample) {
      case _8_BITS_PER_SAMPLE:
        !is_stereo ? add_8_bits_sample(
                         m_apply_bitcrusher_effect ? (sample & 0x0f) : sample)
                   : add_8_bits_sample(
                         m_apply_bitcrusher_effect ? (sample & 0x0f) : sample,
                         m_apply_bitcrusher_effect ? (sample & 0x0f) : sample);
        break;
      case _16_BITS_PER_SAMPLE:
        !is_stereo
            ? add_16_bits_sample(
                  m_apply_bitcrusher_effect
                      ? (m_bitcrusher_gain_value * (sample % (INT16_MAX / 4)))
                      : sample)
            : add_16_bits_sample(
                  m_apply_bitcrusher_effect
                      ? (m_bitcrusher_gain_value * (sample % (INT16_MAX / 4)))
                      : sample,
                  m_apply_bitcrusher_effect
                      ? (m_bitcrusher_gain_value * (sample % (INT16_MAX / 4)))
                      : sample);
        break;
      case _24_BITS_PER_SAMPLE:
        !is_stereo
            ? add_24_32_bits_sample(
                  m_apply_bitcrusher_effect
                      ? (m_bitcrusher_gain_value * (sample % (INT24_MAX / 8)))
                      : sample)
            : add_24_32_bits_sample(
                  m_apply_bitcrusher_effect
                      ? (m_bitcrusher_gain_value * (sample % (INT24_MAX / 8)))
                      : sample,
                  m_apply_bitcrusher_effect
                      ? (m_bitcrusher_gain_value * (sample % (INT24_MAX / 8)))
                      : sample);
        break;
      case _32_BITS_PER_SAMPLE:
        !is_stereo
            ? add_24_32_bits_sample(
                  m_apply_bitcrusher_effect
                      ? (m_bitcrusher_gain_value *
                         (sample %
                          (static_cast<int64_t>(static_cast<double>(INT32_MAX) *
                                                m_bitcrusher_wet_percent))))
                      : sample)
            : add_24_32_bits_sample(
                  m_apply_bitcrusher_effect
                      ? (m_bitcrusher_gain_value *
                         (sample %
                          (static_cast<int64_t>(static_cast<double>(INT32_MAX) *
                                                m_bitcrusher_wet_percent))))
                      : sample,
                  m_apply_bitcrusher_effect
                      ? (m_bitcrusher_gain_value *
                         (sample %
                          (static_cast<int64_t>(static_cast<double>(INT32_MAX) *
                                                m_bitcrusher_wet_percent))))
                      : sample);
        break;
      default:
        m_lfo_terminate.store(true);
        return false;
      }
    }

    m_lfo_terminate.store(true);
    return true;
  }

  bool generate_wave(uint8_t wave_type, size_t sample_size, double frequency,
                     double volume_percent) {

    if (m_header.sample_rate == 0) {
      m_lfo_terminate.store(true);
      return false;
    }

    const bool is_stereo = m_header.number_of_channels == 2;
    const double phase{};
    double time{};

    size_t wave_count =
        static_cast<size_t>((wave_type & wave_type_t::sine) ==
                            (wave_type_t::sine)) +
        static_cast<size_t>((wave_type & wave_type_t::triangle) ==
                            (wave_type_t::triangle)) +
        static_cast<size_t>((wave_type & wave_type_t::square) ==
                            (wave_type_t::square)) +
        static_cast<size_t>((wave_type & wave_type_t::sawtooth) ==
                            (wave_type_t::sawtooth));

    const size_t max_sample_value = get_maximum_sample_value();

    const double sample_rate = static_cast<double>(m_header.sample_rate);

    const double volume = helper::set_volume(
        volume_percent / static_cast<double>(wave_count), max_sample_value);

    m_start_lfo_latch.count_down();

    for (size_t _ = 0ul; _ < sample_size; _++) {
      int64_t sample{};
      if ((wave_type & wave_type_t::sine)) {
        sample += helper::pcm_sine(frequency, time, volume, phase);
      }
      if ((wave_type & wave_type_t::triangle)) {
        sample += helper::pcm_triangle(time, volume, frequency);
      }
      if ((wave_type & wave_type_t::square)) {
        sample += helper::pcm_square(time, volume, frequency);
      }
      if ((wave_type & wave_type_t::sawtooth)) {
        sample += helper::pcm_saw_tooth(time, volume, frequency);
      }
      switch (m_header.bits_per_sample) {
      case _8_BITS_PER_SAMPLE:
        !is_stereo ? add_8_bits_sample(
                         m_apply_bitcrusher_effect ? (sample & 0x0f) : sample)
                   : add_8_bits_sample(
                         m_apply_bitcrusher_effect ? (sample & 0x0f) : sample,
                         m_apply_bitcrusher_effect ? (sample & 0x0f) : sample);
        break;
      case _16_BITS_PER_SAMPLE:
        !is_stereo
            ? add_16_bits_sample(
                  m_apply_bitcrusher_effect
                      ? (m_bitcrusher_gain_value * static_cast<int8_t>(sample))
                      : sample)
            : add_16_bits_sample(
                  m_apply_bitcrusher_effect
                      ? (m_bitcrusher_gain_value * static_cast<int8_t>(sample))
                      : sample,
                  m_apply_bitcrusher_effect
                      ? (m_bitcrusher_gain_value * static_cast<int8_t>(sample))
                      : sample);
        break;
      case _24_BITS_PER_SAMPLE:
        !is_stereo
            ? add_24_32_bits_sample(
                  m_apply_bitcrusher_effect
                      ? (m_bitcrusher_gain_value * static_cast<int16_t>(sample))
                      : sample)
            : add_24_32_bits_sample(
                  m_apply_bitcrusher_effect
                      ? (m_bitcrusher_gain_value * static_cast<int16_t>(sample))
                      : sample,
                  m_apply_bitcrusher_effect
                      ? (m_bitcrusher_gain_value * static_cast<int16_t>(sample))
                      : sample);
        break;
      case _32_BITS_PER_SAMPLE:
        !is_stereo
            ? add_24_32_bits_sample(
                  m_apply_bitcrusher_effect
                      ? (m_bitcrusher_gain_value *
                         (sample %
                          static_cast<int64_t>(static_cast<double>(INT32_MAX) *
                                               m_bitcrusher_wet_percent)))
                      : sample)
            : add_24_32_bits_sample(
                  m_apply_bitcrusher_effect
                      ? (m_bitcrusher_gain_value *
                         (sample %
                          static_cast<int64_t>(static_cast<double>(INT32_MAX) *
                                               m_bitcrusher_wet_percent)))
                      : sample,
                  m_apply_bitcrusher_effect
                      ? (m_bitcrusher_gain_value *
                         (sample %
                          static_cast<int64_t>(static_cast<double>(INT32_MAX) *
                                               m_bitcrusher_wet_percent)))
                      : sample);
        break;
      default:
        m_lfo_terminate.store(true);
        return false;
      }
      time += (1.0 / sample_rate);
    }

    m_lfo_terminate.store(true);
    return true;
  }

  bool save(const std::string &file_path) {
    m_header.sub_chunk_2_size =
        (m_samples.size() * m_header.number_of_channels *
         (m_header.bits_per_sample / BITS_PER_BYTE));
    m_header.block_align = (m_header.number_of_channels *
                            (m_header.bits_per_sample / BITS_PER_BYTE));
    m_header.byte_rate = (m_header.sample_rate * m_header.number_of_channels *
                          (m_header.bits_per_sample / BITS_PER_BYTE));
    m_header.chunk_size = 36 + m_header.sub_chunk_2_size;
    switch (m_header.bits_per_sample) {
    case _8_BITS_PER_SAMPLE: {
      return save_as_8_bits(file_path);
    }
    case _16_BITS_PER_SAMPLE: {
      return save_as_16_bits(file_path);
    }
    case _24_BITS_PER_SAMPLE: {
      return save_as_24_bits(file_path);
    }
    case _32_BITS_PER_SAMPLE: {
      return save_as_32_bits(file_path);
    }
    default: {
      return false;
    }
    }
  }

  std::vector<std::complex<double>> get_frequency_domain(size_t sample_size,
                                                         bool async) {
    std::vector<std::complex<double>> frequency_domain;
    std::vector<double> time_domain;
    time_domain.reserve(m_samples.size());
    for (auto &sample : m_samples) {
      time_domain.push_back(static_cast<double>(sample));
    }
    if (async) {
      helper::discrete_fourier_transform_async(sample_size, time_domain,
                                               frequency_domain);
    } else {
      helper::discrete_fourier_transform(sample_size, time_domain,
                                         frequency_domain);
    }
    return frequency_domain;
  }

  // Second parameter set to false if you want the bitmap to represent the true
  // scale of the wavefile one sample is one pixel so its expensive but fun :)
  // Third parameter if set to true (false is default) then draws the waveform
  // shaded else it plots it without shading underneath the waveform
  // Fourth parameter directs a pc screen font file path to also render the text (fifth) if empty string then displa_filename is treated as if it were false
  bool save_waveform_as_monochrome_bmp(const std::string &file_path,
                                       bool scale_down = true,
                                       bool shade_waveform = false, bool display_filename = false, 
                                       const char* pc_screenfont_file_path = nullptr, const char* text = nullptr) {
    if (file_path.empty()) {
      return false;
    }
    if (m_samples.empty()) {
      return false;
    }

    constexpr const size_t default_width{2048};
    const size_t width =
        scale_down
            ? default_width
            : (m_samples.size() % static_cast<size_t>(m_header.sample_rate) +
               static_cast<size_t>(m_header.sample_rate));
    const size_t height{128};

    std::future<std::vector<std::vector<bool>>> bitmap_future =
        std::async(std::launch::async, [&]() {
          std::vector<std::vector<bool>> bitmap;
          for (size_t _ = 0; _ < height; _++) {
            bitmap.emplace_back(width, false);
          }
          const size_t true_width =
              (m_samples.size() % static_cast<size_t>(m_header.sample_rate) +
               static_cast<size_t>(m_header.sample_rate));
          const size_t max_sample_count = !scale_down ? width : true_width;
          for (size_t column = 0; column < max_sample_count; column++) {
            size_t true_column =
                !scale_down ? column
                            : static_cast<size_t>(
                                  static_cast<double>(width - 1) *
                                  (static_cast<double>(column) /
                                   static_cast<double>(max_sample_count)));
            size_t row = static_cast<size_t>(
                static_cast<double>(height) *
                ((index_as_double(column).value_or(0.0) + 1.0) / 2.0));
            if (row >= height) {
              continue;
            }

            if (!shade_waveform) {
              // Plot the main pixel and surrounding pixels to make it thicker
              if (row > 0) {
                bitmap[row - 1][true_column] = true;
              }
              if (row > 0 && true_column > 0) {
                bitmap[row - 1][true_column - 1] = true;
              }
              bitmap[row][true_column] = true;
              if ((row + 1) < height) {
                bitmap[row + 1][true_column] = true;
              }
              if ((true_column + 1) < max_sample_count) {
                bitmap[row][true_column + 1] = true;
              }
              if (((row + 1) < height) &&
                  ((true_column + 1) < max_sample_count)) {
                bitmap[row + 1][true_column + 1] = true;
              }
            } else {
              // Just shade the waveform below
              for (row += 1; row < height; row++) {
                bitmap[row][true_column] = true;
              }
            }
          }

          if (display_filename) {
            psf_header_t font_header{};
            std::vector<uint8_t> glyphs_raw_buffer;
            if (load_psf2_font(pc_screenfont_file_path, font_header, glyphs_raw_buffer)) {
                if (nullptr != text) {
                    size_t row_offset{4}; // y
                    size_t column_offset{0}; // x
                    for(size_t index = 0; index < strlen(text); index++) {
                        const size_t char_index{static_cast<size_t>(text[index])};
                        if (char_index >= font_header.glyph_size) {
                            continue;
                        }
                        const size_t glyph_index{char_index * font_header.bytes_per_glyph};
                        const size_t ending_index{glyph_index + font_header.bytes_per_glyph}; 
                        if (ending_index >= glyphs_raw_buffer.size()) {
                            continue;
                        }
                        if (font_header.bytes_per_glyph <= sizeof(uint64_t)) {
                            std::bitset<sizeof(uint64_t)> glyph_as_bitset((glyphs_raw_buffer[glyph_index + 3] << 24)
                               | (glyphs_raw_buffer[glyph_index + 2] << 16) | (glyphs_raw_buffer[glyph_index + 1] << 8) 
                                | glyphs_raw_buffer[glyph_index]);
                                //TODO: Index bitset and see which bits are set to set the pixels in the bitmap
                        }
                        row_offset += font_header.glyph_height;
                        column_offset += font_header.glyph_width;                        
                    }
                }
            }
          }

          return bitmap;
        });

    std::vector<uint8_t> bytes;
    bytes.reserve(width * height * 4);
    bitmap_header_t header{};
    header.magic_field[0] = 'B';
    header.magic_field[1] = 'M';
    header.offset_to_pixels = sizeof(bitmap_header_t);
    header.header_size = 40;
    header.width = width;
    header.height = height;
    header.plane_count = 1;
    header.bits_per_pixel = 32;
    header.compression_method =
        0; // No compression just store the 3 byte rgb values
    header.horizontal_resolution = 1;
    header.vertical_resolution = 1;
    header.color_pallete_count = 0;
    header.important_colors_used = 0;

    auto waveform_file_handle = fopen(file_path.c_str(), "wb");

    if (nullptr == waveform_file_handle) {
      return false;
    }

    header.data_size = sizeof(bitmap_header_t) + (width * height * 4);

    size_t expected_bytes = sizeof(header);
    size_t written_bytes =
        fwrite(reinterpret_cast<uint8_t *>(&header), sizeof(uint8_t),
               sizeof(header), waveform_file_handle);

    if (expected_bytes != written_bytes) {
      return false;
    }

    auto bitmap = bitmap_future.get();

    bool flip{false}; // Used to alternate color pixel

    for (auto &row : bitmap) {
      for (bool column : row) {
        if (column) { // If this x y is set to true then we insert a black pixel
                      // (0x00 0x00 0x00 0xFF)
          bytes.push_back(0x00);               // blue
          bytes.push_back(flip ? 0xff : 0x00); // green
          bytes.push_back(0x00);               // red
          bytes.push_back(0xff);               // alpha
          if (!shade_waveform) {
            flip ^= true;
          }
        } else { // Draw default white pixel (0xff 0xff 0xff 0xff)
          bytes.push_back(0xff); // blue
          bytes.push_back(0xff); // green
          bytes.push_back(0xff); // red
          bytes.push_back(0xff); // alpha
        }
      }
    }

    expected_bytes += bytes.size();
    written_bytes += fwrite(bytes.data(), sizeof(uint8_t), bytes.size(),
                            waveform_file_handle);

    return expected_bytes == written_bytes;
  }

  bool save_waveform_as_grayscale_pbm(
      const std::string
          &file_path) { // Function is slow even as async use above function
                        // windows bitmap -- this is just for legacy fun :)

    if (file_path.empty()) {
      return false;
    }

    if (m_samples.empty()) {
      return false;
    }

    const size_t width =
        m_samples.size() % static_cast<size_t>(m_header.sample_rate) +
        static_cast<size_t>(m_header.sample_rate);
    const size_t height{128};

    std::future<std::vector<std::vector<bool>>> bitmap_future =
        std::async(std::launch::async, [&]() {
          std::vector<std::vector<bool>> bitmap;
          for (size_t _ = 0; _ < height; _++) {
            bitmap.emplace_back(width, false);
          }
          for (size_t column = 0; column < width; column++) {
            size_t row = static_cast<size_t>(
                static_cast<double>(height) *
                ((index_as_double(column).value_or(-1.0) + 1.0) / 2.0));
            if (row >= height) {
              continue;
            }
            bitmap[row][column] = true;
            for (row += 1; row < height; row++) {
              bitmap[row][column] = true;
            }
          }
          return bitmap;
        });

    auto bitmap = bitmap_future.get();
    std::vector<std::future<std::string>> column_futures;
    for (size_t row = 0; row < height; row++) {
      column_futures.push_back(
          std::async(std::launch::async, [&width, row, &bitmap]() {
            std::string result;
            for (size_t column = 0; column < width; column++) {
              result += std::format("{} ", bitmap[row][column] ? "1" : "0");
            }
            result += "\n";
            return result;
          }));
    }

    std::string pbm_header =
        std::format("P1\n#Generated by wave_t on {}\n{} {}\n",
                    static_cast<size_t>(time(nullptr)), width, height);
    std::stringstream pbm_data;
    pbm_data << pbm_header;

    for (auto &future : column_futures) {
      pbm_data << future.get();
    }

    auto data = pbm_data.str();
    auto waveform_file_handle = fopen(file_path.c_str(), "wb");

    if (nullptr == waveform_file_handle) {
      return false;
    }

    const size_t expected_bytes = data.size();
    size_t written_bytes =
        fwrite(data.data(), sizeof(int8_t), data.size(), waveform_file_handle);

    fclose(waveform_file_handle);

    return expected_bytes == written_bytes;
  }

#ifdef DEBUG

  std::string get_readable_wave_header(void) {
    std::stringstream header;
    header << "chunk_id=0x" << std::hex << std::byteswap(m_header.chunk_id)
           << std::endl;
    header << "chunk_size=" << std::dec << m_header.chunk_size << std::endl;
    header << "format=0x" << std::hex << std::byteswap(m_header.format)
           << std::endl;
    header << "sub_chunk_1_id=0x" << std::byteswap(m_header.sub_chunk_1_id)
           << std::endl;
    header << "sub_chunk_1_size=" << std::dec << m_header.sub_chunk_1_size
           << std::endl;
    header << "audio_format="
           << ((m_header.audio_format == 1) ? "PCM" : "Unknown") << std::endl;
    header << "number_of_channels=" << m_header.number_of_channels << std::endl;
    header << "sample_rate=" << m_header.sample_rate << std::endl;
    header << "byte_rate=" << m_header.byte_rate << std::endl;
    header << "block_align=" << m_header.block_align << std::endl;
    header << "bits_per_sample=" << m_header.bits_per_sample << std::endl;
    header << "sub_chunk_2_id=0x" << std::hex
           << std::byteswap(m_header.sub_chunk_2_id) << std::endl;
    header << "sub_chunk_2_size=" << std::dec << m_header.sub_chunk_2_size
           << std::endl;
    return header.str();
  }

#endif

private:
  static constexpr uint32_t PSF_FONT_MAGIC{0x72b54a86}; // Used to validate font header's magic number
  // Used for saving wave forms as bitmaps -- used internally
  // Source: https://en.wikipedia.org/wiki/BMP_file_format
  struct bitmap_header_t {
    char magic_field[2];
    uint32_t bitmap_total_size;
    uint32_t reserved;
    uint32_t offset_to_pixels;
    uint32_t header_size;
    int32_t width;
    int32_t height;
    uint16_t plane_count;
    uint16_t bits_per_pixel;
    uint32_t compression_method;
    uint32_t data_size;
    uint32_t horizontal_resolution;
    uint32_t vertical_resolution;
    uint32_t color_pallete_count;
    uint32_t important_colors_used;
  } __attribute__((packed));
  
  //Source: https://en.wikipedia.org/wiki/PC_Screen_Font
  // Used to render text when rendering waveform -- used internally 
  struct psf_header_t {
        uint32_t magic;
        uint32_t version;
        uint32_t this_header_size; // Note: Will be always set to sizeof(psf_header_t) 
        uint32_t has_unicode_table;
        uint32_t glyph_size; // 
        uint32_t bytes_per_glyph;
        uint32_t glyph_height;
        uint32_t glyph_width;

  }; // We don't need to pack hurray or could we?

  // Used for saving 24-bit audio -- custom type
  struct int24_t {
    int8_t byte1;
    int8_t byte2;
    int8_t byte3;
  }; // Same as here -- do we oack here?

  //Internal helper function -- second and third parameter are out parameter please check return value before checking those out values
  static bool load_psf2_font(const char* file_path, psf_header_t& header, std::vector<uint8_t>& glyphs_raw_data) {

    if (nullptr == file_path) {
#ifdef DEBUG
      std::cout << "Passed null file path!!" << std::endl;
#endif
        return false;
    }
    
    auto psf_file_handle = fopen(file_path, "rb");
    
    if (nullptr == psf_file_handle) {
#ifdef DEBUG
      std::cout << "Failed to open PC screen font!!" << std::endl;
#endif
      return false;
    }

#ifdef DEBUG
    std::cout << "\033[01;34mLoaded PC screen font at " << file_path << std::endl;
#endif
    
    constexpr const size_t max_file_size{65536};
    
    glyphs_raw_data.reserve(max_file_size);
    
    uint8_t* temp_buffer = new uint8_t[max_file_size]; // Don't think pc screen fonts go behind 2K
    memset(temp_buffer, '\0', sizeof(temp_buffer));

    size_t bytes_read =
          fread(temp_buffer, sizeof(uint8_t), max_file_size, psf_file_handle);

    memcpy(&header, reinterpret_cast<psf_header_t*>(temp_buffer), sizeof(psf_header_t));

#ifdef DEBUG

    std::cout << "glyph_size=" << header.glyph_size << std::endl;
    std::cout << "bytes_per_glyph=" << header.bytes_per_glyph << std::endl;
    std::cout << "glyph_height=" << header.glyph_height << std::endl;
    std::cout << "glyph_width=" << header.glyph_width << "\033[0m" << std::endl;
#endif

    //Starting after the header so we can load those glyphs into a vector (much more friendler)
    for(size_t index = sizeof(psf_header_t); index < bytes_read; index++) {
        glyphs_raw_data.push_back(temp_buffer[index]);
    }
    
    fclose(psf_file_handle);
    
    delete[] temp_buffer;

    return bytes_read > 0;
  }

  static constexpr int24_t make_int24_t(const int32_t &value) {
    int24_t _24_bit_value{};
    // Grab the 3 bytes within the 32-bit signed value
    _24_bit_value.byte1 = static_cast<int8_t>(value & 0xff);
    _24_bit_value.byte2 =
        static_cast<int8_t>((value & 0xff00) >> BITS_PER_BYTE);
    _24_bit_value.byte3 =
        static_cast<int8_t>((value & 0xff0000) >> (BITS_PER_BYTE * 2));
    return _24_bit_value;
  }

  bool save_as_8_bits(const std::string &file_path) {
    std::vector<int8_t> samples;
    samples.reserve(m_samples.size());
    for (auto &sample : m_samples) {
      samples.push_back(static_cast<int8_t>(sample));
    }

    auto wav_file_handle = fopen(file_path.c_str(), "wb");
    if (nullptr == wav_file_handle) {
      return false;
    }

    const size_t expected_bytes = sizeof(m_header) + samples.size();
    size_t written_bytes =
        fwrite(reinterpret_cast<int8_t *>(&m_header), sizeof(int8_t),
               sizeof(m_header), wav_file_handle);
    written_bytes +=
        fwrite(samples.data(), sizeof(int8_t), samples.size(), wav_file_handle);

    fclose(wav_file_handle);

    return written_bytes == expected_bytes;
  }

  bool save_as_16_bits(const std::string &file_path) {
    std::vector<int16_t> samples;
    samples.reserve(m_samples.size());
    for (auto &sample : m_samples) {
      samples.push_back(static_cast<int16_t>(sample));
    }

    auto wav_file_handle = fopen(file_path.c_str(), "wb");
    if (nullptr == wav_file_handle) {
      return false;
    }

    const size_t expected_bytes =
        sizeof(m_header) + (sizeof(int16_t) * samples.size());
    size_t written_bytes =
        fwrite(reinterpret_cast<int8_t *>(&m_header), sizeof(int8_t),
               sizeof(m_header), wav_file_handle);
    written_bytes += (fwrite(samples.data(), sizeof(int16_t), samples.size(),
                             wav_file_handle) *
                      sizeof(int16_t));
    fclose(wav_file_handle);
    return written_bytes == expected_bytes;
  }

  bool save_as_24_bits(const std::string &file_path) {
    constexpr const size_t bytes{3};
    std::vector<int8_t> samples;
    samples.reserve(bytes * m_samples.size());

    for (auto &sample : m_samples) {
      const int24_t value = make_int24_t(sample);
      samples.push_back(value.byte1);
      samples.push_back(value.byte2);
      samples.push_back(value.byte3);
    }

    auto wav_file_handle = fopen(file_path.c_str(), "wb");
    if (nullptr == wav_file_handle) {
      return false;
    }

    const size_t expected_bytes = sizeof(m_header) + samples.size();
    size_t written_bytes =
        fwrite(reinterpret_cast<int8_t *>(&m_header), sizeof(int8_t),
               sizeof(m_header), wav_file_handle);
    written_bytes +=
        fwrite(samples.data(), sizeof(int8_t), samples.size(), wav_file_handle);
    fclose(wav_file_handle);
    return written_bytes == expected_bytes;
  }

  bool save_as_32_bits(const std::string &file_path) {
    std::vector<int32_t> samples;
    samples.reserve(m_samples.size());
    for (auto &sample : m_samples) {
      samples.push_back(static_cast<int32_t>(sample));
    }
    auto wav_file_handle = fopen(file_path.c_str(), "wb");
    if (nullptr == wav_file_handle) {
      return false;
    }
    const size_t expected_bytes =
        sizeof(m_header) + (sizeof(int32_t) * m_samples.size());
    size_t written_bytes =
        fwrite(reinterpret_cast<int8_t *>(&m_header), sizeof(int8_t),
               sizeof(m_header), wav_file_handle);
    written_bytes += (fwrite(samples.data(), sizeof(int32_t), samples.size(),
                             wav_file_handle) *
                      sizeof(int32_t));
    fclose(wav_file_handle);
    return written_bytes == expected_bytes;
  }

  constexpr size_t get_maximum_sample_value() const {
    size_t max_sample_value = static_cast<size_t>(CHAR_MAX);
    switch (m_header.bits_per_sample) {
    case _8_BITS_PER_SAMPLE:
      max_sample_value = static_cast<size_t>(CHAR_MAX);
      break;
    case _16_BITS_PER_SAMPLE:
      max_sample_value = static_cast<size_t>(INT16_MAX);
      break;
    case _24_BITS_PER_SAMPLE:
      max_sample_value = static_cast<size_t>(INT24_MAX);
      break;
    case _32_BITS_PER_SAMPLE:
      max_sample_value = static_cast<size_t>(INT32_MAX);
      break;
    default:
      max_sample_value = static_cast<size_t>(CHAR_MAX);
      break;
    }
    return max_sample_value;
  }

  // Applies a frequency (could be lfo or not) oscillator to the bitcrusher wet
  // precentage value the oscillator can be of the various waves defined in
  // wave_type_t
  void apply_osc_to_bitcrusher_wet_value(double frequency,
                                         const wave_type_t wave_type,
                                         bool is_lfo = false) {

    if (!m_apply_bitcrusher_effect) {
      return;
    }

    if (is_lfo && frequency > 20) {
      frequency = 1.0;
    }

    if (frequency < 0.0) {
      frequency = 1.0;
    }

    m_lfo_terminate.store(false);
    m_lfo_future = std::async(std::launch::async, [&]() {
      m_start_lfo_latch.wait();
      double time = 0.0;
      constexpr const double volume{1.0};
      constexpr const double phase{0.0};
      while (!m_lfo_terminate.load()) {

        if (wave_type == wave_type_t::linear) {
          if (m_bitcrusher_wet_percent > 1.0) {
            m_bitcrusher_wet_percent = 0.01;
          }
          if (m_bitcrusher_wet_percent < 0.02) {
            m_bitcrusher_wet_percent = 1.0;
          }
          m_bitcrusher_wet_percent -= 0.01;
          // Convert frequency to period and use that as thread sleep value
          std::this_thread::sleep_for(std::chrono::milliseconds(
              static_cast<size_t>(pow(frequency, -1.0) * 1000.0)));
          continue;
        } else if (wave_type == wave_type_t::sawtooth) {
          m_bitcrusher_wet_percent =
              std::abs(static_cast<double>(
                  helper::pcm_saw_tooth(time, volume, frequency))) /
              static_cast<double>(INT32_MAX);
        } else if (wave_type == wave_type_t::square) {
          m_bitcrusher_wet_percent =
              std::abs(static_cast<double>(
                  helper::pcm_square(time, volume, frequency))) /
              static_cast<double>(INT32_MAX);
        } else if (wave_type == wave_type_t::sine) {
          m_bitcrusher_wet_percent =
              std::abs(static_cast<double>(
                  helper::pcm_sine(frequency, time, volume, phase))) /
              static_cast<double>(INT32_MAX);
        }

        time += std::numeric_limits<double>::lowest();
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
    });
  }

  // Applies a frequency (could be lfo or not) oscillator to the bitcrusher wet
  // precentage value the oscillator can be of the various waves defined in
  // wave_type_t
  void apply_osc_to_bitcrusher_gain_value(double frequency,
                                          const wave_type_t wave_type,
                                          bool is_lfo = false) {

    if (!m_apply_bitcrusher_effect) {
      return;
    }

    if (is_lfo && frequency > 20) {
      frequency = 1.0;
    }

    if (frequency < 0.0) {
      frequency = 1.0;
    }

    m_lfo_terminate.store(false);
    m_lfo_future = std::async(std::launch::async, [&]() {
      m_start_lfo_latch.wait();
      double time = 0.0;
      constexpr const double volume{1.0};
      constexpr const double phase{0.0};
      while (!m_lfo_terminate.load()) {

        if (wave_type == wave_type_t::linear) {
          if (m_bitcrusher_gain_value > 1.0) {
            m_bitcrusher_gain_value = 0.01;
          }
          if (m_bitcrusher_gain_value < 0.02) {
            m_bitcrusher_gain_value = 1.0;
          }
          m_bitcrusher_gain_value -= 0.01;
          // Convert frequency to period and use that as thread sleep value
          std::this_thread::sleep_for(std::chrono::milliseconds(
              static_cast<size_t>(pow(frequency, -1.0) * 1000.0)));
          continue;
        } else if (wave_type == wave_type_t::sawtooth) {
          m_bitcrusher_gain_value =
              std::abs(static_cast<double>(
                  helper::pcm_saw_tooth(time, volume, frequency))) /
              static_cast<double>(INT32_MAX);
        } else if (wave_type == wave_type_t::square) {
          m_bitcrusher_gain_value =
              std::abs(static_cast<double>(
                  helper::pcm_square(time, volume, frequency))) /
              static_cast<double>(INT32_MAX);
        } else if (wave_type == wave_type_t::sine) {
          m_bitcrusher_gain_value =
              std::abs(static_cast<double>(
                  helper::pcm_sine(frequency, time, volume, phase))) /
              static_cast<double>(INT32_MAX);
        }

        time += std::numeric_limits<double>::lowest();
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
    });
  }

  wave_header_t m_header;
  std::vector<int64_t> m_samples;
  bool m_apply_bitcrusher_effect{false};
  double m_bitcrusher_wet_percent{1.0};
  double m_bitcrusher_gain_value{1.0};
  std::future<void> m_lfo_future;
  std::atomic<bool> m_lfo_terminate;
  std::latch m_start_lfo_latch{1};
  size_t m_sink_index{0ul};
};

// Not to be directly used but more for the wave_file_t to help generate nice
// synth sounds!
namespace processing_functions {

std::vector<int32_t> oscillator_processing_callback(
    const oscillator_selection_t &osc_to_process, const size_t &sample_size,
    const double &volume, const bool &is_stereo, const uint32_t &sample_rate,
    synth_config_t &configuration,
    std::initializer_list<oscillator_config_t *> selected_oscs) {

  std::vector<int32_t> samples;
  oscillator_config_t* primary_osc{nullptr};

  switch(osc_to_process) {
    case oscillator_selection_t::oscillator_a: {
      primary_osc = &configuration.oscillator_a;
      break;
    }
    case oscillator_selection_t::oscillator_b: {
      primary_osc = &configuration.oscillator_b;
      break;
    }
    case oscillator_selection_t::oscillator_c: {
      primary_osc = &configuration.oscillator_c;
      break;
    }
    case oscillator_selection_t::oscillator_d: {
      primary_osc = &configuration.oscillator_d;
      break;
    }
    case oscillator_selection_t::oscillator_e: {
      primary_osc = &configuration.oscillator_e;
      break;
    }
    case oscillator_selection_t::oscillator_f: {
      primary_osc = &configuration.oscillator_f;
      break;
    }
    case oscillator_selection_t::oscillator_g: {
      primary_osc = &configuration.oscillator_g;
      break;
    }
    default: {
      return samples;
    }
  }

  if (primary_osc->operator_type != oscillator_type_t::carrier) {
     return samples;
  }

  samples.reserve(sample_size);
  double phase{};
  double time{};
  for (size_t _{}; _ < sample_size; _++) {
    int64_t sample{};
    double offset{};
    double frequency_offset{};
    double phase_offset = primary_osc->initial_phase_offset;
    double amplitude_offset{};
    double modulation_amplitude{};
    bool ring_modulation{false};
    for (auto &selected_osc : selected_oscs) {

      if (nullptr == selected_osc) {
        continue;
      }

      if (selected_osc->osc_to_modulate = osc_to_process) {
        continue;
      }

      modulation_amplitude += selected_osc->modulation_amplitude;

      const double modulating_frequency = selected_osc->frequency;

      if ((selected_osc->wave_type & wave_type_t::sine)) {
        offset += helper::pcm_sine(modulating_frequency, time, volume, phase);
      }
      if ((selected_osc->wave_type & wave_type_t::triangle)) {
        offset += helper::pcm_triangle(time, volume, modulating_frequency);
      }
      if ((selected_osc->wave_type & wave_type_t::square)) {
        offset += helper::pcm_square(time, volume, modulating_frequency);
      }
      if ((selected_osc->wave_type & wave_type_t::sawtooth)) {
        offset += helper::pcm_saw_tooth(time, volume, modulating_frequency);
      }

      if (modulation_amplitude <= 0.0) {
        modulation_amplitude = 1.0;
      }

      offset /= volume;
      offset *= modulation_amplitude;

      switch (selected_osc->operator_type) {
      case oscillator_type_t::frequency_modulation: {
        frequency_offset = offset;
        break;
      }
      case oscillator_type_t::phase_modulation: {
        phase_offset = offset;
        break;
      }
      case oscillator_type_t::amplitude_modulation: {
        amplitude_offset = offset;
        break;
      }
      case oscillator_type_t::ring_modulation: {
        ring_modulation = true;
        break;
      }
      default: {
        throw std::invalid_argument("How is this possible?");
      }
      }
    }
    
    const auto wave_type =
        static_cast<uint8_t>(primary_osc->wave_type);

    if ((wave_type & wave_type_t::sine)) {
      sample += helper::pcm_sine(
          primary_osc->frequency + frequency_offset, time,
          volume + amplitude_offset, phase + phase_offset);
    }
    if ((wave_type & wave_type_t::triangle)) {
      sample += helper::pcm_triangle(time, volume + amplitude_offset,
                                     primary_osc->frequency +
                                         frequency_offset);
    }
    if ((wave_type & wave_type_t::square)) {
      sample += helper::pcm_square(time, volume + amplitude_offset,
                                   primary_osc->frequency +
                                       frequency_offset);
    }
    if ((wave_type & wave_type_t::sawtooth)) {
      sample += helper::pcm_saw_tooth(time, volume + amplitude_offset,
                                      primary_osc->frequency +
                                          frequency_offset);
    }

    // Ring modulation we will multiply the carrier signal with the modulating
    // signal (sine, triangle, etc.)
    if (ring_modulation) {
      sample *= offset;
    }

    time += (1.0 / static_cast<double>(sample_rate));
    samples.push_back(sample);
  }

  return samples;
}

} // namespace processing_functions

#endif

#endif
