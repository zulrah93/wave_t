#ifndef WAVE_T_HPP
#define WAVE_T_HPP

#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <stdarg.h>
#include <string>
#include <vector>
#include <numbers>
#include <cmath>
#include <limits>


constexpr auto RIFF_ASCII = std::byteswap(0x52494646);
constexpr auto WAVE_ASCII = std::byteswap(0x57415645);
constexpr auto FMT_ASCII = std::byteswap(0x666d7420);
constexpr auto DATA_ASCII = std::byteswap(0x64617461);
constexpr auto PCM = 1;
constexpr auto _8_BITS_PER_SAMPLE = 8;
constexpr auto _16_BITS_PER_SAMPLE = 16;
constexpr auto _24_BITS_PER_SAMPLE = 24;
constexpr auto DEFAULT_RESERVE_VALUE = 44100 * 60 * 5;

enum wave_type_t : uint8_t { sine = 1, triangle = 2, square = 4, sawtooth = 8 };

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
    m_header.sub_chunk_1_size = 16;
    m_samples.reserve(DEFAULT_RESERVE_VALUE);
  }
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

  bool add_24_bits_sample(const int32_t sample, ...) {
    va_list args;
    if (m_header.number_of_channels > 1) {
      va_start(args, sample);
    }
    m_samples.push_back(sample);
    for (size_t _ = 1;
         m_header.number_of_channels > 1 && _ < m_header.number_of_channels;
         _++) {
      m_samples.push_back(va_arg(args, const int32_t));
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

  
  bool generate_wave(uint8_t wave_type, size_t sample_size,
                     double frequency, double volume_percent) {

      auto set_volume = [](double percent) {
            if (percent > 1.0) {
              percent = 1.0;
            }
            if (percent < 0.0) {
              percent = 0.0;
            }
            return static_cast<double>(INT16_MAX - 1024) * percent;
      };

      //Source: https://en.wikipedia.org/wiki/Sign_function
      auto sign = [](double x) {
          if (x > 0) {
              return 1.0;
          }
          else if (x < 0) {
              return -1.0;
          }
          else {
              return 0.0;
          }
      };

      // Source: https://en.wikipedia.org/wiki/Sine_wave
      auto pcm_sine = [](double _frequency, double time, double amplititude) {
        return static_cast<int16_t>(
            amplititude * sin(2 * std::numbers::pi * _frequency * time));
      };

      // Source: https://en.wikipedia.org/wiki/Triangle_wave
      auto pcm_triangle = [](double time, double amplititude, double _frequency) {
            const double period = (1.0 / _frequency);
            return static_cast<int16_t>(fabs(((2.0 * amplititude) / std::numbers::pi) * asin(sin(2 * time * (std::numbers::pi / period )))));
      };

      // Source: https://en.wikipedia.org/wiki/Square_wave_(waveform)
      auto pcm_square = [sign](double time, double amplititude, double _frequency) {
            return static_cast<int16_t>(amplititude * sign(sin(2 * std::numbers::pi * _frequency * time)));
      };

      // Source: https://en.wikipedia.org/wiki/Sawtooth_wave
      auto pcm_saw_tooth = [](double time, double amplititude, double _frequency) {
          const double period = (1.0 / _frequency);
          return static_cast<int16_t>(2.0 * amplititude * ((time / period) - floor(0.5 + (time / period))));
      };

      bool is_stereo = m_header.number_of_channels == 2;

      double time = std::numbers::pi;
      for (size_t _ = 0; _ < sample_size; _++) {
        size_t wave_count = static_cast<size_t>((wave_type & wave_type_t::sine) == (wave_type_t::sine)) +
                                static_cast<size_t>((wave_type & wave_type_t::triangle) == (wave_type_t::triangle)) +
                                static_cast<size_t>((wave_type & wave_type_t::square) == (wave_type_t::square)) +
                                static_cast<size_t>((wave_type & wave_type_t::sawtooth) == (wave_type_t::sawtooth));
        const double volume = set_volume(volume_percent / static_cast<double>(wave_count));
        int16_t sample = 0;
        if ((wave_type & wave_type_t::sine)) {
          sample += pcm_sine(frequency, time, volume);
        }
        if ((wave_type & wave_type_t::triangle)) {
          sample += pcm_triangle(time, volume, frequency);
        }
        if ((wave_type & wave_type_t::square)) {
          sample += pcm_square(time, volume, frequency);
        }
        if ((wave_type & wave_type_t::sawtooth)) {
           sample += pcm_saw_tooth(time, volume, frequency);
        }
        switch (m_header.bits_per_sample) {
        case _8_BITS_PER_SAMPLE:
          add_8_bits_sample(sample);
          break;
        case _16_BITS_PER_SAMPLE:
          add_16_bits_sample(sample);
          break;
        case _24_BITS_PER_SAMPLE:
          add_24_bits_sample(sample);
          break;
        default:
          return false;
        }
        time += 0.00001;
      }

      return true;
  }

  bool save(const std::string &file_path) {
    m_header.sub_chunk_2_size =
        (m_samples.size() * m_header.number_of_channels *
         (m_header.bits_per_sample / 8));
    m_header.block_align =
        (m_header.number_of_channels * (m_header.bits_per_sample / 8));
    m_header.byte_rate = (m_header.sample_rate * m_header.number_of_channels *
                          (m_header.bits_per_sample / 8));
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
    default: {
      return false;
    }
    }
  }

private:
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
    written_bytes += fwrite(samples.data(), sizeof(int16_t), samples.size(),
                            wav_file_handle);
    fclose(wav_file_handle);
    return written_bytes == expected_bytes;
  }

  bool save_as_24_bits(const std::string &file_path) { return false; }
  wave_header_t m_header;
  std::vector<int32_t> m_samples;
};

#endif
