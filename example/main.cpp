#include <cstdint>
#include <iostream>
#include <limits>
#include <wave_t.hpp>

/* Example using the header only wave file reader and writer class in wave_t.hpp
 */

int main(int arguments_size, char **arguments) {
  std::cout << "wave_t.hpp usage example!" << std::endl;
  wave_file_t output;
  const size_t sample_rate = 44100;
  output.set_sample_rate(sample_rate);
  output.set_number_of_channels(1);
  output.set_bits_per_sample(16);
  const size_t sample_size = 44100 * 60 * 5; // 5 minutes of 16-bit PCM sample
  // This helper member function can generate one or a combination of waves only
  // supports mono or stereo for now
  output.generate_wave(wave_type_t::sine, sample_size, 440.0, 0.6);
  output.save("output.wav");

  wave_file_t input("output.wav");

  if (!input) {
    std::cout << "Failed to load output.wav it has invalid wav header!!"
              << std::endl;
    return 1;
  }

  std::cout << input.get_readable_wave_header() << std::endl
            << "number_of_samples=" << input.sample_size() << std::endl;

  const size_t dft_sample_size = sample_rate / 2;
  const bool async =
      true; // DFT can be calculated asynchronously may help performance
  auto frequency_domain = input.get_frequency_domain(dft_sample_size, async);

  size_t detected_frequency_index = 0;
  double max = std::numeric_limits<float>::min();

  for (size_t frequency = 0; frequency < frequency_domain.size(); frequency++) {
    float magnitude =  std::norm(frequency_domain[frequency]);
    if (magnitude >= max) {
      detected_frequency_index = frequency;
      max = magnitude;
    }
  }

  std::cout << "Index: " << detected_frequency_index << std::endl;
  std::cout << "Fundamental frequency: "
            << (static_cast<float>(detected_frequency_index) *
                (static_cast<float>(sample_rate) /
                 (static_cast<float>(dft_sample_size)))) << " Amplitiude (RMS): " << max << std::endl;

  std::cout << "Testing IDFT by writing wav file from frequency domain..."
            << std::endl;

  wave_file_t idft_wav_file(dft_sample_size, frequency_domain);
  output.set_sample_rate(sample_rate);
  output.set_number_of_channels(1);
  output.set_bits_per_sample(16);
  output.save("idft_output.wav");

  std::cout
      << "Demo finished!! If you don't see this message assume process crashed!"
      << std::endl;

  return 0;
}
