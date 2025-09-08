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

  std::cout << "Generating basic synth using sine 440 hz modulated by triangle and saw combined at  44hz" << std::endl;

  wave_file_t synth_output;
  synth_output.set_sample_rate(sample_rate);
  synth_output.set_number_of_channels(1);
  synth_output.set_bits_per_sample(16);

  wave_file_t::synth_config_t configuration;

  configuration.oscillator_a.operator_type = wave_file_t::carrier;
  configuration.oscillator_a.wave_type = wave_type_t::sine;
  configuration.oscillator_a.frequency = 261.626;
  configuration.oscillator_a.osc_to_modulate = wave_file_t::oscillator_selection_t::none_selected;

  configuration.oscillator_b.operator_type = wave_file_t::modulation;
  configuration.oscillator_b.wave_type = wave_type_t::sawtooth;
  configuration.oscillator_b.frequency = 26.1626 * 5.0;
  configuration.oscillator_b.osc_to_modulate = wave_file_t::oscillator_selection_t::none_selected;
  configuration.oscillator_b.modulation_amplitude = 1.07;

  memset(&configuration.oscillator_c, 0, sizeof(configuration.oscillator_c));
  memset(&configuration.oscillator_d, 0, sizeof(configuration.oscillator_d));
 
  const size_t synth_sample_size = sample_rate * 5ul; // 8 seconds

  if (synth_output.generate_synth(synth_sample_size, 0.6, configuration)) {
      synth_output.save("synth_output.wav");
  }
  else {
     std::cout << "Invalid synth configuration or failed to generate synth -- sorry." << std::endl;
  }


  std::cout
      << "Demo finished!! If you don't see this message assume process crashed!"
      << std::endl;

  return 0;
}
