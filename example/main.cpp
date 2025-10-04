#include <cstdint>
#include <iostream>
#include <limits>
#include <wave_t.hpp>

constexpr double C4_FREQUENCY = 261.626;

double detune(double x) {

  if (x < 0.0) {
    x = 0.0;
  }
  if (x > 1.0) {
    x = 1.0;
  }

  // https://www.adamszabo.com/internet/adam_szabo_how_to_emulate_the_super_saw.pdf

  return (10028.7312891634 * std::pow(x, 11.0)) -
         (50818.8652045924 * std::pow(x, 10.0)) +
         (111363.4808729368 * std::pow(x, 9.0)) -
         (138150.6761080548 * std::pow(x, 8.0)) +
         (106649.6679158292 * std::pow(x, 7.0)) -
         (53046.9642751875 * pow(x, 6)) +
         (17019.9518580080 * std::pow(x, 5.0)) -
         (3425.0836591318 * std::pow(x, 4.0)) +
         (404.2703938388 * std::pow(x, 3.0)) -
         (24.1878824391 * std::pow(x, 2.0)) +
         (0.6717417634 * std::pow(x, 2.0)) + 0.0030115596;
}

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
  output.generate_wave(wave_type_t::sine, sample_size, C4_FREQUENCY, 0.6);
  output.save("output.wav");

  //Try out 24-bit audio!
  /*wave_file_t output24;
  output24.set_sample_rate(sample_rate);
  output24.set_number_of_channels(1);
  output24.set_bits_per_sample(24);
  // This helper member function can generate one or a combination of waves only
  // supports mono or stereo for now
  output24.generate_wave(wave_type_t::sine, sample_size, C4_FREQUENCY, 0.6);
  output24.save("output.wav");*/

  //Read output.wav from our earlier example
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
    float magnitude = std::norm(frequency_domain[frequency]);
    if (magnitude >= max) {
      detected_frequency_index = frequency;
      max = magnitude;
    }
  }

  std::cout << "Index: " << detected_frequency_index << std::endl;
  std::cout << "Fundamental frequency: "
            << (static_cast<float>(detected_frequency_index) *
                (static_cast<float>(sample_rate) /
                 (static_cast<float>(dft_sample_size))))
            << " Amplitiude (RMS): " << max << std::endl;

  std::cout << "Testing IDFT by writing wav file from frequency domain..."
            << std::endl;

  wave_file_t idft_wav_file(dft_sample_size, frequency_domain);
  output.set_sample_rate(sample_rate);
  output.set_number_of_channels(1);
  output.set_bits_per_sample(16);
  output.save("idft_output.wav");

  std::cout << "Generating a super saw (" << MAX_OSC_SUPPORT << "osc ) at C4 (261.626 HZ)"
            << std::endl;

  wave_file_t synth_output;
  synth_output.set_sample_rate(sample_rate);
  synth_output.set_number_of_channels(1);
  synth_output.set_bits_per_sample(16);

  wave_file_t::synth_config_t configuration;

  // Supersaw example -- hopefully :P
  
  constexpr double detune_amount = 0.00;

  configuration.oscillator_a.operator_type = wave_file_t::carrier;
  configuration.oscillator_a.wave_type = wave_type_t::sawtooth;
  configuration.oscillator_a.frequency = C4_FREQUENCY * (1.0 - (0.88997686 * detune(detune_amount)));
  configuration.oscillator_a.osc_to_modulate =
      wave_file_t::oscillator_selection_t::none_selected;

  configuration.oscillator_b.operator_type = wave_file_t::carrier;
  configuration.oscillator_b.wave_type = wave_type_t::sawtooth;
  configuration.oscillator_b.frequency = C4_FREQUENCY * (1.0 - (0.93711560 * detune(detune_amount)));
  configuration.oscillator_b.osc_to_modulate = wave_file_t::oscillator_selection_t::none_selected;

  configuration.oscillator_c.operator_type = wave_file_t::carrier;
  configuration.oscillator_c.wave_type = wave_type_t::sawtooth;
  configuration.oscillator_c.frequency = C4_FREQUENCY * (1.0 - (0.98047643 * detune(detune_amount)));
  configuration.oscillator_c.osc_to_modulate =
      wave_file_t::oscillator_selection_t::none_selected;


  configuration.oscillator_d.operator_type = wave_file_t::carrier;
  configuration.oscillator_d.wave_type = wave_type_t::sawtooth;
  configuration.oscillator_d.frequency = C4_FREQUENCY;
  configuration.oscillator_d.osc_to_modulate = wave_file_t::oscillator_selection_t::none_selected;

   configuration.oscillator_e.operator_type = wave_file_t::carrier;
  configuration.oscillator_e.wave_type = wave_type_t::sawtooth;
  configuration.oscillator_e.frequency = C4_FREQUENCY * (1.0 - (1.01991221 * detune(detune_amount)));
  configuration.oscillator_e.osc_to_modulate =
      wave_file_t::oscillator_selection_t::none_selected;
 
  configuration.oscillator_f.operator_type = wave_file_t::carrier;
  configuration.oscillator_f.wave_type = wave_type_t::sawtooth;
  configuration.oscillator_f.frequency = C4_FREQUENCY * (1.0 - (1.06216538 * detune(detune_amount)));
  configuration.oscillator_f.osc_to_modulate =
      wave_file_t::oscillator_selection_t::none_selected;

  configuration.oscillator_g.operator_type = wave_file_t::carrier;
  configuration.oscillator_g.wave_type = wave_type_t::sawtooth;
  configuration.oscillator_g.frequency = C4_FREQUENCY * (1.0 - (1.10745242 * detune(detune_amount)));
  configuration.oscillator_g.osc_to_modulate =
      wave_file_t::oscillator_selection_t::none_selected;


  constexpr size_t seconds = 3ul;

  const size_t synth_sample_size = sample_rate * seconds;

  if (synth_output.generate_synth(synth_sample_size, 0.12, configuration)) {
    synth_output.save("synth_output.wav");
  } else {
    std::cout
        << "Invalid synth configuration or failed to generate synth -- sorry."
        << std::endl;
  }

  std::cout
      << "Demo finished!! If you don't see this message assume process crashed!"
      << std::endl;

  return 0;
}
