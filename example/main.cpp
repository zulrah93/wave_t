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

#include <cstdint>
#include <iostream>
#include <limits>
#include <wave_t.hpp>

constexpr double C4_FREQUENCY{261.626};
constexpr double A4_FREQUENCY{440.000};
constexpr double D5_FREQUENCY{622.25};

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
  const size_t sample_size = sample_rate * 60 * 5; // 5 minutes of 16-bit PCM sample
  // This helper member function can generate one or a combination of waves only
  // supports mono or stereo for now
  output.generate_wave(wave_type_t::sine, sample_size, A4_FREQUENCY, 0.6);
  if (!output.save("output.wav")) {
    std::cout << "Failed to save 16-bit generated wav file" << std::endl;
  }
  if (!output.save_waveform_as_grayscale_pbm("sine.pbm")) {
        std::cout << "Failed to save generated grayscale portable bitmap of wav file!" << std::endl;
  }

  // Try out 24-bit audio!
  wave_file_t output24;
  output24.set_sample_rate(sample_rate);
  output24.set_number_of_channels(1);
  output24.set_bits_per_sample(24);
  // This helper member function can generate one or a combination of waves only
  // supports mono or stereo for now
  output24.generate_wave(wave_type_t::sine, sample_size, C4_FREQUENCY, 0.6);
  if (!output24.save("output_24.wav")) {
    std::cout << "Failed to save 24-bit generated wav file" << std::endl;
  }

  std::cout << "Finding fundamental frequency of D#5 Trumpet sample reference "
               "frequency is "
            << D5_FREQUENCY << std::endl;

  const char *input_path = "../samples/D#5_Trumpet.wav";

  // Read output.wav from our earlier example
  wave_file_t input(input_path);

  if (!input) {
    std::cout << "Failed to load D#5_Trumpet.wav it has invalid wav header!!"
              << std::endl;
    std::cout << "DEBUG: " << std::endl << input.get_readable_wave_header();
    return 1;
  }

  std::cout << "\033[01;33m" << input.get_readable_wave_header() << "\033[0m"
            << std::endl
            << "number_of_samples=" << input.sample_size() << std::endl;

  const size_t dft_sample_size =
      input.sample_size(); // NOTE: Larger values means slower time since this
                           // is a slow dft implementation
  const bool async =
      true; // DFT can be calculated asynchronously may help performance
  auto frequency_domain = input.get_frequency_domain(dft_sample_size, async);

  size_t detected_frequency_index = 0;
  double max = std::numeric_limits<float>::min();

  const size_t max_frequency_index =
      input.get_nyquist_frequency(); // Nyquist limit means we can't go beyond
                                     // the half the sample size or so I think
  std::cout << "The nyquist frequency is " << max_frequency_index << std::endl;
  for (size_t frequency = 0; frequency < max_frequency_index; frequency++) {
    float magnitude = std::norm(frequency_domain[frequency]);
    if (magnitude >= max) {
      detected_frequency_index = frequency;
      max = magnitude;
    }
  }

  double fundamental_frequency = (static_cast<float>(detected_frequency_index) *
                                  (static_cast<float>(sample_rate) /
                                   (static_cast<float>(dft_sample_size))));

  double percent_error =
      (100.0 * std::abs((fundamental_frequency - D5_FREQUENCY) / D5_FREQUENCY));

  std::cout << "Index: " << detected_frequency_index << std::endl;
  std::cout << "\033[01;32mFundamental frequency: " << fundamental_frequency
            << "\033[0m\033[01;31m "
            << "(percent error is " << percent_error << " %) \033[0m"
            << std::endl;

  if (input.save_waveform_as_grayscale_pbm("d#5_trumpet.pbm")) {
        std::cout << "Failed to save generated grayscale portable bitmap of wav file!" << std::endl;
  }

  std::cout << "Testing IDFT by writing wav file from frequency domain..."
            << std::endl;

  wave_file_t idft_wav_file(dft_sample_size, frequency_domain);
  output.set_sample_rate(sample_rate);
  output.set_number_of_channels(1);
  output.set_bits_per_sample(16);
  output.save("idft_output.wav");

  std::cout << "Generating a super saw (" << MAX_OSC_SUPPORT
            << "osc ) at C4 (261.626 HZ)" << std::endl;

  wave_file_t synth_output;
  synth_output.set_sample_rate(sample_rate);
  synth_output.set_number_of_channels(1);
  synth_output.set_bits_per_sample(
      32); // Set it to 8-bit if you want that retro feel :p

  synth_config_t configuration{};

#pragma message("Uncomment line below to get alternative synth demo")
#define AM_PM_MODULATION_DEMO

#ifndef AM_PM_MODULATION_DEMO
  // Supersaw example -- hopefully :P
  constexpr double detune_amount = 0.0;

  configuration.oscillator_a.operator_type = carrier;
  configuration.oscillator_a.wave_type = wave_type_t::sawtooth;
  configuration.oscillator_a.frequency =
      C4_FREQUENCY * (1.0 - (0.88997686 * detune(detune_amount)));
  configuration.oscillator_a.osc_to_modulate =
      oscillator_selection_t::none_selected;

  configuration.oscillator_b.operator_type = carrier;
  configuration.oscillator_b.wave_type = wave_type_t::sawtooth;
  configuration.oscillator_b.frequency =
      C4_FREQUENCY * (1.0 - (0.93711560 * detune(detune_amount)));
  configuration.oscillator_b.osc_to_modulate =
      oscillator_selection_t::none_selected;

  configuration.oscillator_c.operator_type = carrier;
  configuration.oscillator_c.wave_type = wave_type_t::sawtooth;
  configuration.oscillator_c.frequency =
      C4_FREQUENCY * (1.0 - (0.98047643 * detune(detune_amount)));
  configuration.oscillator_c.osc_to_modulate =
      oscillator_selection_t::none_selected;

  configuration.oscillator_d.operator_type = carrier;
  configuration.oscillator_d.wave_type = wave_type_t::sawtooth;
  configuration.oscillator_d.frequency = C4_FREQUENCY;
  configuration.oscillator_d.osc_to_modulate =
      oscillator_selection_t::none_selected;

  configuration.oscillator_e.operator_type = carrier;
  configuration.oscillator_e.wave_type = wave_type_t::sawtooth;
  configuration.oscillator_e.frequency =
      C4_FREQUENCY * (1.0 - (1.01991221 * detune(detune_amount)));
  configuration.oscillator_e.osc_to_modulate =
      oscillator_selection_t::none_selected;

  configuration.oscillator_f.operator_type = carrier;
  configuration.oscillator_f.wave_type = wave_type_t::sawtooth;
  configuration.oscillator_f.frequency =
      C4_FREQUENCY * (1.0 - (1.06216538 * detune(detune_amount)));
  configuration.oscillator_f.osc_to_modulate =
      oscillator_selection_t::none_selected;

  configuration.oscillator_g.operator_type = carrier;
  configuration.oscillator_g.wave_type = wave_type_t::sawtooth;
  configuration.oscillator_g.frequency =
      C4_FREQUENCY * (1.0 - (1.10745242 * detune(detune_amount)));
  configuration.oscillator_g.osc_to_modulate =
      oscillator_selection_t::none_selected;

  constexpr const double seconds{1.25};

  const size_t synth_sample_size =
      static_cast<size_t>(ceil(static_cast<double>(sample_rate) * seconds));

  // Uncomment if you want to apply bitcrusher :)
  // synth_output.apply_bitcrusher_effect();
  constexpr const double volume{0.13};
  if (synth_output.generate_synth(synth_sample_size, volume, configuration)) {
    std::cout << synth_output.get_peak_decibel_fullscale_of_signal()
              << " dBFS is the peak of this generated super saw!!" << std::endl;
    /*for(size_t index = 0; index < (synth_sample_size / 16); index++) {
        std::cout << synth_output.index_as_dBFS(index).value_or(-144.0) << "
    dBFS" << std::endl;
    }*/
    synth_output.save("synth_output.wav");
  } else {
    std::cout
        << "Invalid synth configuration or failed to generate synth -- sorry."
        << std::endl;
  }

  std::cout
      << "Demo finished!! If you don't see this message assume process crashed!"
      << std::endl;
#else

  constexpr const double E4_FREQUENCY{329.6276};

  configuration.oscillator_a.operator_type = carrier;
  configuration.oscillator_a.wave_type = wave_type_t::sine;
  configuration.oscillator_a.frequency = C4_FREQUENCY;
  configuration.oscillator_a.osc_to_modulate =
      oscillator_selection_t::none_selected;

  configuration.oscillator_b.operator_type = oscillator_type_t::ring_modulation;
  configuration.oscillator_b.wave_type = wave_type_t::sine;
  configuration.oscillator_b.frequency = C4_FREQUENCY / 2;
  configuration.oscillator_b.osc_to_modulate =
      oscillator_selection_t::oscillator_a;
  configuration.oscillator_b.modulation_amplitude = 0.75;

  configuration.oscillator_c.operator_type = oscillator_type_t::ring_modulation;
  configuration.oscillator_c.wave_type = wave_type_t::sine;
  configuration.oscillator_c.frequency = C4_FREQUENCY * 1.5;
  configuration.oscillator_c.osc_to_modulate =
      oscillator_selection_t::oscillator_b;
  configuration.oscillator_c.modulation_amplitude = 0.5;

  configuration.oscillator_d.operator_type = oscillator_type_t::ring_modulation;
  configuration.oscillator_d.wave_type = wave_type_t::sine;
  configuration.oscillator_d.frequency = C4_FREQUENCY * 1.25;
  configuration.oscillator_d.osc_to_modulate =
      oscillator_selection_t::oscillator_c;
  configuration.oscillator_d.modulation_amplitude = 0.5;

  configuration.oscillator_e.operator_type = oscillator_type_t::carrier;
  configuration.oscillator_e.wave_type = wave_type_t::sawtooth;
  configuration.oscillator_e.frequency = E4_FREQUENCY;
  configuration.oscillator_e.osc_to_modulate =
      oscillator_selection_t::none_selected;

  configuration.oscillator_f.operator_type = oscillator_type_t::ring_modulation;
  configuration.oscillator_f.wave_type = wave_type_t::sawtooth;
  configuration.oscillator_f.frequency = E4_FREQUENCY * 8;
  configuration.oscillator_f.osc_to_modulate =
      oscillator_selection_t::oscillator_d;
  configuration.oscillator_f.modulation_amplitude = 2.5;

  configuration.oscillator_g.operator_type = oscillator_type_t::ring_modulation;
  configuration.oscillator_g.wave_type = wave_type_t::sawtooth;
  configuration.oscillator_g.frequency = E4_FREQUENCY * 8;
  configuration.oscillator_g.osc_to_modulate =
      oscillator_selection_t::oscillator_d;
  configuration.oscillator_g.modulation_amplitude = 2.5;

  constexpr const double seconds{1.25};

  const size_t synth_sample_size =
      static_cast<size_t>(ceil(static_cast<double>(sample_rate) * seconds));

  // Uncomment if you want to apply bitcrusher :)
  //synth_output.apply_bitcrusher_effect();
  constexpr const double volume{0.60};
  if (synth_output.generate_synth(synth_sample_size, volume, configuration)) {
    std::cout << synth_output.get_peak_decibel_fullscale_of_signal()
              << " dBFS is the peak of this generated super saw!!" << std::endl;
    if (!synth_output.save_waveform_as_grayscale_pbm("synth.pbm")) {
            std::cout << "Failed to save generated grayscale portable bitmap of wav file!" << std::endl;
    }
    else {
        std::cout << "Checkout the groovy waveform visually!" << std::endl;
    }
    synth_output.save("synth_output.wav");
  } else {
    std::cout
        << "Invalid synth configuration or failed to generate synth -- sorry."
        << std::endl;
  }

  std::cout
      << "Demo finished!! If you don't see this message assume process crashed!"
      << std::endl;
#endif

  return 0;
}
