# C++ .WAV manipulation header only library

To use this header you can just wget like below or clone repo but you only need to depend on header.

```
    wget https://raw.githubusercontent.com/zulrah93/wave_t/refs/heads/master/include/wave_t.hpp
```
# Example usage of header-only library

```
#include <cstdint>
#include <wave_t.hpp>
#include <iostream>

/* Example using the header only wave file reader and writer class in wave_t.hpp */

int main(int arguments_size, char** arguments) {
  std::cout << "wave_t.hpp usage example!" << std::endl;
  wave_file_t output;
  output.set_sample_rate(44100);
  output.set_number_of_channels(1);
  output.set_bits_per_sample(16);
  const size_t sample_size = 44100 * 60 * 5; // 5 minutes of 16-bit PCM sample
  //This helper member function can generate one or a combination of waves only supports mono or stereo for now 
  output.generate_wave(wave_type_t::sine, sample_size, 440.0, 0.8);
  output.save("output.wav");
  return 0;
}
```

<img width="2268" height="190" alt="image" src="https://github.com/user-attachments/assets/293356ba-3ebd-47be-b062-50411e6b18f8" />

```
#include <cstdint>
#include <wave_t.hpp>
#include <iostream>

constexpr double C4_FREQUENCY = 261.626;

/* Example generating a super saw with no detune feel free to modify to see how timbre changes. Source of the numbers used are in this paper here https://www.adamszabo.com/internet/adam_szabo_how_to_emulate_the_super_saw.pdf */

int main(int arguments_size, char** arguments) {

std::cout << "Generating a super saw (" << MAX_OSC_SUPPORT << "osc ) at C4 (261.626 HZ)"
            << std::endl;

  wave_file_t synth_output;
  synth_output.set_sample_rate(sample_rate);
  synth_output.set_number_of_channels(1);
  synth_output.set_bits_per_sample(24);

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
}

```

<img width="2462" height="201" alt="image" src="https://github.com/user-attachments/assets/42e5af5b-9645-4d87-aaa2-4ce2b154f0c6" />


# Supports

DFT and IDFT for pitch detection or FM synthesis, it also supports calculating the DFT/IDFT async. This could be more optimized.

Also supports adding samples by converting a frequency domain (a vector of complex numbers) to PCM samples (essentially the time domain of the signal).

Has support for FM based synthesis using 7 oscillators of the basic wave shapes/types. Still needs more improvent but working for now.




