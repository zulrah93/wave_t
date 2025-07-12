#include <cstdint>
#include <wave_t.hpp>
#include <iostream>

/* Example using the header only wave file reader and writer class in wave_t.hpp */

int main(int arguments_size, char** arguments) {
  std::cout << "wave_t.hpp usage example!" << std::endl;
  //wave_file_t input("output.wav");
  
  //std::cout << input.get_readable_wave_header() << std::endl;

  // Source: https://en.wikipedia.org/wiki/Sine_wave
  auto pcm_sine = [](double _frequency, double time, double amplititude, double phase) {
       return static_cast<int16_t>(amplititude * sin((2 * std::numbers::pi * _frequency * time) + phase));
  };
  auto set_volume = [](double percent) {
      if (percent > 1.0) {
          percent = 1.0;
      }
      if (percent < 0.0) {
          percent = 0.0;
      }
      return static_cast<double>(INT16_MAX) * percent;
  };
  wave_file_t output;
  output.set_sample_rate(44100);
  output.set_number_of_channels(1);
  output.set_bits_per_sample(16);
  const size_t sample_size = 44100 * 60 * 5; // 5 minutes of 16-bit PCM sample
  //const double frequency = 440.0;
  //const double phase_1 = 0.0;
  //const double phase_2 = std::numbers::inv_pi;
  //const double percent = 0.4;
  //const double amplititude = set_volume(percent);
  //double time = 0.0;
  //for(size_t _ = 0; _ < sample_size; _++) {
  //    const size_t pcm_sample = pcm_sine(frequency, time, amplititude, phase_1)+  pcm_sine(frequency, time, amplititude, phase_2);
  //    output.add_16_bits_sample(pcm_sample);
  //    time += (1.0 / 44100.0);
  //}
  //This helper member function can generate one or a combination of waves only supports mono or stereo for now 
  output.generate_wave(wave_type_t::sawtooth, sample_size, 440.0, 0.6);
  output.save("output.wav");

  wave_file_t input("output.wav");
  
  std::cout << input.get_readable_wave_header() << std::endl;

  return 0;
}
