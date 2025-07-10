#include <cstdint>
#include <wave_t.hpp>
#include <iostream>
#include <numbers>
#include <cmath>
#include <limits>

/* Example using the header only wave file reader and writer class in wave_t.hpp */

// Source: https://en.wikipedia.org/wiki/Sine_wave
inline int16_t pcm_sine(double frequency, double time, double amplititude) {
    return static_cast<int16_t>(amplititude * sin(2 * std::numbers::pi * frequency * time));
}

// Source: https://en.wikipedia.org/wiki/Triangle_wave
inline int16_t pcm_triangle_wave(double time, double amplititude, double frequency) {
  const double period = (1.0 / frequency);
  return static_cast<int16_t>(fabs(((2.0 * amplititude) / std::numbers::pi) * asin(sin(2 * time * (std::numbers::pi / period )))));
}

inline double set_volume(double percent) {
   if (percent > 1.0) {
      percent = 1.0;
   }
   if (percent < 0.0) {
      percent = 0.0;
   }
   return static_cast<double>(INT16_MAX - 1024) * percent; 
}

int main(int arguments_size, char** arguments) {
  std::cout << "wave_t.hpp usage example!" << std::endl;
  wave_file_t output;
  output.set_sample_rate(44100);
  output.set_number_of_channels(1);
  output.set_bits_per_sample(16);
  const size_t sample_size = 44100 * 60 * 5; // 5 minutes of 16-bit PCM samples
  double time = std::numbers::pi;
  const double frequency = 440.0;
  for(size_t _ = 0; _ < sample_size; _++) {
      int16_t sample = pcm_triangle_wave(time, set_volume(0.35), frequency);
      sample += pcm_sine(frequency, time, set_volume(0.35));
      output.add_16_bits_sample(sample);
      time += 0.00001;
  }
  output.save("output.wav");
  return 0;
}
