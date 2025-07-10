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
  output.generate_wave(wave_type_t::sine | wave_type_t::triangle | wave_type_t::square, sample_size, 440.0, 0.8);
  output.save("output.wav");
  return 0;
}
