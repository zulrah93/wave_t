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

# Supports

DFT and IDFT for pitch detection or FM synthesis, it also supports calculating the DFT/IDFT async. This could be more optimized.

Also supports adding samples by converting a frequency domain (a vector of complex numbers) to PCM samples (essentially the time domain of the signal).


