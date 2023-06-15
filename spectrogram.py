#!/usr/bin/python
import numpy as np
from PIL import Image
import wave
from array import array

length = 7  # seconds
samplerate = 96000

def make_wav(image_filename):
    """ Make a WAV file having a spectrogram resembling an image """
    # Load image
    image = Image.open(image_filename)
    w_img, h_img = image.width, image.height
    image = image.resize((int(length * samplerate / (h_img * 2)), h_img))
    image = np.sum(image, axis = 2).T[:, ::-1]
    w, h = image.shape

    max32 = 2**31-1
    data = image
    # phase randomization
    rng = np.random.default_rng()
    noise = rng.standard_normal(size=data.shape) * 2*np.pi
    data = (np.real(data) * np.sin(noise)) + (np.imag(data) * np.cos(noise))
    # Fourier transform
    data = np.fft.irfft(data, h*2, axis=1)
    # normalization and DC offset fix
    data = np.real(data) + np.imag(data)
    data -= np.average(data)
    data *= (np.float64(max32))/np.amax(data)
    # value cast
    data = np.int32(data)

    # Write to disk
    output_file = wave.open(image_filename+".wav", "wb")
    output_file.setparams((1, 4, samplerate, 0, "NONE", "not compressed"))
    output_file.writeframes(data)
    output_file.close()
    print("Wrote %s.wav" % image_filename)


if __name__ == "__main__":
    import sys
    make_wav(sys.argv[1])

