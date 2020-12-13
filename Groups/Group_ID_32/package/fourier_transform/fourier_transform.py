import numpy as np

'''This function takes an imae as argument and calculates discrete 
fourier transform'''
def dft(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))                    
    M = np.exp(-2j * np.pi * k * n / N)      #formula for fourier transform
    return np.dot(M, x)

'''Since image is a 2d object we need to use dft twice first column
wise then row wise'''
def fourier_transform(img):
  fx = dft(img.T)
  fy = dft(fx.T)
  '''Shifting the low frequency values to the center and 
  higher at the corners'''
  fshift = np.fft.fftshift(fy)
  '''the magnitude spectrum is large hence we calculate 
  the log of it'''
  magnitude_spectrum = 20*np.log(np.abs(fshift))
  return magnitude_spectrum
