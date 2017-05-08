# Italo Tobler Silva - nUSP 8551910

import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fftpack import fft, ifft

# Main function---------------------------------------------------------------------
# Obtains the desired filename
filename = input("What is the binary filename?")
# Obtains the functions parameters
s = float(input("What is the s parameter? "))
c = float(input("What is the c parameter? "))
show = int(input("Should the function plots be shown? (0 = no; 1 = yes)"))

# Checks if the passed function value is valid
if(show == 0 or show == 1):
    # Reads the binary file and stores it's contents in an array
    fileArray = np.fromfile(filename, np.int32, -1, "")
    # n stores the array size
    n = len(fileArray)
    # g stores the gaussian windowing of the signal
    g = np.multiply(fileArray, signal.gaussian(n, (n/s), True))
    # F stores the fft of the signal
    F = fft(fileArray, n)
    # G stores the fft of g
    G = fft(g, n)
    # The NFS of F and G are stored
    FNFS = np.absolute(F) / (2 * n)
    GNFS = np.absolute(G) / (2 * n)

    # Plots f, FNFS, g and GNFS if needed
    if(show ==1):
        fig = plt.figure()
        a1 = fig.add_subplot(221)
        a2 = fig.add_subplot(222)
        a3 = fig.add_subplot(223)
        a4 = fig.add_subplot(224)
        a1.plot(fileArray)
        a2.plot(FNFS[0 : (int)(n/2)])
        a3.plot(g)
        a4.plot(GNFS[0 : (int)(n/2)])
        plt.show()

    # Obtains the threshold for lowpass filtering
    Threshold = (int)(c * np.argmax(np.abs(G)))
    FFilter = np.copy(F)
    # Nullifies the necessary values to pass the filter
    for i in range(Threshold, len(FFilter)):
        FFilter[i] = 0
    fFilter = ifft(FFilter, n)

    # Plots f and fFilter if needed
    if(show == 1):
        fig = plt.figure()
        a1 = fig.add_subplot(211)
        a2 = fig.add_subplot(212)
        a1.plot(fileArray)
        a2.plot(fFilter)
        plt.show()

    # Prints the desired output
    print("Max frequency of |F|: ", np.argmax(np.abs(F)) )
    print("Max frequency of |G|: ", np.argmax(np.abs(G)) )
    print("Max value of f: ", np.amax(fileArray) )
    print("Max value of f^: ", "%.0f" % np.real(np.amax(fFilter)) )
        
# If an invalid function identifier is passed, show error message
else:
    print("Invalid parameter passed")
