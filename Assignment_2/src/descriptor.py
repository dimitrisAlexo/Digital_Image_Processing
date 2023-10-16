import numpy as np


def get_descriptor(contour):
    # Defines a complex sequence r[i] = x[i] + jy[i], where (x[i], y[i]) are the points that describe the contour. Then
    # takes the DFT of r[i], suppose R[i]. The descriptor of the contour is defined as the absolute value of the R[i] if
    # we remove the first term, for i = 0.
    # - contour: a list of the points that make up the contour
    # - descriptor: the descriptor of the contour

    # Define a complex sequence r[i] = x[i] + jy[i] from the contour points
    r = np.array(contour)[:, 0] + 1j * np.array(contour)[:, 1]
    # Take the DFT of r[i] to get R[i]
    R = np.fft.fft(r)
    # Remove the first term R[0]
    R = R[1:]
    # Find the absolute value of the remaining terms in R[i]
    descriptor = np.abs(R)

    return descriptor
