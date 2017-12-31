import math
import numpy as np

def dec2bin(x):
    INT = bin(int(x))
    INT = str(INT)[2:]
    LEN = len(INT)
    x -= int(x)
    bins = ''
    while x:
        x *= 2
        bins = bins + str(1 if x>=1. else 0)
        x -= int(x)
        if len(bins) >= LEN:
            break
    return INT + '.' + bins
print(dec2bin(499.2345))
      # [1, 1, 0, 1]