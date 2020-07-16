#!/usr/bin/env python3

#Define a series of functions to work with f0 vectors of the format returned by parseltongue

#meant to smooth out errors in the f0 contour (might not use)
def smooth(T, nb_cw=3):
    newT = T[:]
    i = 0
    while i < (len(T)-1):
        if (T[i] == 0) and (T[i+1] != 0):
            n = i
            while (n < (len(T)-1)) and (T[n+1] != 0):
                n += 1
            window = T[i+1:n+1]
            if len(window) < nb_cw:
                for t in range(i+1, n+1):
                    newT[t] = 0
            i = n
        else:
            i+=1
    return(newT)
