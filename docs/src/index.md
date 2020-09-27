# Fourier

Fourier implements functionality for performing Discrete Fourier Transforms (DFTs).  The core functionality of the package is a _highly optimized_ set of routines for performing forward and reverse DFTs of 2D data using the matrix triple product method as described in [Soummer et al (2007)](https://www.osapublishing.org/oe/abstract.cfm?uri=oe-15-24-15935).  The routines work for any `NxM` matrix for `N, M > 1` include square and nonsquare dimensions of even or odd size.  Performance is at worst on par with the FFTW invocation:
```julia
using FFTW
N=2; M=3;
a = rand(N,M);
b = a |> ifftshift |> fft |> fftshift;
```
for any `N*M <= 1024*1024`.

## Installation

Fourier is a registered Julia package, it may be installed the usual way:
```sh
julia> Pkg.add("Fourier")
```

## Functions
```@docs
mdft
imdft
```

## Utilities
```@docs
fftrange
```
