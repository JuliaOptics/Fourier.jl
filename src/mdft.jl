# reading that the indices are out of order in the function bodies may lead
# one to conclude that this code is written backwards, but julia is column major
# and the code is correct.  If this pithy comment doesn't convince you,
# the tests that match FFTW might.

"""
    mdft(ary, samples[, shift, Q])

Compute and return the 2D DFT via a matrix triple product.
Equivalent to fftshift(fft(ifftshift(ary))) if shift==0, samples=size(ary), Q=1.

samples is the number of samples in the output domain.  If it is an integer, it
is broadcast to both dimensions.  Otherwise it may be a tuple to provide each.

shift is used to move the origin.  If the output grid with shift=0 spanned
-10:1:9 Hz, shift=10 will move the output grid to 0:1:19Hz.

Q controls the oversampling factor.  It is equivalent to zero padding the input
array by the same factor.  E.g. mdft of a 256x256 array with Q=2 is the same
as zero padding it to 512x512 and FFTing that.

See also: [`imdft`](@ref)
"""
function mdft(ary, samples::Integer; shift::Tuple{Real,Real}=(0,0), Q=1)
    return mdft(ary, (samples,samples), shift=shift, Q=Q)
end

function mdft(ary, samples::Tuple{Integer,Integer}; shift::Tuple{Real,Real}=(0,0), Q=1)
    ξ = fftrange(samples[2]);
    η = fftrange(samples[1]);
    if shift[1] != 0
        ξ += shift[1]
    end
    if shift[2] != 0
        η += shift[2]
    end
    return _mdft(ary, ξ, η, Q)
end

function _mdft(ary, ξ, η, Q)
    n, m = size(ary);
    # X,Y,ξ,η look like-128 : 127, say
    X = fftrange(m);
    Y = fftrange(n);
    kernel = -1im * 2 * π / Q;
    pre = exp.(kernel / n .* (Y * η'));
    post = exp.(kernel / m .* (X * ξ'));
    return pre' * ary * post;
end

"""
    imdft(ary, samples[, shift, Q])

Compute and return the 2D inverse DFT via a matrix triple product.
Equivalent to fftshift(ifft(ifftshift(ary))) if shift==0, samples=size(ary), Q=1.

samples is the number of samples in the output domain.  If it is an integer, it
is broadcast to both dimensions.  Otherwise it may be a tuple to provide each.

shift is used to move the origin.  If the output grid with shift=0 spanned
-10:1:9 Hz, shift=10 will move the output grid to 0:1:19Hz.

Q controls the oversampling factor.  It is equivalent to zero padding the input
array by the same factor.  E.g. mdft of a 256x256 array with Q=2 is the same
as zero padding it to 512x512 and FFTing that.

See also: [`mdft`](@ref)
"""
function imdft(ary, samples::Integer; shift::Tuple{Real,Real}=(0,0), Q=1)
    return imdft(ary, (samples,samples), shift=shift, Q=Q)
end

function imdft(ary, samples::Tuple{Integer,Integer}; shift::Tuple{Real,Real}=(0,0), Q=1)
    ξ = fftrange(samples[2]);
    η = fftrange(samples[1]);
    if shift[1] != 0
        ξ += shift[1]
    end
    if shift[2] != 0
        η += shift[2]
    end
    return _imdft(ary, ξ, η, Q)
end

function _imdft(ary, ξ, η, Q)
    n, m = size(ary);
    # X,Y,ξ,η look like-128 : 127, say
    X = fftrange(m);
    Y = fftrange(n);
    kernel = 1im * 2 * π / Q;
    pre = exp.(kernel / n .* (Y * η'));
    post = exp.(kernel / m .* (X * ξ'));
    return pre' * ary * post;
end

"""
    fftrange(n)

Computes a range that always matches FFT expectations.  I.e., for even N ranges
matches the example n=4 => -2:1:1 and for odd N matches the example 5 => -2:1:2.

This can be understood as "the output range shall always contain zero, if there
may be only one value for n÷2, it will be the negative one."

The range is always centered; frequency or sampling bins align to it with
fftshift.  It can be fftshifted to be placed in post-FFT coordinates.
"""
function fftrange(n)
    if iseven(n)
        return -(n÷2):(n÷2-1);
    end
    return -(n÷2):(n÷2);
end
