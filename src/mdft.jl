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
function mdft(ary, samples::Integer, shift::Tuple{Real,Real}=(0,0), Q=1)
    return mdft(ary, (samples,samples), shift, Q)
end

function mdft(ary, samples::Tuple{Integer,Integer}, shift::Tuple{Real,Real}=(0,0), Q=1)
    U = -(samples[1]÷2):(samples[1]÷2-1);
    V = -(samples[2]÷2):(samples[2]÷2-1);
    if shift[1] != 0
        U += shift[1]
    end
    if shift[2] != 0
        V += shift[2]
    end
    return _mdft(ary, U, V, Q)
end

function _mdft(ary, U, V, Q)
    n, m = size(ary);
    # X,Y,U,V look like-128 : 127, say
    X = -(n÷2):(n÷2-1);
    Y = -(m÷2):(m÷2-1);
    kernel = -1im * 2 * π / Q;
    pre = exp.(kernel / n .* (Y * V'));
    post = exp.(kernel / m .* (X * U'));
    return pre * ary * post;
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
function imdft(ary, samples::Integer, shift::Tuple{Real,Real}=(0,0), Q=1)
    return imdft(ary, (samples,samples), shift, Q)
end

function imdft(ary, samples::Tuple{Integer,Integer}, shift::Tuple{Real,Real}=(0,0), Q=1)
    U = -(samples[1]÷2):(samples[1]÷2-1);
    V = -(samples[2]÷2):(samples[2]÷2-1);
    if shift[1] != 0
        U += shift[1]
    end
    if shift[2] != 0
        V += shift[2]
    end
    return _mdft(ary, U, V, Q)
end

function _imdft(ary, U, V, Q)
    n, m = size(ary);
    # X,Y,U,V look like-128 : 127, say
    X = -(n÷2):(n÷2-1);
    Y = -(m÷2):(m÷2-1);
    kernel = 1im * 2 * π / Q;
    pre = exp.(kernel / n .* (Y * V'));
    post = exp.(kernel / m .* (X * U'));
    return pre * ary * post;
end
