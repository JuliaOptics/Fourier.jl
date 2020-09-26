using Test

using FFTW

using Fourier

function test_core(ary; forward)
    if forward
        truth = fftshift(fft(ifftshift(ary)));
        candidate = mdft(ary, size(ary));
        @test candidate ≈ truth
        return
    end
    truth = fftshift(ifft(ifftshift(ary)))
    candidate = imdft(ary, size(ary))./length(ary);  # match ifft convention
    @test candidate ≈ truth
end


@testset "regression against FFTW (even square array)" begin
    ary_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 100]; # includes a non power of two
    for sz in ary_sizes
        a = rand(sz,sz);
        test_core(a, forward=true)
        test_core(a, forward=false)
    end
end

@testset "regression against FFTW (odd square array)" begin
    ary_sizes = [3, 5, 11, 19, 33, 51, 101, 201, 707, 555];
    for sz in ary_sizes
        a = rand(sz,sz);
        test_core(a, forward=true)
        test_core(a, forward=false)
    end
end

@testset "regression against FFTW (nonsquare even array)" begin
    ary_sizes = [
        (2, 4),
        (4, 8),
        (4, 2),
        (8, 4),
        (100, 200),
        (200, 100),
        (256, 512),
        (512, 2)
    ];
    for sz in ary_sizes
        a = rand(sz...)
        test_core(a, forward=true)
        test_core(a, forward=false)
    end
end

@testset "regression against FFTW (nonsquare odd array)" begin
    ary_sizes = [
        (3, 5),
        (5, 7),
        (7, 9),
        (7, 101),
        (101, 7),
        (3, 9),
        (101, 201),
        (301, 201)
    ];
    for sz in ary_sizes
        a = rand(sz...)
        test_core(a, forward=true)
        test_core(a, forward=false)
    end
end


@testset "regression against FFTW (zoom, Q=2)" begin
    b = rand(2,2);
    a = zeros(4, 4);
    a[2:3,2:3]=b;
    # the comparison is slightly subtle
    # mdft does not require padding to achieve
    # sampling, so fft uses a (padded) while mdft uses b (unpadded)
    truth = fftshift(fft(ifftshift(a)));
    candidate = mdft(b, 4, Q=2);
    @test candidate ≈ truth;
    truth = fftshift(ifft(ifftshift(a)));
    candidate = imdft(b, 4, Q=2)./length(truth);
    @test candidate ≈ truth;
end

@testset "shift works as expected" begin
    data = ones(100,100);  # uniform plane has all energy in zero DFT bin
    # => bump at (50,50) before shift
    shifted_fwd = abs.(mdft(data, size(data), shift=(50,0)));
    @test argmax(shifted_fwd)[2] == 1;
    shifted_fwd = abs.(mdft(data, size(data), shift=(-49,0)));
    @test argmax(shifted_fwd)[2] == 100;

    shifted_fwd = abs.(imdft(data, size(data), shift=(50,0)));
    @test argmax(shifted_fwd)[2] == 1;
    shifted_fwd = abs.(imdft(data, size(data), shift=(-49,0)));
    @test argmax(shifted_fwd)[2] == 100;
end
