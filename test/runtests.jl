using Test

using FFTW

using Fourier

@testset "regression against FFTW (even square array)" begin
    ary_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 100]; # includes a non power of two
    for sz in ary_sizes
        a = rand(sz,sz);
        truth = fftshift(fft(ifftshift(a)));
        candidate = mdft(a, size(a));
        @test truth ≈ candidate
    end
end

@testset "regression against FFTW (odd square array)" begin
    ary_sizes = [3, 5, 11, 19, 33, 51, 101, 201, 707, 555];
    for sz in ary_sizes
        a = rand(sz,sz);
        truth = fftshift(fft(ifftshift(a)));
        candidate = mdft(a, size(a));
        @test truth ≈ candidate
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
        truth = fftshift(fft(ifftshift(a)));
        candidate = mdft(a, size(a));
        @test truth ≈ candidate
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
        truth = fftshift(fft(ifftshift(a)));
        candidate = mdft(a, size(a));
        @test truth ≈ candidate
    end
end
