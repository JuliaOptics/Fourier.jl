using Documenter, Fourier

makedocs(modules = [Fourier], doctest = false, sitename = "Fourier")
deploydocs(repo = "github.com/JuliaOptics/Fourier.jl.git")
