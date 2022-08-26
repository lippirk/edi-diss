using Turing
using Distributions
using StatsPlots
using DynamicHMC
using AdvancedVI
using DistributionsAD

# @model function __HT_turing_mixture_model(X, Y, K)
  # n = length(X)
  # C = tzeros(Int,n)

  # # π ~ Dirichlet(K, 1.) # use π = [1/K,…,1/K] prior
  # σ ~ filldist(Uniform(1, 10), K)
  # μ ~ filldist(Normal(), K)
  # β ~ filldist(Uniform(), K)
  # α_ ~ filldist(Uniform(), K)
  # α = sort(α_, rev=true)
  # # α = arraydist([Uniform((k-1)/K, k/K) for k in 1:K])
  # # α ~ arraydist([Uniform(0, 0.5), Uniform(0.5, 1)])
  # # α ~ arraydist([Uniform(0, 1) for k in 1:K])

  # ds = [Normal(μ[k], σ[k]) for k in 1:K]
  # # C = Vector{Int}(undef, n)
  # for i in 1:n
    # x = X[i]; y = Y[i]
    # z = (y .- α .* x) ./ x .^ β
    # # zs = [y]

    # ## bayes probs
    # # v = π .* pdf.(Normal.(μ .* x .^ β .+ α .* x, σ .* x .^ β), y)
    # # if isnan.(v)|>any
      # # @warn "v=$v is nan"
    # # end
    # # if any(isnan.(v)) || !isprobvec(v)
      # # p̃ = [1/K for k in 1:K]
    # # else
      # # p̃ = v ./ sum(v); p̃[end] = 1-sum(p̃[1:end-1])
    # # end
    # v = pdf.(Normal.(μ .* x .^ β .+ α .* x, σ .* x .^ β), y)
    # if sum(v) ≈ 0
      # p̃ = [1/K for k in 1:K]
    # else
      # p̃ = v ./ sum(v); p̃[end] = 1-sum(p̃[1:end-1])
    # end
    # if isnan.(p̃)|>any
      # @warn "p̃=$p̃ is nan"
    # end


    # ## sample a class for the mixture
    # # C = Categorical(p̃)|>rand
    # # C[i] = Categorical(π)|>rand
    # # C[i] ~ Categorical(π)
    # # C = Categorical(π)|>rand

    # ##
    # # y ~ Normal(μ[C] * x ^ β[C] + α[C] * x, σ[C] * x ^ β[C])

    # for i in 1:n
      # # C ~ Categorical(p̃)
      # z ~ ds[C]
      # # ds[k] = Normal(μ[k] * x ^ β[k] + α[k] * x, σ[k] * x ^ β[k])
    # end
    # # z[C[i]] ~ ds[C[i]]
    # # y ~ Normal(0, 1)
  # end
  # return π,α,β,μ,σ
# end

# @model function _HT_turing_mixture_model(X, Y, K)
  # n = length(X)
  # C = tzeros(Int,n)
  # # π ~ Dirichlet(ones(K))

  # # π ~ Dirichlet(K, 1.) # use π = [1/K,…,1/K] prior
  # # σ ~ filldist(Uniform(1e-1, 5), K)
  # σ ~ filldist(InverseGamma(2,3), K) #
  # μ ~ filldist(Normal(), K)
  # β ~ filldist(Uniform(), K)
  # # α ~ filldist(Beta(2,2), K)
  # α ~ arraydist([Beta(k+1,K-k+2) for k in 1:K])
  # # β ~ filldist(Uniform(), K)
  # # α = sort(α_, rev=true)

  # for i in 1:n
    # x = X[i]; y = Y[i]
    # μy = μ .* x .^ β .+ α .* x
    # σy = sqrt.(σ) .* x .^ β

    # v = pdf.(Normal.(μy, σy), y)
    # if sum(v) ≈ 0
      # p̃ = [1/K for k in 1:K]
    # else
      # p̃ = v ./ sum(v); p̃[end] = 1-sum(p̃[1:end-1])
    # end

    # # C[i] ~ Categorical(π)
    # C[i] ~ Categorical(p̃)
    # y ~ Normal(μy[C[i]], σy[C[i]])
  # end
  # return (α,β,μ,σ)
# end

@model function _HT_turing_mixture_model(X, Y, K)
  n = length(X)
  C = tzeros(Int,n)
  π ~ Dirichlet(ones(K).*10)
  a = 200; b = 200
  σ ~ filldist(InverseGamma(a,b), K)
  # σ ~ filldist(Uniform(0.1,15), K)
  μ ~ filldist(Normal(0, 1), K)
  β ~ filldist(Beta(0.5,0.5), K)
  κ = 20
  α ~ arraydist([Beta(1+κ,1), Beta(1,1+κ)])

  for i in 1:n
    x = X[i]; y = Y[i]
    C[i] ~ Categorical(π)
    y ~ Normal(μ[C[i]]*x^β[C[i]]+α[C[i]]*x, σ[C[i]]*x^β[C[i]])
  end
  return (π,α,β,μ,σ)
end

"""
Fits the following model using the HT normal assumption

  Y≤y|X=x ~ π1 Φ((y-α1x)/x^β1); μ1, σ1^2) +...+ πkΦ((y-αKx)/x^βK); μK, σK^2)

where Φ is the normal cdf
"""
function fit_HT_normal_K_turing(X, Y; inits=nothing, nchain=1_000, verbose=false,
                                  seed=0, K=2)
  Random.seed!(seed)
  model = _HT_turing_mixture_model(X, Y, K)
  # sample(model, HMC(1e-2, 5), nchain)
  # sample(model, HMC(1e-2, 10), nchain)
  # sample(model, Gibbs(PG(100,:C), HMC(0.05, 10,:π,:α_,:β,:μ,:σ)), nchain)
  # sample(model, Gibbs(PG(100), HMC(0.01, 10)), nchain)
  # sample(model, PG(20,200), nchain)
  # sample(model, HMC(0.01, 5), nchain)
  # q = vi(model, ADVI(100, 1000))
  chn = sample(model, Gibbs(PG(50, :C), HMC(0.01,5, :π,:α,:β,:μ,:σ)), nchain)
  # sample(model, Gibbs(HMC(0.1, 5),PG(30)), nchain)
  # chn = sample(model, Gibbs(HMC(1e-1, 1, :π,:α,:β,:μ,:σ), PG(20, :C)), nchain)
  # chn = sample(model, Gibbs(PG(100,:C,:κ), HMC(0.05,10,:π,:α,:β,:μ,:σ)), nchain)
  # chn = sample(model, Gibbs(PG(5,:C,:κ), MH()), nchain)
  # chn = sample(model, Gibbs(HMC(0.01, 5, :π,:μ,:σ,:α,:β),
                            # PG(10, :C)),
               # nchain)
  # chn = sample(model, SMC(1000), nchain)

  chn = chn[["π[1]","π[2]","α[1]","α[2]","β[1]","β[2]","μ[1]","μ[2]","σ[1]","σ[2]"]]
  # chn = chn[["α[1]","α[2]","β[1]","β[2]","μ[1]","μ[2]","σ[1]","σ[2]","κ"]]
  chn
  # q
end

function fit_HT_normal_vi(X, Y; inits=nothing, nchain=1_000, verbose=false,
                                  seed=0, K=2)
  Random.seed!(seed)
  model = _HT_turing_mixture_model(X, Y, K)
  q = vi(model, ADVI(100, 1000))
  q
end

