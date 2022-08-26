using Distributions
using Plots
using Random
using Optim

"""
maps ordered α ∈ [-1,1]ᴷ (α1≥α2≥...≥αK) to unordered ℝᴷ
"""
function map_αs(α)
  α_ = (α .+ 1) ./ 2
  logit.(α_ ./ [1;α_[1:end-1]])
end

"""
unmaps unordered ℝᴷ to ordered α ∈ [-1,1]ᴷ (α1≥α2≥...≥αK)
"""
function unmap_αs(α_)
  α = sigmoid.(α_)
  for k in 2:(length(α_))
    α[k] = α[k-1]*α[k]
  end
  2 .* α  .- 1
end

"""
Fits the following model using the HT normal assumption

  Y≤y|X=x ~ π1 Φ((y-α1x)/x^β1; μ1, σ1^2) +...+ πkΦ((y-αKx)/x^βK; μK, σK^2)

where Φ is the normal cdf
"""
function fit_HT_normal_K_EM(X, Y; inits=nothing, maxiter=1000, verbose=false,
                            seed=0,
                            K=2, optim_method=:lbfgs, weights = ones(length(X)),
                            σ_min = 0., γα=0.,γμ=0.,γσ=0.,γπ=0.)
  @assert optim_method == :lbfgs || optim_method == :default
  if !isnothing(seed)
    Random.seed!(seed)
  end
  n = length(X)
  if length(X) != length(Y) != length(weights)
    ArgumentError("X and Y and weights must have the same length")|>throw
  end
  if any(X .≤ 0)
    ArgumentError("X  must be positive")|>throw
  end
  if isnothing(inits)
    ps = rand(Dirichlet(K, 1))[1:end-1]#π
    αs = [rand() for _ in 1:K]|>unmap_αs; ps = [ps; αs]#α
    ps = [ps; [rand() for _ in 1:K]]#β
    ps = [ps; [5*rand() for _ in 1:K]]#μ
    ps = [ps; [1+rand() for _ in 1:K]]#σ
  else
    ps = inits
  end
  @assert length(ps) == K-1 + 4*K
  πixs = 1:K-1; αixs = K:2K-1; βixs = 2K:3K-1
  μixs = 3K:4K-1; σixs = 4K:5K-1
  shift_π(ixs) = (ixs|>collect) .- (K-1)

  function ll(ps)
    ## true ll
    π = ps[πixs]; α = ps[αixs]; β = ps[βixs]; μ = ps[μixs];
    σ = ps[σixs]
    π = [π; 1-sum(π)]
    res = 0
    for i in 1:n
      ds = Normal.(μ .* X[i] .^ β .+ α .* X[i], σ .* X[i] .^ β)
      res += weights[i] * log.((π .* pdf.(ds, Y[i]))|>sum)
    end
    res
  end

  ass_conv(ps_old, ps_new) = assess_convergence(ps_old, ps_new,
                                                ll(ps_old), ll(ps_new))

  regularize(π,α,β,μ,σ) = γα * sum(diff(α) .^2)/(K-1) +
                         -γμ * sum(μ.^2 / 100)/K +
                         -γσ * sum(1 ./ (σ .^2) .+ σ .^2 ./ 20)/K +
                         -γπ * sum(1 ./ (π .^2))/(2*(K-1))+
                         -γπ * sum(1 ./ ((1 .- π).^2))/(2*(K-1))


  function mkQ(ps)
    π = ps[πixs]
    π = [π; 1-sum(π)]
    α = ps[αixs]
    β = ps[βixs]
    μ = ps[μixs]
    σ = ps[σixs]

    ### M step
    ϕ = zeros(n, K)
    π̃ = zeros(n, K)
    for k in 1:K
      ds = Normal.(μ[k] .* X .^ β[k] .+ α[k] .* X, σ[k] .* X .^ β[k])
      ϕ[:,k] = pdf.(ds, Y) .* weights
    end

    @assert all(π .≥ 0)
    @assert all(ϕ .≥ 0)

    πdenom = zeros(n)
    for i in 1:n
      πdenom[i] = sum(π .* ϕ[i,:])
    end

    for k in 1:K-1
      π̃[:,k] = π[k] .* ϕ[:,k] ./ πdenom
      π̃[:,k] = ifelse.(isnan.(π̃[:,k]), 0, π̃[:,k])
    end
    for i in 1:n
      π̃[i,K] = 1 - sum(π̃[i,1:K-1])
    end

    for i in eachindex(π̃)
      if isnan(π̃[i])
        error("π̃[$i] is nan")
      elseif π̃[i] < 0 || π̃[i] > 1
        @warn "π̃[$i] is out of bounds (correcting...)"
        π̃[i] = maximum([π̃[i], 0.])
        π̃[i] = minimum([π̃[i], 1.])
      end
    end

    ## E-step
    πnew = zeros(K)
    for k in 1:K-1
      πnew[k] = mean(π̃[:,k])
    end
    πnew[K] = 1-sum(πnew[1:end-1])

    ## Q(⋅, θᵗ) -- E-step
    function Q(ps)
      # π = ps[πixs]
      π = πnew
      α = ps[αixs|>shift_π]
      β = ps[βixs|>shift_π]
      μ = ps[μixs|>shift_π]
      σ = ps[σixs|>shift_π]

      # put parameters on boxed scale
      σ = exp.(σ) .+ σ_min
      α = unmap_αs(α)
      β = sigmoid.(β)

      res = 0
      # lik
      for k in 1:K
        ds = Normal.(μ[k] .* X .^ β[k] .+ α[k] .* X, σ[k] .* X .^ β[k])
        res += weights .* π̃[:,k] .* (log(π[k]) .+ logpdf.(ds, Y))  |> sum
      end
      res += n * regularize(π,α,β,μ,σ)
      res
    end
    return (πnew, Q)
  end

  for i ∈ 1:maxiter
    πnew,Q = mkQ(ps)
    # unboxed scale (this is for the optimizer)
    ps_new_scale =[map_αs(ps[αixs]); logit.(ps[βixs]);
                   ps[μixs]; log.(ps[σixs] .- σ_min)]
    if verbose
      println("iter=$i")
      println("====")
      println("π = $(round.(ps[πixs], digits=4))")
      println("α = $(round.(ps[αixs], digits=4))")
      println("β = $(round.(ps[βixs], digits=4))")
      println("μ = $(round.(ps[μixs], digits=4))")
      println("σ = $(round.(ps[σixs], digits=4))")
      println("Q = $(Q(ps_new_scale))")
      println("====\n")
    end
    res = nothing
    if optim_method == :lbfgs
      res = Optim.optimize(x -> -Q(x), ps_new_scale, LBFGS(), autodiff=:forward);
    elseif optim_method == :default
      res = Optim.optimize(x -> -Q(x), ps_new_scale);
    end
    _tmp = res.minimizer
    # back to boxed scale
    ps_new = [πnew[1:K-1];
              unmap_αs(_tmp[αixs|>shift_π]);
              sigmoid.(_tmp[βixs|>shift_π]);
              _tmp[μixs|>shift_π];
              exp.(_tmp[σixs|>shift_π]) .+ σ_min]

    conv_res = ass_conv(ps, ps_new)
    ps = ps_new

    @assert all(ps[σixs] .≥ 0)
    if all(conv_res.θ_converged)
      break
    end
  end

  if n ≥ maxiter
    @warn "maxiter=$maxiter reached, may not have converged..."
  end
  π = ps[πixs]
  (π=[π;(1-sum(π))], α=ps[αixs], β=ps[βixs], μ=ps[μixs], σ=ps[σixs],ll=ll(ps))
end


function HT_normal_fhat(ps)
  π = ps.π; α = ps.α; β = ps.β; μ = ps.μ; σ = ps.σ
  ds = Normal.(μ, σ)
  return function(x, y)
    z = (y .- α .* x) ./ x .^ β
    return sum(π .* pdf.(ds, z))
  end
end


## WIP
# """
# Fits the following model using the HT normal assumption

  # X3≤x3|X2=x2,X1=x1 ~ π1Φ(x3-a1(x1,x2;ν) ; μ1, σ1^2) +
                      # π2Φ((x3-α2x1)/x1^β2; μ2, σ2^2) +
                      # π3Φ((x3-α3x2)/x2^β3; μ3, σ3^2) +
                      # π4Φ(x3             ; μ4, σ4^2) +
                      # π5Φ(x3             ; μ5, σ5^2) +
                      # π6Φ(x3             ; μ6, σ6^2) +

# where Φ is the normal cdf
# """
# function fit_HT_normal_6_mix_asym_log(X1, X2, X3;
                                      # inits=nothing,
                                      # maxiter=10000,
                                      # verbose=false,
                                      # seed=0)
  # a11(x1, x2, α)= -0.5 * log(exp(-x1/0.5) + exp(-x2/0.5))
  # Random.seed!(seed)
  # n = length(X1)
  # if length(X2) != n || length(X3) != n
    # ArgumentError("X and Y must have the same length")|>throw
  # end
  # if isnothing(inits)
    # ps = rand(Dirichlet(ones(6) .* 3))[1:end-1]#π (remove rest because it's determined by previous
    # # ps = [ps; [rand()]]#ν
    # ## TODO remove
    # ps = [ps; [0.5]]#ν
    # αs = [rand() for _ in 1:2]|>unmap_αs; ps = [ps; αs]#α
    # ps = [ps; [rand() for _ in 1:2]]#β
    # ps = [ps; [5*rand() for _ in 1:6]]#μ
    # ps = [ps; [3+rand() for _ in 1:6]]#σ
  # else
    # ps = inits
  # end
  # @assert length(ps) == 5+#π
                        # 1+#ν
                        # 2+#α
                        # 2+#β
                        # 6 + 6#μ,σ
  # πixs = 1:5; νix = 6; αixs = 7:8; βixs = 9:10
  # μixs = 11:16; σixs = 17:22
  # shift_π(ixs) = (ixs|>collect) .- 5

  # function ll(ps)
    # ## true ll
    # π = ps[πixs]; ν = ps[νix]; α = ps[αixs];
    # β = ps[βixs]; μ = ps[μixs]; σ = ps[σixs]
    # π = [π; 1-sum(π)]
    # res = 0
    # for i in 1:n
      # x3 = X3[i]; x2 = X2[i]; x1 = X1[i]
      # acc =  x1 ≤ 0. || x2 ≤ 0. ? 0 : π[1] * pdf(Normal(μ[1] + a11(x1,x2,ν)         , σ[1])             , x3)
      # acc += x1 ≤ 0. ? 0 :            π[2] * pdf(Normal(μ[2] * x1 ^ β[1] + α[1] * x1, σ[2] .* x1 ^ β[1]), x3)
      # acc += x2 ≤ 0. ? 0 :            π[3] * pdf(Normal(μ[3] * x2 ^ β[2] + α[2] * x2, σ[3] .* x2 ^ β[2]), x3)
      # acc +=                          π[4] * pdf(Normal(μ[4]                        , σ[4])             , x3)
      # acc +=                          π[5] * pdf(Normal(μ[5]                        , σ[5])             , x3)
      # acc +=                          π[6] * pdf(Normal(μ[6]                        , σ[6])             , x3)
      # res += log(acc)
    # end
    # res
  # end

  # ass_conv(ps_old, ps_new) = assess_convergence(ps_old, ps_new,
                                                # ll(ps_old), ll(ps_new),
                                                 # atol=1e-6, rtol=1e-6,
                                                # θatol=1e-6, θrtol=1e-6)

  # function mkQ(ps)
    # π = ps[πixs]
    # π = [π; 1-sum(π)]
    # ν = ps[νix]
    # α = ps[αixs]
    # β = ps[βixs]
    # μ = ps[μixs]
    # σ = ps[σixs]

    # K = 6

    # ### M step
    # ϕ = zeros(n, K)
    # π̃ = zeros(n, K)
    # for i in 1:n
      # x3 = X3[i]; x2 = X2[i]; x1 = X1[i]
      # ϕ[i,1] = x1 ≤ 0. || x2 ≤ 0. ? 0. : π[1] * pdf(Normal(μ[1] + a11(x1,x2,ν)         , σ[1])            , x3)
      # ϕ[i,2] = x1 ≤ 0. ? 0. :            π[2] * pdf(Normal(μ[2] * x1 ^ β[1] + α[1] * x1, σ[2] * x1 ^ β[1]), x3)
      # ϕ[i,3] = x2 ≤ 0. ? 0. :            π[3] * pdf(Normal(μ[3] * x2 ^ β[2] + α[2] * x2, σ[3] * x2 ^ β[2]), x3)
      # ϕ[i,4] =                           π[4] * pdf(Normal(μ[4]                        , σ[4])            , x3)
      # ϕ[i,5] =                           π[5] * pdf(Normal(μ[5]                        , σ[5])            , x3)
      # ϕ[i,6] =                           π[6] * pdf(Normal(μ[6]                        , σ[6])            , x3)
    # end

    # # ϕ = ifelse.(ϕ .≤ 0 .|| isnan.(ϕ), 0., ϕ)
    # @assert all(π .≥ 0)
    # @assert all(ϕ .≥ 0)


    # πdenom = zeros(n)
    # for i in 1:n
      # πdenom[i] = sum(π .* ϕ[i,:])
    # end

    # for k in 1:K-1
      # π̃[:,k] = π[k] .* ϕ[:,k] ./ πdenom
      # π̃[:,k] = ifelse.(isnan.(π̃[:,k]), 0., π̃[:,k])
    # end
    # for i in 1:n
      # π̃[i,K] = 1 - sum(π̃[i,1:K-1])
    # end

    # bounds_fail = false
    # for i in eachindex(π̃)
      # if isnan(π̃[i])
        # error("π̃[$i] is nan")
      # elseif π̃[i] < 0 || π̃[i] > 1
        # @warn "π̃[$i] is out of bounds (correcting...)"
        # π̃[i] = maximum([π̃[i], 0.])
        # π̃[i] = minimum([π̃[i], 1.])
        # bounds_fail = true
      # end
    # end
    # if bounds_fail # perform adjustment again...
      # @warn "bounds_failed..."
      # for i in 1:n
        # π̃[i,K] = 1 - sum(π̃[i,1:K-1])
      # end
      # for i in 1:n
        # if !(π̃[i,:]|>isprobvec)
          # π̃[i,:] = [1/K for _ in 1:K]
        # end
      # end
    # end

    # ## E-step
    # πnew = zeros(K)
    # for k in 1:K-1
      # πnew[k] = mean(π̃[:,k])
    # end
    # πnew[K] = 1-sum(πnew[1:end-1])

    # ## Q(⋅, θᵗ) -- E-step
    # function Q(ps)
      # π = πnew
      # ν = ps[νix|>shift_π]
      # α = ps[αixs|>shift_π]
      # β = ps[βixs|>shift_π]
      # μ = ps[μixs|>shift_π]
      # σ = ps[σixs|>shift_π]

      # # put parameters on boxed scale
      # ν = sigmoid(ν)
      # σ = exp.(σ)
      # α = unmap_αs(α)
      # β = sigmoid.(β)

      # res = 0
      # for i in 1:n
        # x1 = X1[i]; x2 = X2[i]; x3 = X3[i]
        # d1 = x1 ≤ 0. || x2 ≤ 0. ? nothing : Normal(μ[1] + a11(x1,x2,ν)         , σ[1])
        # d2 = x1 ≤ 0. ? nothing :            Normal(μ[2] * x1 ^ β[1] + α[1] * x1, σ[2] .* x1 ^ β[1])
        # d3 = x2 ≤ 0. ? nothing :            Normal(μ[3] * x2 ^ β[2] + α[2] * x2, σ[3] .* x2 ^ β[2])
        # d4 =                                Normal(μ[4]                        , σ[4])
        # d5 =                                Normal(μ[5]                        , σ[5])
        # d6 =                                Normal(μ[6]                        , σ[6])
        # res += isnothing(d1) ? 0 : π̃[i,1] * (log(π[1]) + logpdf(d1, x3))
        # res += isnothing(d2) ? 0 : π̃[i,2] * (log(π[2]) + logpdf(d2, x3))
        # res += isnothing(d3) ? 0 : π̃[i,3] * (log(π[3]) + logpdf(d3, x3))
        # res += π̃[i,4] * (log(π[4]) + logpdf(d4, x3))
        # res += π̃[i,5] * (log(π[5]) + logpdf(d5, x3))
        # res += π̃[i,6] * (log(π[6]) + logpdf(d6, x3))
      # end
      # res
    # end
    # return (πnew, Q)
  # end

  # for _i ∈ 1:maxiter
    # πnew,Q = mkQ(ps)
    # # unboxed scale (this is for the optimizer)
    # ps_new_scale =[[logit(ps[νix])]; map_αs(ps[αixs]);
                   # logit.(ps[βixs]);
                   # ps[μixs]; log.(ps[σixs])]
    # if verbose
      # println("iter=$_i")
      # println("====")
      # println("π = $(round.(ps[πixs], digits=4))")
      # println("ν = $(round.(ps[νix],  digits=4))")
      # println("α = $(round.(ps[αixs], digits=4))")
      # println("β = $(round.(ps[βixs], digits=4))")
      # println("μ = $(round.(ps[μixs], digits=4))")
      # println("σ = $(round.(ps[σixs], digits=4))")
      # println("Q=$(Q(ps_new_scale))")
      # println("====\n")
    # end
    # res = Optim.optimize(x -> -Q(x), ps_new_scale, LBFGS(), autodiff=:forward);
    # # res = Optim.optimize(x -> -Q(x), ps_new_scale);
    # _tmp = res.minimizer
    # # back to boxed scale
    # ps_new = [πnew[1:5];
              # sigmoid(_tmp[νix|>shift_π]);
              # unmap_αs(_tmp[αixs|>shift_π]);
              # sigmoid.(_tmp[βixs|>shift_π]);
              # _tmp[μixs|>shift_π];
              # exp.(_tmp[σixs|>shift_π])]

    # conv_res = ass_conv(ps, ps_new)
    # ps = ps_new
    # if all(conv_res.θ_converged)
      # break
    # end
  # end

  # if n ≥ maxiter
    # @warn "maxiter=$maxiter reached, may not have converged..."
  # end
  # π = ps[πixs]
  # (π=[π;(1-sum(π))], ν=ps[νix], α=ps[αixs], β=ps[βixs], μ=ps[μixs], σ=ps[σixs])
# end

"""
Returns the assignment of a set of data points, according to the normal HT
model above:

Example:

```
ps =fit_HT_normal_K_EM(X,Y,K=3)
C = class_or_normal_ps(ps,X,Y,K=3)
# [ 3, 3, 2, 1, ... ]

@assert length(C) == length(X) == length(Y)
```
"""
function class_of_normal_ps(ps,X,Y;K=2,random=false)
  @assert length(X) == length(Y)
  n = length(X)
  π = ps.π; α = ps.α; β = ps.β; μ = ps.μ; σ = ps.σ;

  if !random
    C = zeros(n)
    for i in 1:n
      logpdfs = zeros(K)
      for k in 1:K
        d = Normal(μ[k]*X[i]^β[k]+α[k]*X[i], σ[k]*X[i]^β[k])
        logpdfs[k] = log(π[k]) + logpdf(d, Y[i])
        C[i] = argmax(logpdfs)
      end
    end
    C |> c->map(Int,c)
  else
    C = zeros(n)
    for i in 1:n
      pdfs = zeros(K)
      for k in 1:K
        d = Normal(μ[k]*X[i]^β[k]+α[k]*X[i], σ[k]*X[i]^β[k])
        pdfs[k] = π[k]*pdf(d, Y[i])
      end
      pdfs[:] = pdfs[:]/sum(pdfs)
      @assert isprobvec(pdfs)
      C[i] = rand(Categorical(pdfs))
    end
    C |> c->map(Int,c)
  end
end

function ht_pdf(ps)
  K=length(ps.π);
  π = ps.π
  μ = ps.μ; σ = ps.σ
  α = ps.α; β = ps.β;
  return function(x::Real, y::Real)
    μy =  μ .* x .^ β + α .* x ; σy = σ .* x .^ β ;
    ds = Normal.(μy, σy);
    res = sum(π .* pdf.(ds, y))
    return res
  end
end

function ht_loglik(X,Y;K)
  @assert length(X) == length(Y)
  return function(π,α,β,μ,σ)
    @assert length(π) == K-1
    res = 0
    for i in eachindex(X)
      x = X[i]; y = Y[i]
      μy =  μ .* x .^ β + α .* x ; σy = σ .* x .^ β ;
      ds = Normal.(μy, σy);
      res += sum([π; 1-sum(π)] .* pdf.(ds, y))
    end
    return res
  end
end


function quickplot(ps, x1, x2; u, xmax=10, ymax=10, ymin=-10)
  K = ps.α|>length
  C = class_of_normal_ps(ps, x1, x2, K=K)

  _ht_pdf = ht_pdf(ps)

  p = Plots.contour(u:0.05:xmax,ymin:0.05:ymax,(x,y)->_ht_pdf(x,y), alpha=0.5, color=:jet);
  colors=[:red, :green, :black]
  for k in 1:K
    Plots.scatter!(p,x1[C.==k], x2[C.==k], color=colors[k], xlim=[u,xmax], ylim=[ymin,ymax])
  end
  display(p)
end

function sample_Zs(ps, X, Y; M = 10)
  @assert length(X) == length(Y)
  K = length(ps.α)
  n = length(X)
  π = ps.π; α = ps.α; β = ps.β; μ = ps.μ; σ = ps.σ;
  d = [Float64[] for _ in 1:K]

  for m in 1:M
    for i in 1:n
      pdfs = zeros(K)
      for k in 1:K
        dist = Normal(μ[k]*X[i]^β[k]+α[k]*X[i], σ[k]*X[i]^β[k])
        pdfs[k] = π[k] * pdf(dist, Y[i])
      end
      pdfs[:] = pdfs[:]/sum(pdfs)
      if !isprobvec(pdfs)
        pdfs[:] = ps.π
      end
      @assert isprobvec(pdfs)
      C = rand(Categorical(pdfs))
      push!(d[C], (Y[i] - α[C]*X[i])/X[i]^β[C])
    end
  end
  Tuple(values(d))
end
