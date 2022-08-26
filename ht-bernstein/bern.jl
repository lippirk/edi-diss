using Optim
using Distributions
using Random
using Statistics
using Plots

"""
Fits a Bernstein mixture model using EM

data: data used to fit each component. if the same data is used for all
      components, then the data can be passed as a single array.
      if different data is used for each component, then the data can be
      passed as a matrix
P: number of outer mixutre components
K: number of inner Bernstein components per outer mixture component
inits: if provided, should be of the form (π=..., w=...)
"""
function fit_bernstein_mixture_EM(data::Union{AbstractMatrix,AbstractArray};
                                  P=2,
                                  inits=nothing,
                                  K=10,
                                  maxiter=2000,
                                  verbose=false,
                                  seed=nothing)
  ## arg checking
  if K isa AbstractVector && length(K) != P
    ArgumentError("length(K) != P: $(length(K)) != $P")|>throw
  elseif K isa Number
    K = [K for _ in 1:P]
  else
    ArgumentError("K must be a number or a vector")|>throw
  end
  if data isa AbstractVector
    getX = p -> data
    N = length(data)
  elseif data isa AbstractMatrix
    getX = p -> data[:,p]
    N = size(data,1)
  else
    ArgumentError("data must be a vector or matrix")|>throw
  end
  if !all(x-> 0 ≤ x && x ≤ 1, data)
    ArgumentError("data must be in [0,1]")|>throw
  end

  ## setup
  if !isnothing(seed)
    Random.seed!(seed)
  end
  if isnothing(inits)
    # use a random starting point
    ps = (π=rand(Dirichlet(ones(P) .* 10)),
          w=[rand(Dirichlet(ones(k) .* 10)) for k in K])
  else
    ps = inits
  end

  ## util
  maxK = maximum(K)

  ## define beta densities mixture components
  βs = [[Beta(j, k - j + 1) for j in 1:k] for k in K]
  # use ϕ(p,x;ws) to evaluate the pth outer mixture component
  # at x (where ws is the vector of weights)
  ϕ(p::Integer, x::Number; ws) = sum(ws .* pdf.(βs[p], x))

  ## log likelihood function
  ## not really necessary, but it is used to check for convergence
  function ll(ps)
    log.(sum([ps.π[p] .* ϕ.(p, getX(p), ws=ps.w[p]) for p in 1:P])) |> sum
  end

  # convenience wrapper
  function ass_conv(ps_old, ps_new)
    to_θ_format(ps) = [ps.π..., vcat(ps.w...)...]
    θ_old = to_θ_format(ps_old)
    θ_new = to_θ_format(ps_new)
    assess_convergence(θ_old, θ_new, ll(ps_old), ll(ps_new))
  end

  ## main EM loop
  for _i in 1:maxiter

    π_t = ps.π
    w_t = ps.w
    @assert length(π_t) == P && all(length(w_t[p]) == K[p] for p in 1:P)

    if verbose && (_i == 1 || _i % 100 == 0)
      println("========")
      println("iter=$_i")
      println("π(t)=$π_t")
      for i = 1:P
        println("w$i(t)=$(w_t[i])")
      end
      println("ll=$(ll(ps))")
    end

    ## caching (E step...)

    # _ϕ is N by P
    _ϕ = [ϕ.(p, getX(p), ws=w_t[p]) for p in 1:P] |> m_of_vv
    _βs = zeros(N, P, maxK)
    for p in 1:P
      for k in 1:K[p]
        _βs[:,p,k] = pdf.(βs[p][k], getX(p))
      end
    end

    π̃ = zeros(N, P)
    for i in 1:N
      for p in 1:P
        π̃[i,p] = π_t[p] * _ϕ[i,p]
      end
    end
    for i in 1:N
      π̃[i,:] = π̃[i,:] ./ sum(π̃[i,:])
      @assert sum(π̃[i,:]) ≈ 1
    end

    w̃ = zeros(N, P, maxK)
    for p in 1:P
      for k in 1:K[p]
        w̃[:,p,k] = w_t[p][k] .* _βs[:,p,k]
      end
    end
    for p in 1:P
      for i in 1:N
        w̃[i,p,:] = w̃[i,p,:] ./ sum(w̃[i,p,:])
        @assert sum(w̃[i,p,:]) ≈ 1
      end
    end

    ## M step
    π_new = [mean(π̃[:,p]) for p in 1:P]
    w_new = zeros(P, maxK)
    for p in 1:P
      w_new[p,1:K[p]] = [sum(π̃[:,p] .* w̃[:,p,k]) for k in 1:K[p]]
      w_new[p,:] = w_new[p,:] ./ sum(w_new[p,:])
    end

    ## new parameters
    ps_new = (π=π_new, w=[w_new[p,1:K[p]] for p in 1:P])

    ## convergence check
    conv_res = ass_conv(ps, ps_new)
    if verbose && (_i == 1 || _i % 100 == 0)
      println("convergence results:")
      println("--------------------")
      println("failed = $(conv_res.convergence_failed)")
      println("θ converged = $(all(conv_res.θ_converged))")
      println("f converged = $(conv_res.f_converged)")
      println("Δll = $(ll(ps_new) - ll(ps))")
      println("ll_new = $(ll(ps_new))")
      println("ll_old = $(ll(ps))")
    end

    if conv_res.f_converged || all(conv_res.θ_converged)
      return ps_new
    elseif conv_res.convergence_failed
      error("fit_mixture_EM: convergence failed")
    end

    ps = ps_new # use new parameters
  end
  return ps
end

"""
Fits a single mixture of Beta distributions to a data set.
In particular, it will fit

  X ~ w(1) β(1, K) + w(2) β(2, K-1) + ⋯ + w(K) β(K, 1)

For example:

```
X = rand(Beta(2, 2), 200)
fit_bernstein_EM(X, K=3)
[1.4429836073400377e-18, 0.999854801945374, 0.00014519805462602763]
# some randomness expected, since initial parameters are selected randomly
# the important thing here is that w(2) is close to 1 (expected)
```
"""
function fit_bernstein_EM(args; kwargs...)
  ps = fit_bernstein_mixture_EM(args; P=1,kwargs...)
  (w = ps.w[1],)
end

"""
Fits α and β in the model

  Y|X ~ π(1) ϕ1((Y - αX)/X^β)
      + π(2) ϕ2(Y)

where ϕ1,ϕ2 are bernstein polynomials

For example:

```
## let w1,w2 be the bernstein polynomial weights
ps = fit_α_β_in_mixture(X, Y; ϕ=[w1, w2], πs=[0.3, 0.7])
(α=0.98, β=0.1) ## result
```
"""
function fit_α_β_2_mixture(X,Y;
    ws, inits=nothing, maxiter=2000,
    πs, verbose=false,
    f01s=[sigmoid,sigmoid],
    f01invs=[logit,logit])

  αβ_to_R = logit
  R_to_αβ = sigmoid
  N = length(Y)
  @assert length(ws) == 2
  @assert sum(πs) ≈ 1
  if length(X) != N
    ArgumentError("length(X) must equal length(Y)")|>throw
  end
  if isnothing(inits)
    ps = [0.5, 0.5]
  elseif length(inits) != 2
    ArgumentError("inits must be length 2")|>throw
  else
    ps = inits
  end

  K = [length(w) for w in ws]
  maxK = maximum(K)

  # βs is maxK by 2
  βs = [[Beta(j, k - j + 1) for j in 1:k]
        for k in K] |> m_of_vv
  ws = ws |> m_of_vv # ws is maxK by 2


  function ll(ps)
    α,β=ps
    Ỹ = (Y .- α .* X) ./ abs.(X).^β

    res = 0
    for i in 1:N
      res += log.(πs[1] * sum(ws[:,1] .* pdf.(βs[:,1], f01s[1](Ỹ[i]))) +
                  πs[2] * sum(ws[:,2] .* pdf.(βs[:,2], f01s[2](Y[i]))))
    end
    return res
  end

  function ass_conv(ps_old, ps_new)
    assess_convergence(ps_old, ps_new, ll(ps_old), ll(ps_new))
  end

  _βs = zeros(N, 2, maxK)
  w̃ = zeros(N, 2, maxK)
  π̃ = zeros(N, 2)
  for _i in 1:maxiter

    αt,βt = ps
    @assert 0 <= αt <= 1
    @assert 0 <= βt <= 1

    ## cache
    for i in 1:N
      Ỹ = (Y[i] - αt * X[i])/abs(X[i])^βt
      for k in 1:K[1]
        _βs[i,1,k] = pdf(βs[k,1], f01s[1](Ỹ))
      end
      for k in 1:K[2]
        _βs[i,2,k] = pdf(βs[k,2], f01s[2](Y[i]))
      end
    end

    for i in 1:N
      π̃[i,1] = πs[1] * sum(_βs[i,1,:] .* ws[:,1])
      π̃[i,2] = πs[2] * sum(_βs[i,2,:] .* ws[:,2])

      # normalize
      π̃[i,:] = π̃[i,:] ./ sum(π̃[i,:])
    end

    for i in 1:N
      for k in 1:K[1]
        w̃[i,1,k] = ws[k,1] * _βs[i,1,k]
      end
      for k in 1:K[2]
        w̃[i,2,k] = ws[k,2] * _βs[i,2,k]
      end

      # normalize
      w̃[i,1,:] = w̃[i,1,:] ./ sum(w̃[i,1,:])
      w̃[i,2,:] = w̃[i,2,:] ./ sum(w̃[i,2,:])
    end

    ## CRUCIAL we put α and β on logit scale to avoid
    ## box optimization
    function Q(ps)
      α,β=αβ_to_R.(ps)
      Ỹ = (Y .- α .* X)./ abs.(X) .^β

      res = 0
      for i in 1:N
        for k in 1:K[1]
          if π̃[i,1] * w̃[i,1,k] ≈ 0
            # pass
            # if this product is zero, it's possible
            # that logpdf is also -Inf, and we want to avoid
            # 0 * -Inf = NaN
          else
            res += π̃[i,1] * w̃[i,1,k] * logpdf(βs[k,1], f01s[1].(Ỹ[i]))
          end
        end
        for k in 1:K[2]
          if π̃[i,2] * w̃[i,2,k] ≈ 0
            # similarly to above, pass here
          else
            res += π̃[i,2] * w̃[i,2,k] * logpdf(βs[k,2], f01s[2](Y[i]))
          end
        end
      end
      return res
    end


    optim_res = Optim.optimize(x -> -Q(x), αβ_to_R.(ps), LBFGS(),
                               autodiff=:forward)
    ps_new = R_to_αβ.(optim_res.minimizer) # remember to put α,β back on right scale
    conv_res = ass_conv(ps, ps_new)
    if conv_res.f_converged || all(conv_res.θ_converged)
      return ps_new
    elseif conv_res.convergence_failed
      error("fit_α_β_2_mixture: convergence failed")
    end

    if verbose
      println("iteration: $_i")
      println("==============")
      println("ll=$(ll(ps_new))")
      println("==============")
    end
    ps = ps_new
  end
  return ps
end

function fit_bernstein_2_mixture_with_α_β_EM(X, Y; K=5,
    α0=0.9, β0=0.1,
    f01s=[sigmoid, sigmoid], f01invs=[logit, logit],
     nouteriter=10, π0=[0.4,0.6], verbose=false)

  P = 2
  w0 = [rand(Dirichlet(ones(K).*10)) for _ in 1:P]
  α=α0; β=β0
  ps = (π=π0,w=w0)
  for n in 1:nouteriter
    α = 1; β = 0
    Y_ = (Y .- α .* X) ./ X .^ β
    data_ = hcat(f01s[1].(Y_) , f01s[2].(Y))
    ps = fit_bernstein_mixture_EM(data_,
                                  P=P,
                                  inits=ps,
                                  verbose=verbose,
                                  K=K)
    α,β = fit_α_β_2_mixture(X, Y, ws=ps.w, πs=ps.π, maxiter=100,
                            f01s=f01s,
                            f01invs=f01invs,
                            verbose=verbose)
  end
  (α=α, β=β, ps...)
end


function test_fit_α_β_2_mixture()
  data = SimulatedData.get_laplace_filtered_data()
  X = data[:,1]; Y = data[:,2]

  meanX = mean(X); meanY = mean(Y)
  stdX = std(X); stdY = std(Y)

  # f01 = cauchy_cdf
  # f01inv=cauchy_quantile
  # name = "cauchy"
  f01 = sigmoid
  f01inv = logit
  name = "sigmoid"
  # f01(x) = cdf(Normal(),x)
  # f01inv(u) = quantile(Normal(),u)
  # name = "normal"

  # f01(x) = cdf(Logistic(),x)
  # name = "logistic"
  #
  # f01(x) = cdf(Logistic(),x)
  # name = "logistic"
  #
  # f01(x) = Ftestinv(x)
  # f01inv(x) = Ftest(x)
  # name = "Ftestinv"

  α,β=[0.9,0.1]
  ps = (π=[], w=[]); Y_ = zeros(length(Y))
  N = length(Y)
  for n in 1:10
    println("n=$n")
    println("=======")
    Y_ = (Y .- α .* X) ./ abs.(X) .^ β
    data_ = hcat(f01.(Y_) , f01.(Y))
    println("fitting bernstein...")
    ps = fit_bernstein_mixture_EM(data_,
                                  P=2,
                                  K=10)
    println("fitting bernstein... DONE")
    println("π=$(ps.π), w=$(ps.w)")
    println("------")
    println("fitting α,β...")
    α,β = fit_α_β_2_mixture(X, Y, ws=ps.w, πs=ps.π, maxiter=100, f01s=(f01,f01inv))
    println("α=$α, β=$β")
    println("fitting α,β...DONE")
    println("=======")
    println()
  end


  fhat1 = ebde((w=ps.w[1],))
  fhat2 = ebde((w=ps.w[2],))
  C = (fhat1.(f01.(Y_)) .- fhat2.(f01.(Y))) .> 0
  p = Plots.histogram(vcat(f01.(Y[.!C]), f01.(Y_[C])), normalize=:pdf, alpha=0.5, label="$name(Y)", bins=20)
  # p = Plots.histogram(f01.(Y_[C]), normalize=:pdf, alpha=0.5, label="$name(Y-aX / X^b)", bins=10);
  # Plots.histogram!(p, f01.(Y[.!C]), normalize=:pdf, alpha=0.5, label="$name(Y)", bins=10)

  Plots.plot!(p, 0:0.001:1,ebde(ps).(0:0.001:1), label="bernstein density");
  # savefig(p, "/home/ben/edi/diss/code/figures/bernsteinfitting.svg")
  display(p)


  return (ps=ps,Y=Y,X=X,α=α,β=β,Y_=Y_,C=C)
end



"""
Returns the pdf of the bernstein mixture model associated
with the given parameters
"""
function ebde(ps)
  P = length(ps.π)
  K = [length(ps.w[p]) for p in 1:P]
  βs = [[Beta(j, k - j + 1) for j in 1:k] for k in K]
  return function(x)
    res = 0
    for p in 1:P
      for k in 1:K[p]
        res += ps.π[p] * ps.w[p][k] * pdf.(βs[p][k], x)
      end
    end
    return res
  end
end

function ebde(ps)
  w = ps.w
  K = length(w)
  βs = [Beta(j, K - j + 1) for j in 1:K]
  return function(x)
    res = 0
    for k in 1:K
      res += w[k] * pdf.(βs[k], x)
    end
    return res
  end
end

function bernstein_πw_fhat(ps; α,β,f01s)
  πs = ps.π; w = ps.w
  grad1(x) = gradient(f01s[1], x)[1]
  grad2(x) = gradient(f01s[2], x)[1]
  K = [length(w) for w in ps.w]
  βs = [[Beta(j, k - j + 1) for j in 1:k] for k in K]

  return function(x,y)
    ## f_Y(y | x) = G_1 (g((y - αx)/x^β) * dg/dy
    y1 = (y - α*x)/x^β
    y2 = y
    z1 = f01s[1](y1)
    z2 = f01s[2](y2)

    res = 0
    for k in 1:K[1]
      # res += ps.π[1] * ps.w[1][k] * pdf.(βs[1][k], z1) * z1 * (1 - z1) / x^β
      res += ps.π[1] * ps.w[1][k] * pdf.(βs[1][k], z1) * grad1(y1) / x^β
    end

    for k in 1:K[2]
      # res += ps.π[2] * ps.w[2][k] * pdf.(βs[2][k], z2) * z2 * (1-z2)
      res += ps.π[2] * ps.w[2][k] * pdf.(βs[2][k], z2) * grad2(y2)
    end

    return res

  end
end

function bernstein2_fhat(ps; f01=sigmoid, f01inv=logit)
  α = ps.α
  β = ps.β
  πs = ps.π
  K = [length(w) for w in ps.w]
  βs = [[Beta(j, k - j + 1) for j in 1:k] for k in K]
  grad(x) = gradient(f01, x)[1]
  return function(x,y)
    ## f_Y(y | x) = G_1 (g((y - αx)/x^β) * dg/dy
    y1 = (y - α*x)/x^β
    y2 = y
    z1 = f01(y1)
    z2 = f01(y2)

    res = 0
    for k in 1:K[1]
      # res += ps.π[1] * ps.w[1][k] * pdf.(βs[1][k], z1) * z1 * (1 - z1) / x^β
      res += ps.π[1] * ps.w[1][k] * pdf.(βs[1][k], z1) * grad(y1) / x^β
    end

    for k in 1:K[2]
      # res += ps.π[2] * ps.w[2][k] * pdf.(βs[2][k], z2) * z2 * (1-z2)
      res += ps.π[2] * ps.w[2][k] * pdf.(βs[2][k], z2) * grad(y2)
    end

    return res
  end
end

