using Zygote
using Random
using LinearAlgebra
using Roots
using Random
using Distributions
using Distributed
using ProgressMeter
using CSV
using DataFrames

function test_inverse(f, finv; us=nothing, skip=false)
  if skip
    return true
  end;
  # println("test_inverse: running tests...")
  if isnothing(us)
    us = [1e-5:1e-5:0.0001; 0.9999:1e-5:0.99999]
  end
  for u in us
    try
      if f(finv(u)) ≈ u
        continue
      else
        println("test_inverse failed at u=$u")
        return false
      end;
    catch e
        showerror(stderr, e)
        println("\ntest_inverse failed at u=$u")
        return false
    end;
  end;
  return true
end

function check_params(θ,ν)
  @assert all(z -> z ≤ 1, ν)
  @assert sum(θ[[1,4,5,6]]) ≈ 1
  @assert dot(θ[[2,4,6]], [1,2,1]) ≈ 1
  @assert sum(θ[3:6]) ≈ 1
end

function norm(x; ν, η=ν)
  sum(x .^ (-1/ν))^η
end

function V(x::Vector{<:Real}, θ, ν)
  @assert length(x) == 3
  @assert length(θ) == 6
  @assert length(ν) == 3
  if any(z->z==0,x)
    Inf
  else
    dot(1 ./ x, θ[1:3]) +
    θ[4]*(norm(x[[1,2]], ν=ν[1]) + norm(x[[2,3]], ν=ν[1])) +
    θ[5]*norm(x[[1,3]], ν=ν[2]) +
    θ[6]*norm(x, ν=ν[3])
  end;
end

function ∂V∂0(x::Vector{<:Real}, θ, ν)
  if any(z->z==0,x)
    Inf
  else
    -θ[1]*x[1]^-2 -
    θ[4]*x[1]^(-1/ν[1]-1)*norm(x[[1,2]], ν=ν[1], η=ν[1]-1) -
    θ[5]*x[1]^(-1/ν[2]-1)*norm(x[[1,3]], ν=ν[2], η=ν[2]-1) -
    θ[6]*x[1]^(-1/ν[3]-1)*norm(x, ν=ν[3], η=ν[3]-1)
  end;
end

function ∂V∂1(x::Vector{<:Real}, θ, ν)
  if any(z->z==0,x)
    Inf
  else
    -θ[2]*x[2]^-2 -
    θ[4]*x[2]^(-1/ν[1]-1)*(norm(x[[1,2]], ν=ν[1], η=ν[1]-1) + norm(x[[2,3]], ν=ν[1], η=ν[1]-1)) -
    θ[6]*x[2]^(-1/ν[3]-1)*norm(x, ν=ν[3], η=ν[3]-1)
  end;
end

function ∂2V∂0∂1(x::Vector{<:Real}, θ, ν)
  if any(z->z==0,x)
    Inf
  else
    θ[4]*x[1]^(-1/ν[1]-1)*x[2]^(-1/ν[1]-1)*norm(x[[1,2]],ν=ν[1],η=ν[1]-2)*((ν[1]-1)/ν[1]) +
    θ[6]*x[1]^(-1/ν[3]-1)*x[2]^(-1/ν[3]-1)*norm(x,ν=ν[3],η=ν[3]-2)*((ν[3]-1)/ν[3])
  end;
end

function gen_X_f(;θ,ν,skip=false)
  check_params(θ,ν)

  #### V
  V_(x0::Real,x1::Real,x2::Real) = V([x0,x1,x2],θ,ν)
  dV0(x0,x1,x2) = ∂V∂0([x0,x1,x2],θ,ν)
  dV1(x0,x1,x2) = ∂V∂1([x0,x1,x2],θ,ν)
  d2V01(x0,x1,x2) = ∂2V∂0∂1([x0,x1,x2],θ,ν)

  #### F
  F(x::Vector{<:Real}) = exp(-V_(x[1],x[2],x[3]))
  F(x0::Real,x1::Real,x2::Real) = F([x0,x1,x2])

  #### generate values from F0
  F0(x::Real) = F(x, Inf, Inf)
  F0_inv(u::Real) = -sum(θ[[1,4,5,6]])/log(u)
  @assert test_inverse(F0, F0_inv,skip=skip)

  #### generate values from F_{1|0}
  function F1_0(x, x0)
    # work on log scale and then exp up
    res = -V_(x0, x, Inf) + V_(x0, Inf, Inf)
    res += log(-dV0(x0, x, Inf))
    res -= log(-dV0(x0, Inf, Inf))
    exp(res)
  end
  function F1_0_inv(u::Real, x0::Real)
    # we want to find the zero in the domain [0,∞] but
    # the solver prefers it if we instead solve in the domain [0,1].
    # we do this by solving f(x/(1-x))=0, x∈[0,1]. finally, map that
    # back to the real line with x/(1-x)
    function f(x)
      if x≈0
        -u
      elseif x≈1
        1-u
      else
        y = x/(1-x)
        F1_0(y, x0) - u
      end
    end
    res = fzero(f, (0,1))
    res/(1-res)
  end
  @assert test_inverse(x->F1_0(x,1e-4), x->F1_0_inv(x,1e-4),skip=skip)
  @assert test_inverse(x->F1_0(x,1e4), x->F1_0_inv(x,1e4),skip=skip)

  #### generate values from F_{2|0,1}
  function F2_01(x::Real, x0::Real, x1::Real)
    # work on log scale and then exp up
    res = -V_(x0, x1, x) + V_(x0, x1, Inf)
    a = -dV0(x0,x1,x)*dV1(x0,x1,x)+d2V01(x0,x1,x)
    b = -dV0(x0,x1,Inf)*dV1(x0,x1,Inf)+d2V01(x0,x1,Inf)
    @assert sign(a) == sign(b)
    res += log(abs(a))
    res -= log(abs(b))
    exp(res)
  end
  function F2_01_inv(u::Real, x0::Real, x1::Real)
    function f(x)
      if x≈0
        -u
      elseif x≈1
        1-u
      else
        y = x/(1-x)
        F2_01(y, x0, x1) - u
      end
    end
    res = fzero(f, (1e-8,1-1e-8))
    res/(1-res)
  end
  @assert test_inverse(x->F2_01(x,1e4,1e4), x->F2_01_inv(x,1e4,1e4),skip=skip)
  @assert test_inverse(x->F2_01(x,1e-4,1e-4), x->F2_01_inv(x,1e-4,1e-4),skip=skip)
  function _gen(;n,seed=nothing,q=0.999)
    #### generate values from the markov chain
    X = zeros(n)
    if !isnothing(seed)
      Random.seed!(seed)
    end
    next_u = () -> rand(Uniform(), 1)[1]
    X[1] = quantile(Frechet(), next_u()*(1-q)+q)
    X[2] = F1_0_inv(next_u(), X[1])
    @showprogress for i ∈ 3:length(X)
      X[i] = F2_01_inv(next_u(), X[i-2], X[i-1])
    end

    X
  end
  return _gen
end

function _gen_X(;θ,ν,n,skip=false,seed=nothing,y0_thresh=9)
  gen_ = gen_X_Y_f(θ=θ,ν=ν,skip=skip)
  gen_(;n=n,seed=seed,y0_thresh=y0_thresh)
end

function gen_Xs(;M=100,N=30,q=0.995,
                 seed=1,θ=[0.3,0.3,0.3,0.3,0.3,0.1],ν=[0.5,0.5,0.5])
  h = hash([M,N,seed,q,θ...,ν...])
  fname = "data/$h.csv"
  if isfile(fname)
    Xs = CSV.read(fname, DataFrame)|>Matrix
  else
    gen = gen_X_f(;θ=θ,ν=ν)
    Xs = zeros(M,N)
    Random.seed!(seed)
    @showprogress for m in 1:M
      X = gen(n=N,q=q, seed=nothing)
      Xs[m,:] = X
    end
    CSV.write("data/$h.csv", DataFrame(Xs,:auto))
  end
  Xs
end

function frechet_to_exp(x)
    -log.(1 .- exp.(-1 ./ x))
end

function quick_test_data_2d(;scale=Laplace(),u=3,
  θ=[0.3,0.3,0.3,0.3,0.3,0.1],ν=[0.5,0.5,0.5])
  Xs = gen_Xs(θ=θ,ν=ν)
  Xs = quantile.(scale, cdf.(Frechet(), Xs))# move to exp scale
  u_ixs1 = Xs.>u
  u_ixs1[:,end] .= 0 # don't include last point in each series
  u_ixs2 = mapslices(x->x>>1, u_ixs1, dims=2) # shift each row to the right

  x2 = Xs[u_ixs2]; x1 = Xs[u_ixs1]
  return (x1, x2)
end

function quick_test_data_3d(;scale=Laplace(),u=3)
  Xs = gen_Xs()
  Xs = quantile.(scale, cdf.(Frechet(), Xs))# move to exp scale
  u_ixs1 = Xs.>u
  u_ixs1 = u_ixs1 .|| mapslices(x->x<<1, u_ixs1, dims=2)
  u_ixs1[:,end-1:end] .= 0 # don't include last point in each series
  u_ixs2 = mapslices(x->x>>1, u_ixs1, dims=2) # shift each row to the right
  u_ixs3 = mapslices(x->x>>2, u_ixs1, dims=2) # shift each row to the right

  x3 = Xs[u_ixs3]; x2 = Xs[u_ixs2]; x1 = Xs[u_ixs1]
  return (x1, x2, x3)
end

## calculates F1_0, with marginal scale `scale`
function F1_0(x, x0, θ, ν;scale)
  if x ≤ -Inf return 0 end
  if x ≥ Inf return 0 end
  H(x) = quantile(Frechet(), cdf(scale, x))
  x0 = H(x0); x = H(x) # move to frechet

  V_(x0,x1,x2) = V([x0,x1,x2],θ,ν)
  dV0(x0,x1,x2) = ∂V∂0([x0,x1,x2],θ,ν)
  dV1(x0,x1,x2) = ∂V∂1([x0,x1,x2],θ,ν)
  d2V01(x0,x1,x2) = ∂2V∂0∂1([x0,x1,x2],θ,ν)
  # work on log scale and then exp up
  res = -V_(x0, x, Inf) + V_(x0, Inf, Inf)
  if res ≤ -Inf return 0 end
  res += log(-dV0(x0, x, Inf))
  res -= log(-dV0(x0, Inf, Inf))

  if isnan(res) return 0 end
  exp(res)
end


function f_laplace_1_0(x, x0, θ, ν)
  # dF_{1|0}/dy in laplace margins
  dV0(x0,x1,x2) = ∂V∂0([x0,x1,x2],θ,ν)
  dV1(x0,x1,x2) = ∂V∂1([x0,x1,x2],θ,ν)
  d2V01(x0,x1,x2) = ∂2V∂0∂1([x0,x1,x2],θ,ν)
  H(x) = quantile(Frechet(), cdf(Laplace(), x))
  function ∂H(x)
    if x ≤ 0
      # differentiate log(0.5*exp(x))^-1
      return -1/(x - log(2))^2
    else
      # differentiate (log(1-0.5*exp(-x)))^-1
      res = -0.5 / ((exp(x) - 0.5) * log(1-0.5*exp(-x))^2)
      if isnan(res)
        return 0
      else
        return res
      end
    end
  end
  Hx0 = H(x0)
  Hx = H(x)

  res = -V([Hx0, Hx, Inf],θ,ν)

  ## this line causes NaNs, not sure why
  res += log(dV0(Hx0, Hx, Inf)*dV1(Hx0, Hx, Inf) - d2V01(Hx0, Hx, Inf))

  res += V([Hx0, Inf, Inf], θ, ν) - log(-dV0(Hx0, Inf, Inf))
  res += log(-∂H(x))

  if isnan(res)
    0
  else
    exp(res)
  end
end


function f_logistic_1_0(x, x0, θ, ν)
  # dF_{1|0}/dy in laplace margins
  dV0(x0,x1,x2) = ∂V∂0([x0,x1,x2],θ,ν)
  dV1(x0,x1,x2) = ∂V∂1([x0,x1,x2],θ,ν)
  d2V01(x0,x1,x2) = ∂2V∂0∂1([x0,x1,x2],θ,ν)
  H(x) = quantile(Frechet(), cdf(Logistic(), x))
  ∂H(x) = gradient(H, x)[1]
  Hx0 = H(x0)
  Hx = H(x)

  res = -V([Hx0, Hx, Inf],θ,ν)

  ## this line causes NaNs, not sure why
  res += log(dV0(Hx0, Hx, Inf)*dV1(Hx0, Hx, Inf) - d2V01(Hx0, Hx, Inf))

  res += V([Hx0, Inf, Inf], θ, ν) - log(-dV0(Hx0, Inf, Inf))
  res += log(∂H(x))

  if isnan(res)
    0
  else
    exp(res)
  end
end
