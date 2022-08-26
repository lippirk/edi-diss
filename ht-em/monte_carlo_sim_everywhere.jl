using ProgressMeter

@everywhere begin
  using QuadGK
  using Distributions
  using StatsBase
  include("./../util/util.jl")
  include("./../asym-log/gen.jl")
  include("./normal.jl")

  debug(x) = println(x)

  function do_work(seed,M,N,θ,ν,q,u,scale)
    println("---")
    println("begin: seed=$seed")
    data = gen_Xs(M=M,N=N,θ=θ,ν=ν,q=q,seed=seed)
    df = inner_loop1(θ=θ,ν=ν,uinit=u,scale=scale,data=data,seed=seed)
    println("end: seed=$seed")
    println("---")
    return df
  end

  function X_Y_of_Xs(Xs; u)
    u_ixs1 = Xs.>u
    u_ixs1[:,end] .= 0
    u_ixs2 = mapslices(x->x>>1, u_ixs1, dims=2)

    X = Xs[u_ixs1][:]; Y = Xs[u_ixs2][:]
    return (X,Y)
  end

  function inner_loop1(;θ,ν,uinit,scale,data,seed,M,N)

    Xs = quantile.(scale, cdf.(Frechet(), data))

    # u = quantile(Laplace(), 0.975)
    u = uinit # approx the above
    X,Y = X_Y_of_Xs(Xs,u=u)

    inits = [.5,.9,.1,.1,.2,.1,-.1,1.1,0.9]
    ps = fit_HT_normal_K_EM(X,Y,K=2, inits=inits, seed=nothing)

    C = class_of_normal_ps(ps,X,Y;K=2)
    Z1 = (Y[C.==1] .- ps.α[1] .* X[C.==1]) ./ X[C.==1].^ps.β[1]
    Z2 = (Y[C.==2] .- ps.α[2] .* X[C.==2]) ./ X[C.==2].^ps.β[2]

    π̂ = length(Z1)/(length(Z1) + length(Z2))
    Fhat_z1 = ecdf(Z1); Fhat_z2 = ecdf(Z2)
    Fhat_resids(x,y) = π̂ * Fhat_z1((y - ps.α[1]*x)/x^ps.β[1]) +
                      (1-π̂) * Fhat_z2((y - ps.α[2]*x)/x^ps.β[2])


    Ftrue(x,y) = F1_0(y,x,θ,ν,scale=scale)

    x0s = [0.97,0.98,0.99,0.999,0.9999]|>x->map(x->quantile(scale,x),x)
    absdiffs=zeros(length(x0s)); abs_sq_diffs=zeros(length(x0s))
    # maxdiff=zeros(length(x0s))
    for i in eachindex(x0s)
      x0 = x0s[i]
      absdiffs[i] = quadgk(y->abs(Ftrue(x0,y) - Fhat_resids(x0,y)), -Inf, Inf,rtol=1e-4)[1]
      abs_sq_diffs[i] = quadgk(y->(Ftrue(x0,y) - Fhat_resids(x0,y))^2, -Inf, Inf,rtol=1e-4)[1]
    end

    df = DataFrame(scale=scale,
                   θ0=θ[1],θ1=θ[2],θ2=θ[3],ν01=ν[1],ν02=ν[2],ν012=ν[3],
                   θ01=θ[3],θ02=θ[4],θ12=θ[5],θ012=θ[6],
                   π=ps.π[1], α1=ps.α[1],α2=ps.α[2],β1=ps.β[1],β2=ps.β[2],
                   π̂=π̂,
                   μ1=ps.μ[1],μ2=ps.μ[2],σ1=ps.σ[1],σ2=ps.σ[2],
                   ll=ps.ll,
                   absdiffp97=absdiffs[1],absdiffp98=absdiffs[2],absdiffp99=absdiffs[3],
                   absdiffp999=absdiffs[4],absdiffp9999=absdiffs[5],
                   abs_sq_diffp97=abs_sq_diffs[1],abs_sq_diffp98=abs_sq_diffs[2],
                   abs_sq_diffp99=abs_sq_diffs[3],
                   abs_sq_diffp999=abs_sq_diffs[4],abs_sq_diffp9999=abs_sq_diffs[5],
                   seed=seed,
                   u=u
                   )
    df
  end
end


function mc_sim_study()
  θs=[[0.1,0.1,0.1,0.3,0.3,0.3],
      [0.3,0.3,0.3,0.3,0.3,0.1]]
  νs=[[0.5,0.5,0.5],[0.3,0.3,0.3], [0.7,0.7,0.7]]
  scales=[Logistic(), Laplace()]
  qs = [0.99]
  # us = [0.95,0.96,0.97,0.98]
  us = [0.95,0.98]
  # MNs = [(100, 30), (200, 30), (50, 30)]
  MNs = [(100, 30)]

  seeds = 0:99

  daddy_df = nothing

  i = 0
  @showprogress "mc_sim_study" for θ in θs, ν in νs, q in qs,
     (M,N) in MNs, u in us,
     scale in scales
    i += 1
    debug("outer_loop iter = $i")
    debug("====================")
    debug("θ = $θ")
    debug("ν = $ν")
    debug("q = $q")
    debug("(M,N) = ($M,$N)")
    debug("u = $u")
    debug("scale = $scale")
    debug("--------------------")
    dfs = pmap(seed->do_work(seed,M,N,θ,ν,q,u,scale), seeds)
    df = vcat(dfs...)
    daddy_df =  isnothing(daddy_df) ? df : vcat(daddy_df, df)
    debug("--------------------")
    debug("====================")
  end
  daddy_df
end



