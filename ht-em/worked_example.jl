using LaTeXStrings
using StatsPlots
using StatsBase
using KernelDensity



function worked_example()
  # default(thickness_scaling=1, markersize=3, markerstrokewidth=-1,
          # legend=false)
  default(thickness_scaling=1, markersize=2, markerstrokewidth=-1,
          legend=false, titlefontsize=12, xguidefontsize=12, yguidefontsize=12)#plot default
  θ=[0.1,0.1,0.1,0.3,0.3,0.3];ν=[0.5,0.5,0.5]
  # θ=[0.3,0.3,0.3,0.3,0.3,0.1];ν=[0.5,0.5,0.5]
  # θ=[0.3,0.3,0.3,0.3,0.3,0.1];ν=[0.5,0.5,0.5]
  # Xs = gen_Xs(M=100,N=40,θ=θ,ν=ν,q=0.95, seed=1293483)
  Xs = gen_Xs(M=100,N=30,θ=θ,ν=ν,q=0.99, seed=1293483)
  Xs = quantile.(Laplace(), cdf.(Frechet(), Xs))

  # u = quantile(Laplace(), 0.975)
  u = 3 # approx the above
  X,Y = X_Y_of_Xs(Xs,u=u)

  plt1 = scatter(Xs[:][1:end], Xs[:][2:end], legend=false, xlab=L"X_{t}",
                 ylab=L"X_{t+1}", aspectratio=1, color=:black)
  Plots.vline!(plt1, [u], linestyle=:dash, color="grey")
  plt2 = scatter(X,Y, xlab=L"X_{t}", ylab=L"X_{t+1}", xlim=[u,Inf], color=:black)
  # layout = @layout [  a b  ;  c  ]
  l = @layout [ a b ]
  plt = Plots.plot(plt1, plt2, layout=l)

  savefig(plt, "./figures/worked_example_data.pdf")


  plt3 = plot(Xs[1,:], label="j=1", xlab=L"t", ylab=L"X_t^j", legend=true);
  for j in 2:3
    plot!(plt3, Xs[j,:], label="j=$j")
  end
  hline!(plt3, [u], color="black", linestyle=:dash, label=L"u")


  savefig(plt3, "./figures/worked_example_series.pdf")

  df = DataFrame(seed=[],π=[],α1=[],α2=[],β1=[],β2=[],ll=[])
  n = 10
  for seed in 1:n
    ps = fit_HT_normal_K_EM(X,Y,K=2, seed=seed)
    push!(df, Any[seed,ps.π[1], ps.α..., ps.β..., ps.ll])
  end
  colnames = ["Seed", "\$\\hat\\pi\$","\$\\hat\\alpha_1\$", "\$\\hat\\alpha_2\$", "\$\\hat\\beta_1\$", "\$\\hat\\beta_2\$", "Log Likelihood"]
  table_str = tex_table_string_of_df(df, headers=colnames,sigfig=2)
  open("./tex/worked_example_table_1.tex", "w") do io
    write(io, table_str)
  end

  seed = 2
  ps = fit_HT_normal_K_EM(X,Y,K=2, seed=seed)

  # fit viz
  K = ps.α|>length
  C = class_of_normal_ps(ps, X, Y, K=K)
  _ht_pdf = ht_pdf(ps)
  xmax=13.; ymax=14.; ymin=-7.
  p = Plots.contour(u:0.05:xmax,ymin:0.05:ymax,(x,y)->_ht_pdf(x,y), alpha=0.5, color=:jet,
                    xlab=L"X_{t}", ylab=L"X_{t+1}", colorbar=false);
  colors=[:red, :green, :black]
  for k in 1:K
    Plots.scatter!(p,X[C.==k], Y[C.==k], color=colors[k], xlim=[u,xmax], ylim=[ymin,ymax])
  end
  display(p)
  savefig(p, "./figures/worked_example_contour_fit.pdf")

  fhat = ht_pdf(ps)
  ftrue(x0, x) = f_laplace_1_0(x, x0, θ, ν)
  x = -4:0.1:11

  u = 3;
  plt1 = Plots.plot(x, fhat.(u, x))
  Plots.plot!(plt1, x, ftrue.(u, x),title=L"x=%$u",yticks=[0,0.1,0.2])

  u = 5;
  plt2 = Plots.plot(x, fhat.(u, x))
  Plots.plot!(plt2, x, ftrue.(u, x),title=L"x=%$u",yticks=[0,0.1,0.2])

  u = 7;
  plt3 = Plots.plot(x, fhat.(u, x))
  Plots.plot!(plt3, x, ftrue.(u, x),title=L"x=%$u",yticks=[0,0.1,0.2])

  u = 9;
  plt4 = Plots.plot(x, fhat.(u, x))
  Plots.plot!(plt4, x, ftrue.(u, x),title=L"x=%$u",yticks=[0,0.1,0.2])


  layout = @layout [ a ; b ; c ; d ]
  plt = Plots.plot(plt1, plt2, plt3, plt4, layout=layout, legend=false);
  display(plt)
  savefig(plt, "./figures/worked_example_vs_true.pdf")


  df = DataFrame(u=[],α1=[],α2=[],β1=[],β2=[],μ1=[],μ2=[],σ1=[],σ2=[],ll=[])
  for u in 2.5:0.05:3.0
    X_, Y_ = X_Y_of_Xs(Xs,u=u)
    ps_ = fit_HT_normal_K_EM(X_,Y_,K=2, seed=seed)
    push!(df, [u, ps_.α..., ps_.β..., ps_.μ..., ps_.σ..., ps_.ll])
  end
  p1 = @df df plot(:u,[:α1,:α2,:β1,:β2],
                   labels=[L"\alpha_1",L"\alpha_2",L"\beta_1",L"\beta_2"]|>permutedims,
                   legend=:outerright)
  p2 = @df df plot(:u,[:μ1,:μ2,:σ1,:σ2],
                   labels=[L"\mu_1",L"\mu_2",L"\sigma_1",L"\sigma_2"]|>permutedims,
              legend=:outerright)
  l = @layout [ a ; b ]
  p = Plots.plot(p1, p2, layout=l)
  savefig(p, "./figures/worked_example_check_u.pdf")

  ## choose u = 2.9
  u = 2.9
  X, Y = X_Y_of_Xs(Xs,u=u)
  ps = fit_HT_normal_K_EM(X,Y,K=2, seed=seed)
  # C = class_of_normal_ps(ps,X,Y;K=2)

  # Z1 = (Y[C.==1] .- ps.α[1] .* X[C.==1]) ./ X[C.==1].^ps.β[1]
  # Z2 = (Y[C.==2] .- ps.α[2] .* X[C.==2]) ./ X[C.==2].^ps.β[2]
  (Z1, Z2) = sample_Zs(ps, X, Y, M=100)

  # π̂ = length(Z1)/(length(Z1) + length(Z2))
  π̂=ps.π[1]
  fhat_z1 = ecdf(Z1); fhat_z2 = ecdf(Z2)
  Fhat_resids(x,y) = π̂ * fhat_z1((y - ps.α[1]*x)/x^ps.β[1]) +
                    (1-π̂) * fhat_z2((y - ps.α[2]*x)/x^ps.β[2])

  us = rand(Uniform(), 10000)

  u = 3;
  resids = solve_ecdf_eq_u_in_R.(x->Fhat_resids(u,x), us)
  plt1 = Plots.plot(kde(resids), xlim=[-4,11], yticks=[0,0.2])
  Plots.plot!(plt1, x, ftrue.(u, x),title=L"x=%$u")
  Plots.histogram!(resids, normalize=:pdf, alpha=0.2)

  u = 5;
  resids = solve_ecdf_eq_u_in_R.(x->Fhat_resids(u,x), us)
  plt2 = Plots.plot(kde(resids), xlim=[-4,11], yticks=[0,0.2])
  Plots.plot!(plt2, x, ftrue.(u, x),title=L"x=%$u")
  Plots.histogram!(resids, normalize=:pdf, alpha=0.2)

  u = 7;
  resids = solve_ecdf_eq_u_in_R.(x->Fhat_resids(u,x), us)
  plt3 = Plots.plot(kde(resids), xlim=[-4,11], yticks=[0,0.2])
  Plots.plot!(plt3, x, ftrue.(u, x),title=L"x=%$u")
  Plots.histogram!(resids, normalize=:pdf, alpha=0.2)


  u = 9;
  resids = solve_ecdf_eq_u_in_R.(x->Fhat_resids(u,x), us)
  plt4 = Plots.plot(kde(resids), xlim=[-4,11], yticks=[0,0.2])
  Plots.plot!(plt4, x, ftrue.(u, x),title=L"x=%$u")
  Plots.histogram!(resids, normalize=:pdf, alpha=0.2)

  layout = @layout [ a ; b ; c ; d ]
  plt = Plots.plot(plt1, plt2, plt3, plt4, layout=layout, legend=false)
  savefig(plt, "./figures/worked_example_vs_true_ecdf.pdf")

end

function worked_example_logistic()
  # default(thickness_scaling=1, markersize=3, markerstrokewidth=-1,
          # legend=false)
  default(thickness_scaling=1, markersize=2, markerstrokewidth=-1,
          legend=false, titlefontsize=12, xguidefontsize=8, yguidefontsize=8)#pl
  θ=[0.1,0.1,0.1,0.3,0.3,0.3];ν=[0.5,0.5,0.5]
  Xs = gen_Xs(M=100,N=30,θ=θ,ν=ν,q=0.99, seed=1293483)
  Xs = quantile.(Logistic(), cdf.(Frechet(), Xs))


  u = 3.65 # quantile(Logistic(), 0.975)

  u_ixs1 = Xs.>u
  u_ixs1[:,end] .= 0 # don't include last point in each series
  u_ixs2 = mapslices(x->x>>1, u_ixs1, dims=2) # shift each row to the right

  X = Xs[u_ixs1][:]; Y = Xs[u_ixs2][:]

  plt1 = scatter(Xs[:][1:end], Xs[:][2:end], legend=false, xlab=L"X_{t}", ylab=L"X_{t+1}")
  Plots.vline!(plt1, [u], linestyle=:dash, color="black")
  plt2 = scatter(X,Y, xlab=L"X_{t-1}", ylab=L"X_{t}")

  layout = @layout [ a b ]
  plt = Plots.plot(plt1, plt2, layout=layout)

  savefig(plt, "./figures/worked_example_data_logistic.pdf")


  # df = DataFrame(seed=[],π=[],α1=[],α2=[],β1=[],β2=[],ll=[])
  # n = 50
  # for seed in 1:n
    # ps = fit_HT_normal_K_EM(X,Y,K=2, seed=seed)
    # push!(df, Any[seed,ps.π[1], ps.α..., ps.β..., ps.ll])
  # end

  # seed 2 also gives highest log lik for logistic data
  seed = 2
  ##
  ps = fit_HT_normal_K_EM(X,Y,K=2, seed=2)
  # quickplot(ps, X, Y, u=u,xmax=7)


  fhat = ht_pdf(ps)
  ftrue(x0, x) = f_logistic_1_0(x, x0, θ, ν)
  x = -5:0.1:12

  u = 3;
  plt1 = Plots.plot(x, fhat.(u, x))
  Plots.plot!(plt1, x, ftrue.(u, x),title=L"x=%$u",yticks=[0,0.1,0.2])

  u = 5;
  plt2 = Plots.plot(x, fhat.(u, x))
  Plots.plot!(plt2, x, ftrue.(u, x),title=L"x=%$u",yticks=[0,0.1,0.2])

  u = 7;
  plt3 = Plots.plot(x, fhat.(u, x))
  Plots.plot!(plt3, x, ftrue.(u, x),title=L"x=%$u",yticks=[0,0.1,0.2])

  u = 9;
  plt4 = Plots.plot(x, fhat.(u, x))
  Plots.plot!(plt4, x, ftrue.(u, x),title=L"x=%$u",yticks=[0,0.1,0.2])


  layout = @layout [ a ; b ; c ; d ]
  plt = Plots.plot(plt1, plt2, plt3, plt4, layout=layout, legend=false);
  savefig(plt, "./figures/worked_example_logistic_vs_true.pdf")

  df = DataFrame(u=[],α1=[],α2=[],β1=[],β2=[],μ1=[],μ2=[],σ1=[],σ2=[],ll=[])
  for u in 3.2:0.05:3.65
    X_, Y_ = X_Y_of_Xs(Xs,u=u)
    ps_ = fit_HT_normal_K_EM(X_,Y_,K=2, seed=seed)
    push!(df, [u, ps_.α..., ps_.β..., ps_.μ..., ps_.σ..., ps_.ll])
  end
  p1 = @df df plot(:u,[:α1,:α2,:β1,:β2],
                   labels=[L"\alpha_1",L"\alpha_2",L"\beta_1",L"\beta_2"]|>permutedims,
                   legend=:outerright)
  p2 = @df df plot(:u,[:μ1,:μ2,:σ1,:σ2],
                   labels=[L"\mu_1",L"\mu_2",L"\sigma_1",L"\sigma_2"]|>permutedims,
              legend=:outerright)
  l = @layout [ a ; b ]
  p = Plots.plot(p1, p2, layout=l)



  u = 3.45
  X, Y = X_Y_of_Xs(Xs,u=u)
  ps = fit_HT_normal_K_EM(X,Y,K=2, seed=2)
  # C = class_of_normal_ps(ps,X,Y;K=2)
  # Z1 = (Y[C.==1] .- ps.α[1] .* X[C.==1]) ./ X[C.==1].^ps.β[1]
  # Z2 = (Y[C.==2] .- ps.α[2] .* X[C.==2]) ./ X[C.==2].^ps.β[2]
  (Z1, Z2) = sample_Zs(ps, X, Y, M=100)

  π̂ = length(Z1)/(length(Z1) + length(Z2))
  Plots.histogram(Z1, bins=30)
  Plots.histogram(Z2, bins=30)

  fhat_z1 = ecdf(Z1); fhat_z2 = ecdf(Z2)
  Fhat_resids(x,y) = π̂ * fhat_z1((y - ps.α[1]*x)/x^ps.β[1]) +
                    (1-π̂) * fhat_z2((y - ps.α[2]*x)/x^ps.β[2])

  us = rand(Uniform(), 10000)

  u = 3;
  resids = solve_ecdf_eq_u_in_R.(x->Fhat_resids(u,x), us)
  plt1 = Plots.plot(kde(resids), xlim=[-4,11], yticks=[0,0.2])
  Plots.plot!(plt1, x, ftrue.(u, x),title=L"x=%$u")
  Plots.histogram!(resids, normalize=:pdf, alpha=0.2)

  u = 5;
  resids = solve_ecdf_eq_u_in_R.(x->Fhat_resids(u,x), us)
  plt2 = Plots.plot(kde(resids), xlim=[-4,11], yticks=[0,0.2])
  Plots.plot!(plt2, x, ftrue.(u, x),title=L"x=%$u")
  Plots.histogram!(resids, normalize=:pdf, alpha=0.2)

  u = 7;
  resids = solve_ecdf_eq_u_in_R.(x->Fhat_resids(u,x), us)
  plt3 = Plots.plot(kde(resids), xlim=[-4,11], yticks=[0,0.2])
  Plots.plot!(plt3, x, ftrue.(u, x),title=L"x=%$u")
  Plots.histogram!(resids, normalize=:pdf, alpha=0.2)


  u = 9;
  resids = solve_ecdf_eq_u_in_R.(x->Fhat_resids(u,x), us)
  plt4 = Plots.plot(kde(resids), xlim=[-4,11], yticks=[0,0.2])
  Plots.plot!(plt4, x, ftrue.(u, x),title=L"x=%$u")
  Plots.histogram!(resids, normalize=:pdf, alpha=0.2)

  layout = @layout [ a ; b ; c ; d ]
  plt = Plots.plot(plt1, plt2, plt3, plt4, layout=layout, legend=false)
  savefig(plt, "./figures/worked_example_logistic_vs_true_resids.pdf")
end

function perturb_inits(inits)
  function _perturb(x; σ=0.01, domain=(-Inf,Inf))
    res = x + rand(Normal(0, σ))
    while !(domain[1] ≤ res && res ≤ domain[2])
      res = x + rand(Normal(0, σ))
    end
    res
  end

  return [_perturb.(inits[1:5], domain=(0,1));
          _perturb.(inits[6:7], σ=0.1, domain=(-Inf,Inf));
          _perturb.(inits[8:9], σ=0.1, domain=(0,Inf))
         ]
end

function test_bootstrap()
  θ=[0.1,0.1,0.1,0.3,0.3,0.3];ν=[0.5,0.5,0.5]
  # θ=[0.3,0.3,0.3,0.3,0.3,0.1];ν=[0.5,0.5,0.5]
  # θ=[0.3,0.3,0.3,0.3,0.3,0.1];ν=[0.5,0.5,0.5]
  # Xs = gen_Xs(M=100,N=40,θ=θ,ν=ν,q=0.95, seed=1293483)
  Xs = gen_Xs(M=100,N=30,θ=θ,ν=ν,q=0.99, seed=1293483)
  Xs = quantile.(Laplace(), cdf.(Frechet(), Xs))

  # u = quantile(Laplace(), 0.975)
  u = 2.8 # approx the above
  X,Y = X_Y_of_Xs(Xs,u=u)
  ## fit on whole dataset
  ps = (π = [0.5110933630061099, 0.48890663699389014], α = [0.9372120805549247,
                                                            8.531577053784681e-19],
        β = [1.7855784472813222e-15, 0.17281552598253289], μ =
        [0.07804325756788069, 0.2933619002273819], σ = [0.8752042676819046,
                                                        1.3349158039627964], ll
        = -1041.3519313206205)
  C = class_of_normal_ps(ps,X,Y;K=2)
  Z1 = (Y[C.==1] .- ps.α[1] .* X[C.==1]) ./ X[C.==1].^ps.β[1]
  Z2 = (Y[C.==2] .- ps.α[2] .* X[C.==2]) ./ X[C.==2].^ps.β[2]

  π̂ = length(Z1)/(length(Z1) + length(Z2))

  fhat_z1 = ecdf(Z1); fhat_z2 = ecdf(Z2)
  Fhat_resids(x,y) = π̂ * fhat_z1((y - ps.α[1]*x)/x^ps.β[1]) +
                    (1-π̂) * fhat_z2((y - ps.α[2]*x)/x^ps.β[2])

  # ps = fit_HT_normal_K_EM(X,Y,K=2, seed=nothing, inits=[0.5,0.9,0.1,0.1,0.1,0.1])


  Random.seed!(23948036)
  subsampling_iters = 50
  # m = (1-exp(-1))*length(X)|>ceil|>Int
  m = 0.95*length(X)|>ceil|>Int
  inits = [ps.π[1:end-1]; ps.α; ps.β; ps.μ; ps.σ]
  results_df = DataFrame(π1=[], π2=[], α1=[], α2=[], β1=[], β2=[], μ1=[], μ2=[], σ1=[], σ2=[], ll=[])
  Fhat_residss = Vector{Function}(undef, subsampling_iters)
  @showprogress for iter in 1:subsampling_iters
    ## subsample
    sample_ixs = sample(eachindex(X),m)
    X_ = X[sample_ixs]; Y_ = Y[sample_ixs]

    ## fit model
    ps_ = fit_HT_normal_K_EM(X_,Y_,K=2, seed=nothing,inits=perturb_inits(inits))
    df = DataFrame(π1=[ps_.π[1]], π2=[ps_.π[2]], α1=[ps_.α[1]], α2=[ps_.α[2]],
                   β1=[ps_.β[1]], β2=[ps_.β[2]], μ1=[ps_.μ[1]], μ2=[ps_.μ[2]],
                   σ1=[ps_.σ[1]], σ2=[ps_.σ[2]], ll=[ps_.ll])
    results_df = vcat(results_df, df)

    C = class_of_normal_ps(ps_,X_,Y_;K=2)
    Z1 = (Y_[C.==1] .- ps_.α[1] .* X_[C.==1]) ./ X_[C.==1].^ps_.β[1]
    Z2 = (Y_[C.==2] .- ps_.α[2] .* X_[C.==2]) ./ X_[C.==2].^ps_.β[2]

    # π̂ = length(Z1)/(length(Z1) + length(Z2))
    fhat_z1 = ecdf(Z1); fhat_z2 = ecdf(Z2)
    Fhat_residss[iter] = ((x,y) -> ps_.π[1] * fhat_z1((y - ps_.α[1]*x)/x^ps_.β[1]) +
                                   ps_.π[2] * fhat_z2((y - ps_.α[2]*x)/x^ps_.β[2]))
  end
  df = results_df
  map(f->f(3.,0.), Fhat_residss)|>plot

  default(legend=false)
  alpha=0.6
  p1 = histogram(df.α1, label="α1",bins=20,normalize=:pdf, title=L"\alpha_1",xlim=[0,1],alpha=alpha)
  vline!(p1,[ps.α[1]],color="red",linestyle=:dash)
  # vline!(p1,[df.α1|>mean],color="black")
  p2 = histogram(df.α2, label="α2",bins=20,normalize=:pdf, title=L"\alpha_2",xlim=[0,1],alpha=alpha)
  vline!(p2,[ps.α[2]],color="red",linestyle=:dash)
  # vline!(p2,[df.α2|>mean],color="black")
  p3 = histogram(df.β1, label="β1",bins=20,normalize=:pdf, title=L"\beta_1",xlim=[0,1],alpha=alpha)
  vline!(p3,[ps.β[1]],color="red",linestyle=:dash)
  # vline!(p3,[df.β1|>mean],color="black")
  p4 = histogram(df.β2, label="β2",bins=20,normalize=:pdf, title=L"\beta_2",xlim=[0,1],alpha=alpha)
  vline!(p4,[ps.β[2]],color="red",linestyle=:dash)
  # vline!(p4,[df.β2|>mean],color="black")
  layout = @layout [ a b ; c d ]
  p = Plots.plot(p1, p2, p3, p4, layout=layout)
  savefig(p,"./figures/ci_for_cevvm_params.pdf")


  ##
  x = -8:0.1:8
  x0 = 5.
  y_Fhat_bootstrap = map(f->f.(x0,x), Fhat_residss)|>x->hcat(x...)
  y_Fhat_bootstrap_q95 = mapslices(x->quantile(x,0.975), y_Fhat_bootstrap,dims=2)
  y_Fhat_bootstrap_q05 = mapslices(x->quantile(x,0.025), y_Fhat_bootstrap,dims=2)
  y_Fhat_bootstrap_mean = y_Fhat_bootstrap|>x->mean(x,dims=2)
  y_Fhat_full = Fhat_resids.(x0,x)
  ribbon = (y_Fhat_bootstrap_mean.-y_Fhat_bootstrap_q05,
            y_Fhat_bootstrap_q95 .- y_Fhat_bootstrap_mean)
  p = plot(x,y_Fhat_bootstrap_mean, ribbon=ribbon)
  plot!(x,y_Fhat_full, color="red")
  Ftrues = map(x->F1_0(x,x0,θ,ν,scale=Laplace()), x)
  plot!(x,Ftrues,color="black")
  savefig("./figures/ci_for_F.pdf")
  # plot(-10:0.1:10,



end

