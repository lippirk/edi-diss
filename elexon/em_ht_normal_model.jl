
default(thickness_scaling=1, markersize=2, markerstrokewidth=-1,
          legend=false, titlefontsize=12, xguidefontsize=12, yguidefontsize=12)#plot default
function fit_ht_em_first_order()
  resids = get_resids()
  default(thickness_scaling=1, markersize=3, markerstrokewidth=-1,
          legend=false)
  u = quantile(Laplace(), 0.95)
  xs = resids
  u_ixs = xs .> u
  u_ixs[end] = 0

  x1 = xs[u_ixs]; x2 = xs[u_ixs>>1]

  inits = [0.5, 0.9, -0.9, 0.5, 0.5, 0., 0., 2., 2.] # K = 2

  ## no reg
  nseeds = 30
  lls = zeros(nseeds)
  lls[:] .= -Inf
  @Threads.threads for s in 10:9+nseeds
    try
      ps = fit_HT_normal_K_EM(x1, x2, K=2, seed=s,
                              maxiter=1000,γσ=0.0)
      lls[s-9] = ps.ll
    catch
      println("seed=$s failed")
    end
  end
  @assert 3 == argmax(lls)
  seed = 10+3
  ps = fit_HT_normal_K_EM(x1, x2, K=2, seed=seed,γσ=0.)
  df = DataFrame(π=ps.π[1], α1=ps.α[1], α2=ps.α[2], β1=ps.β[1], β2=ps.β[2],
                 μ1=ps.μ[1], μ2=ps.μ[2], σ1=ps.σ[1], σ2=ps.σ[2], ll=ps.ll)
  colnames = [map(x->"\$\\hat\\$(x)\$", ["pi", "alpha_1", "alpha_2", "beta_1",
                                        "beta_2", "mu_1", "mu_2", "sigma_1",
                                        "sigma_2"])..., "Log likelihood"]
  table_str = tex_table_string_of_df(df, headers = colnames)
  open("./tex/seed_18.tex", "w") do io
    write(io, table_str)
  end
  C = class_of_normal_ps(ps, x1, x2, K=2)

  _ht_pdf = ht_pdf(ps)
  colors=[:red, :green, :black]
  p = Plots.scatter(x1[C.==1],x2[C.==1], color=colors[1], xlab=L"y_{t}", ylab=L"y_{t+1}",xlim=[u,Inf])
  for k in 2:2
    Plots.scatter!(p,x1[C.==k], x2[C.==k], color=colors[k]);
  end
  savefig(p, "figures/seed_18.pdf")
  sum(C .== 1) / length(C)


  inits = [0.5, 0.9, -0.9, 0.1, 0.1, 0., 0., 2., 2.] # K = 2
  df = DataFrame(u=[],π=[],α1=[],α2=[],β1=[],β2=[],μ1=[],μ2=[],σ1=[],σ2=[],ll=[])
  γ = 0.05
  for u in 1.6:0.05:3.9
    ixs = xs .> u
    X_ = xs[ixs]; Y_ = xs[ixs>>1]
    ps_ = fit_HT_normal_K_EM(X_,Y_,K=2, inits=inits, maxiter=2000,γσ=γ)
    push!(df, [u, ps_.π[1], ps_.α..., ps_.β..., ps_.μ..., ps_.σ..., ps_.ll])
  end
  p1 = @df df plot(:u,[:α1,:α2,:β1,:β2,:π],
                   labels=[L"\alpha_1",L"\alpha_2",L"\beta_1",L"\beta_2",L"\pi"]|>permutedims,
                   legend=:outerright,xticks=(1.6:0.2:3.6))
  p2 = @df df plot(:u,[:μ1,:μ2,:σ1,:σ2],
                   labels=[L"\mu_1",L"\mu_2",L"\sigma_1",L"\sigma_2"]|>permutedims,
              legend=:outerright, xticks=(1.6:0.2:3.6))
  l = @layout [ a ; b ]
  p = Plots.plot(p1, p2, layout=l)
  savefig(p, "./figures/imb_param_vals_changing_u.pdf")


  K = 2
  γ = 0.05
  u = 2.8
  xs = get_resids()
  u_ixs = xs .> u
  u_ixs[end] = 0
  x1 = xs[u_ixs]; x2 = xs[u_ixs>>1]
  inits = [0.5, 0.9, -0.9, 0.5, 0.5, 0., 0., 2., 2.] # K = 2
  ps = fit_HT_normal_K_EM(x1, x2, K=2, inits=inits,γσ=γ)
  quickplot(ps,x1,x2,u=u)
  C = class_of_normal_ps(ps, x1, x2, K=K)
  df = DataFrame(π=ps.π[1], α1=ps.α[1], α2=ps.α[2], β1=ps.β[1], β2=ps.β[2],
                 μ1=ps.μ[1], μ2=ps.μ[2], σ1=ps.σ[1], σ2=ps.σ[2], ll=ps.ll)
  colnames = [map(x->"\$\\hat\\$(x)\$", ["pi", "alpha_1", "alpha_2", "beta_1",
                                        "beta_2", "mu_1", "mu_2", "sigma_1",
                                        "sigma_2"])..., "Log likelihood"]
  table_str = tex_table_string_of_df(df, headers = colnames)
  open("./tex/imb_reg05.tex", "w") do io
    write(io, table_str)
  end
  C = class_of_normal_ps(ps, x1, x2, K=K)
  _ht_pdf = ht_pdf(ps)
  p = Plots.contour(u:0.05:10.,-10:0.05:10.,
                    (x,y)->_ht_pdf(x,y),color=:jet,alpha=0.7,
                    xlab=L"y_{t}",ylab=L"y_{t+1}");
  display(p)
  colors=[:red, :green, :black]
  for k in 1:K
    Plots.scatter!(p,x1[C.==k], x2[C.==k], color=colors[k],xlim=[u,Inf]);
  end
  display(p)
  savefig(p, "./figures/imbalance_contour_with_reg05.pdf")
  display(p)
end

function jan22_pred()
  ys = get_jan22_resids()
  ys_true = get_jan22_true_prices()
  arima_forecasts = CSV.read("./data/arima-point-forecasts.csv", DataFrame)|>Matrix|>x->x[:,2]

  function mk_ys_to_laplace()
    ys_ecdf = get_ys_ecdf()
    return function(y)
      ys_ecdf(y)|>y->quantile(Laplace(), y)
    end
  end
  ys_to_laplace_scale = mk_ys_to_laplace()

  function mk_laplace_to_ys()
    ys_ecdf = get_ys_ecdf()
    unif_to_ys(u) = solve_ecdf_eq_u_in_R(ys_ecdf, u)
    return function(y)
      cdf(Laplace(), y)|>unif_to_ys
    end
  end
  laplace_to_ys_scale = mk_laplace_to_ys()

  jan_day_of_ix(ix) = ceil(ix/48)|>Int
  period_of_ix(ix) = ifelse(ix % 48 == 0, 48, ix % 48)


  u = 2.8
  ys[ys .> u]|>length
  p = plot(ys, xlim=[1,1488], xlab=L"t", ylab=L"y_t",color=:black)
  Plots.abline!(p, 0, u, color=:grey, linestyle=:dash)
  savefig("./figures/resids_jan_22.pdf")

  y1 = ys[ys .> u]; y2 = ys[(ys .> u) >> 1]

  Random.seed!(2934787)
  (Z1, Z2) = sample_Zs(ps, x1, x2, M=500)
  _fhat_z1 = ecdf(Z1); _fhat_z2 = ecdf(Z2)
  π̂ = length(Z1) / (length(Z1) + length(Z2))
  Fhat_full = ((x,y) -> π̂ * _fhat_z1((y - ps.α[1]*x)/x^ps.β[1]) +
                            (1-π̂) * _fhat_z2((y - ps.α[2]*x)/x^ps.β[2]))

  ## uncertainty in parameters
  Random.seed!(802938)
  block_size = 48
  bootstrap_iters = 1000
  N = length(xs)
  block_options = eachindex(xs)[xs .> u]|>x->filter(x -> x + block_size ≤ N,x)
  num_blocks_req = N / block_size |> Int

  colnames = ["π1","π2","α1","α2","β1","β2","μ1","μ2","σ1","σ2","ll"]
  colnames_and_types = zip(colnames, [Float64 for _ in eachindex(colnames)])
  results_df = DataFrame([i => Vector{T}(undef, bootstrap_iters) for (i, T) in  colnames_and_types],
                      copycols=false)
  _size = length(block_options)

  datasets = Array{Float64}(undef, bootstrap_iters, N)
  for iter in 1:bootstrap_iters
    # bootstrap the data
    boots = wsample(block_options,
                    ones(_size) ./ _size,
                    num_blocks_req)
    Xs_boot = vcat([xs[b:b+block_size-1] for b in boots]...)
    @assert length(Xs_boot) == N
    datasets[iter,:] = Xs_boot
  end

  # progr = Progress(bootstrap_iters,1,"bootstrapping...");
  # @Threads.threads for iter in 1:bootstrap_iters
    # Xs_boot = datasets[iter,:]
    # X_b = Xs_boot[1:end-1]; Y_b = Xs_boot[2:end]
    # X_exceed_b = X_b[X_b.>u]; Y_exceed_b = Y_b[X_b.>u]
    # ps_b = fit_HT_normal_K_EM(X_exceed_b,Y_exceed_b,K=2,inits=inits,maxiter=2000,γσ=γ)

    # results = [ps_b.π; ps_b.α; ps_b.β; ps_b.μ; ps_b.σ; ps_b.ll]
    # results_df[iter,:] = deepcopy(results)

    # ProgressMeter.next!(progr)
  # end
  # ProgressMeter.finish!(progr)
  # CSV.write("./bootstrap-data/k2.csv", results_df)
  results_df = CSV.read("./bootstrap-data/k2.csv", DataFrame)

  progr = Progress(bootstrap_iters,1,"Fhats_boot...");
  Random.seed!(574875847)
  Fhats_boot = Vector{Function}(undef, bootstrap_iters)
  @Threads.threads for i in 1:bootstrap_iters
    Xs_boot = datasets[i,:]
    X_b = Xs_boot[1:end-1]; Y_b = Xs_boot[2:end]
    X_exceed_b = X_b[X_b.>u]; Y_exceed_b = Y_b[X_b.>u]
    ps_b = results_df[i,:]|>collect
    ps_b =(π=ps_b[1:K], α=ps_b[K+1:2K], β=ps_b[2K+1:3K],
           μ=ps_b[3K+1:4K], σ=ps_b[4K+1:5K])

    (Z1, Z2) =  sample_Zs(ps_b, X_exceed_b,Y_exceed_b, M=100)
    _fhat_z1 = ecdf(Z1); _fhat_z2 = ecdf(Z2)
    π̂ = length(Z1) / (length(Z1) + length(Z2))
    Fhats_boot[i] = ((x,y) -> π̂ * _fhat_z1((y - ps.α[1]*x)/x^ps.β[1]) +
                              (1-π̂) * _fhat_z2((y - ps.α[2]*x)/x^ps.β[2]))
    ProgressMeter.next!(progr)
  end
  ProgressMeter.finish!(progr)

  alpha=0.6
  p1 = histogram(results_df.α1, label="α1",bins=20,normalize=:pdf, title=L"\alpha_1",xlim=[0.85,1],alpha=alpha,
                 ylim=[0,13])
  vline!(p1,[ps.α[1]],color="red",linestyle=:dash,linewidth=2)
  p2 = histogram(results_df.α2, label="α2",bins=20,normalize=:pdf, title=L"\alpha_2",alpha=alpha,
                 ylim=[0,Inf],xlim=[-1.0,-0.90])
  vline!(p2,[ps.α[2]],color="red",linestyle=:dash,linewidth=2)
  p3 = histogram(results_df.β1, label="β1",bins=20,normalize=:pdf, title=L"\beta_1",alpha=alpha,
                 )
  vline!(p3,[ps.β[1]],color="red",linestyle=:dash,linewidth=2)
  p4 = histogram(results_df.β2, label="β2",bins=20,normalize=:pdf, title=L"\beta_2",alpha=alpha)
  vline!(p4,[ps.β[2]],color="red",linestyle=:dash,linewidth=2)
  layout = @layout [ a b ; c d ]
  p = Plots.plot(p1, p2, p3, p4, layout=layout, rightmargin=5Plots.mm)
  savefig(p,"./figures/parameter_uncert_k2.pdf")

  Mpred = 2000
  ypreds = zeros(Mpred, length(y1))
  ytrues = zeros(length(y1))
  progr = Progress(bootstrap_iters,1,"ypreds...");
  @Threads.threads for _n in eachindex(y1)
    _y0 = y1[_n]
    ytrues[_n] = y2[_n]
    Finv = x->Fhat_full(_y0,x)
    us = rand(Mpred)
    ypreds[:,_n] = solve_ecdf_eq_u_in_R.(Finv, us)
    ProgressMeter.next!(progr)
  end
  ProgressMeter.finish!(progr)

  plts = Vector{Plots.Plot}(undef, length(y1))
  nbins = 30
  for i in eachindex(y1)

    ix = ((ys .> u)|>findall)[i]
    v = round(ys[ix],digits=2)
    plt = Plots.histogram(ypreds[:,i], normalize=:pdf,
                          title=L"y_{t+1} | y_{%$(jan_day_of_ix(ix)),%$(period_of_ix(ix))} = %$(v)",
                         nbins=nbins, alpha=0.7,xticks=(-6:2:6))
    vline!(plt, [ytrues[i]], color=:red,linewidth=2);
    vline!(plt, [arima_forecasts[i]|>ys_to_laplace_scale], color=:darkgreen, linewidth=2);
    plts[i] = plt
  end

  layout = @layout [ grid(4,4) ]
  pred_plot = Plots.plot(plts[1:16]..., layout=layout, size=(900, 900))
  savefig(pred_plot, "./figures/ytp1_giv_yt_jan_22_k2_full_pred_final.pdf")
  π̂ = ps.π[1]

  lik_jan22_k_2 = zeros(length(y1))
  for i in eachindex(y1)
    _ypreds = ypreds[:,i]
    empirical_dist = kde(_ypreds)
    lik_jan22_k_2[i] = pdf(empirical_dist, ytrues[i])
  end

  ypreds_ys_scale = laplace_to_ys_scale.(ypreds)


  log_price_preds = CSV.read("./residuals/jan-22.csv", DataFrame)
  log_price_preds = log_price_preds[:,2]
  _mean = 5.27880672777335 ## from the deseasonalization model
  _sd = 0.370842882674566 ## from the deseasonalization model

  plts = Vector{Plots.Plot}(undef, length(y1))
  nbins = 50
  for i in eachindex(y1)
    ix = findall(ys .> u)[i]
    pred = log_price_preds[ix]
    log_price_preds_i = pred .+ ypreds_ys_scale[:,i]
    prices_i = exp.(_sd * log_price_preds_i .+ _mean) .- 91
    v = round(ys[ix],digits=2)
    plt =histogram(prices_i, normalize=:pdf,
                   title = L"P_{t+1} | y_{%$(jan_day_of_ix(ix)),%$(period_of_ix(ix))} = %$(v)", xlim=[-91,2000],
                   xticks=[-91,500,1000], nbins=nbins, alpha=0.7)
    vline!(plt, [ys_true[ix+1]], color=:red, linestyle=:dash, linewidth=2)
    plts[i] = plt
  end
  layout = @layout [ grid(4,4) ]
  pred_plot = Plots.plot(plts[1:16]..., layout=layout, size=(1000, 1000))
  savefig(pred_plot, "./figures/price_preds_jan_22.pdf")
end

function fit_K_eq_3()
  K = 3
  γ = 0.05
  u = 2.8
  xs = get_resids()
  u_ixs = xs .> u
  u_ixs[end] = 0
  x1 = xs[u_ixs]; x2 = xs[u_ixs>>1]
  inits = [0.4,0.2, 0.9, 0., -0.9, 0.1, 0.1, 0.1, 1., 0.,-1.,2., 2., 2.] # K = 3
  ps = fit_HT_normal_K_EM(x1, x2, K=K, inits=inits,γσ=γ)
  ps_k_3 = deepcopy(ps)
  df = DataFrame(π1=ps.π[1], π2=ps.π[2], π3=ps.π[3],
                 α1=ps.α[1], α2=ps.α[2], α3=ps.α[3],
                 β1=ps.β[1], β2=ps.β[2], β3=ps.β[3],
                 )
  colnames = map(x->"\$\\hat\\$(x)\$", ["pi_1", "pi_2", "pi_3", "alpha_1", "alpha_2", "alpha_3",
                                        "beta_1", "beta_2", "beta_3"])
  table_str = tex_table_string_of_df(df, headers = colnames)
  open("./tex/k_3_fit.tex", "w") do io
    write(io, table_str)
  end
  colnames = [map(x->"\$\\hat\\$(x)\$", ["mu_1", "mu_2", "mu_3",
                                          "sigma_1", "sigma_2", "sigma_3"])..., "Log likelihood"]
  df = DataFrame(μ1=ps.μ[1], μ2=ps.μ[2], μ3=ps.μ[3],
                 σ1=ps.σ[1], σ2=ps.σ[2], σ3=ps.σ[3],
                 ll=ps.ll)
  table_str = tex_table_string_of_df(df, headers = colnames)
  open("./tex/k_3_fit.tex", "a") do io
    write(io, table_str)
  end

  Random.seed!(84579483)
  (Z1, Z2, Z3) = sample_Zs(ps_k_3, x1, x2, M=500)
  _fhat_z1 = ecdf(Z1); _fhat_z2 = ecdf(Z2); _fhat_z3 = ecdf(Z3)
  Clen = length(Z1) + length(Z2) + length(Z3)
  π̂1 = length(Z1) / Clen; π̂2 = length(Z2) / Clen; π̂3 = length(Z3) / Clen
  @assert π̂3 > 0 && π̂2 > 0 && π̂1 > 0
  Fhat_full_K3 = ((x,y) -> π̂1 * _fhat_z1((y - ps_k_3.α[1]*x)/x^ps_k_3.β[1]) +
                               π̂2 * _fhat_z2((y - ps_k_3.α[2]*x)/x^ps_k_3.β[2]) +
                               π̂3 * _fhat_z3((y - ps_k_3.α[3]*x)/x^ps_k_3.β[3]))




  ys = get_jan22_resids()
  ys_true = get_jan22_true_prices()

  function mk_ys_to_laplace()
    ys_ecdf = get_ys_ecdf()
    return function(y)
      ys_ecdf(y)|>y->quantile(Laplace(), y)
    end
  end
  ys_to_laplace_scale = mk_ys_to_laplace()

  function mk_laplace_to_ys()
    ys_ecdf = get_ys_ecdf()
    unif_to_ys(u) = solve_ecdf_eq_u_in_R(ys_ecdf, u)
    return function(y)
      cdf(Laplace(), y)|>unif_to_ys
    end
  end
  laplace_to_ys_scale = mk_laplace_to_ys()

  jan_day_of_ix(ix) = ceil(ix/48)|>Int
  period_of_ix(ix) = ifelse(ix % 48 == 0, 48, ix % 48)


  u = 2.8
  ys[ys .> u]|>length
  p = plot(ys, xlim=[1,1488], xlab=L"t", ylab=L"y_t")
  Plots.abline!(p, 0, u, color=:black, linestyle=:dash)

  y1 = ys[ys .> u]; y2 = ys[(ys .> u) >> 1]

  Random.seed!(1203989)
  M_ = 2000
  Fhats_3 = Vector{Function}(undef, M_)
  for m1 in 1:M_
    C = class_of_normal_ps(ps,x1,x2;K=3,random=true)
    z1s = (x2[C.==1] .- ps.α[1] .* x1[C.==1]) ./ x1[C.==1].^ps.β[1]
    z2s = (x2[C.==2] .- ps.α[2] .* x1[C.==2]) ./ x1[C.==2].^ps.β[2]
    z3s = (x2[C.==3] .- ps.α[3] .* x1[C.==3]) ./ x1[C.==3].^ps.β[3]
    _fhat_z1 = ecdf(z1s); _fhat_z2 = ecdf(z2s); _fhat_z3 = ecdf(z3s)
    π̂1 = length(z1s) / length(C); π̂2 = length(z2s) / length(C); π̂3 = length(z3s) / length(C)
    @assert π̂3 > 0 && π̂2 > 0 && π̂1 > 0
    Fhats_3[m1] = ((x,y) -> π̂1 * _fhat_z1((y - ps.α[1]*x)/x^ps.β[1]) +
                          π̂2 * _fhat_z2((y - ps.α[2]*x)/x^ps.β[2]) +
                          π̂3 * _fhat_z3((y - ps.α[3]*x)/x^ps.β[3]))
  end



  Random.seed!(39482793)
  Mpred = 2000
  ypreds_k_3 = zeros(Mpred, length(y1))
  ytrues = zeros(length(y1))
  progr = Progress(bootstrap_iters,1,"ypreds3...");
  @Threads.threads for _n in eachindex(y1)
    _y0 = y1[_n]
    ytrues[_n] = y2[_n]
    Finv = x->Fhat_full_K3(_y0,x)
    us = rand(Mpred)
    ypreds_k_3[:,_n] = solve_ecdf_eq_u_in_R.(Finv, us)
    ProgressMeter.next!(progr)
  end
  ProgressMeter.finish!(progr)


  plts = Vector{Plots.Plot}(undef, length(y1))
  nbins = 50
  for i in eachindex(y1)

    ix = ((ys .> u)|>findall)[i]
    v = round(ys[ix],digits=2)
    plt = Plots.histogram(ypreds_k_3[:,i], normalize=:pdf,
                          title=L"y_{t+1} | y_{%$(jan_day_of_ix(ix)),%$(period_of_ix(ix))} = %$(v)",
                          nbins=nbins, alpha=0.7,xticks=(-6:2:6)
                        )
    vline!(plt, [ytrues[i]], color=:red,linewidth=2);
    plts[i] = plt
  end

  layout = @layout [ grid(4,4) ]
  pred_plot = Plots.plot(plts[1:16]..., layout=layout, size=(900, 900))
  savefig(pred_plot, "./figures/ytp1_giv_yt_jan_22_Keq3.pdf")

  lik_jan22_k_3 = zeros(length(y1))
  for i in eachindex(y1)
    _ypreds_3 = ypreds_k_3[:,i]
    empirical_dist = kde(_ypreds_3)
    plot(empirical_dist)|>display
    lik_jan22_k_3[i] = pdf(empirical_dist, ytrues[i])
  end
  df_violin = DataFrame(lik=vcat(lik_jan22_k_2,lik_jan22_k_3), K=vcat([2 for _ in 1:length(y1)], [3 for _ in 1:length(y1)]))
  @df df_violin violin(:K,:lik, linewidth=0, xticks=([2,3], [L"K=2", L"K=3"]))
  @df df_violin boxplot!(:K,:lik, alpha=0.75, linewidth=2)
  savefig("./figures/2vs3_violin.pdf")
end
