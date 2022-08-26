function choice_of_T_graph(;seed=452)
  Random.seed!(seed)

  function go(μ)
    X_orig = rand(Normal(μ, 1), 500)
    X = sigmoid.(X_orig)
    ps = fit_bernstein_EM(X, K = 20)
    ps_fhat = ebde(ps)

    fhat_orig(x) = ps_fhat(sigmoid(x)) * sigmoid(x) * (1 - sigmoid(x))
    X_orig, fhat_orig
  end

  X_0, fhat_μ_0 = go(0)
  X_3, fhat_μ_3 = go(2.5)
  X_6, fhat_μ_6 = go(5)

  plt1 = Plots.plot(x ->fhat_μ_0(x), legend = false, xlim=(-3, 8),title=L"\mu=0")
  Plots.histogram!(plt1, X_0, normalize=:pdf, alpha=0.5)

  plt2 = Plots.plot(x ->fhat_μ_3(x), legend = false, xlim=(-3, 8),title=L"\mu=2.5")
  Plots.histogram!(plt2, X_3, normalize=:pdf, alpha=0.5)

  plt3 = Plots.plot(x ->fhat_μ_6(x), legend = false, xlim=(-3, 8),title=L"\mu=5")
  Plots.histogram!(plt3, X_6, normalize=:pdf, alpha=0.5)

  l = @layout [ a ; b ; c ]
  p = Plots.plot(plt1, plt2, plt3, layout=l);
  savefig(p, "figures/bernstein_normal_T_graph.pdf")
end

function compare_fits(;seed=35)
  θ=[0.3,0.3,0.3,0.3,0.3,0.1]
  ν=[0.5,0.5,0.5]

  Random.seed!(seed)
  u = 3
  X,Y = quick_test_data_2d(u=u)

  ht_ps = fit_HT_normal_K_EM(X, Y, K=2)
  ht_fhat = HT_normal_fhat(ht_ps)

  bern_ps = fit_bernstein_2_mixture_with_α_β_EM(X, Y, nouteriter=100, K=30,
      f01s=   [x->sigmoid(x,loc=-0.44,scale=2),x->sigmoid(x,loc=-0.08,scale=2)],
      f01invs=[p->logit(p,loc=-0.44,scale=2),  p->logit(p,loc=-0.08,scale=2)])
      # f01s=   [x->sigmoid(x,loc=-3.15,scale=2.8),x->sigmoid(x,loc=-1.9,scale=2.7)],
      # f01invs=[p->logit(p,loc=-3.15,scale=2.8),  p->logit(p,loc=-1.9,scale=2.7)])
  bern_fhat = bernstein2_fhat(bern_ps)

  println("BERN")
  println(bern_ps)
  println("NORM")
  println(ht_ps)

  plt = Plots.plot(y->ht_fhat(u, y), label=L"$\hat{f}_{HT}(y|X=%$u)$");
  Plots.plot!(plt,
              x->f_laplace_1_0(x, u, θ, ν),
              label=L"$f(y|X=%$u)$")

  # bern_fhat(u, 1.)
  Plots.plot!(plt, y->bern_fhat(u,y), label=L"$\hat{f}_{bern}(y|X=%$u)$");
end

function normal_then_bern_fit(;seed=35)
  θ=[0.3,0.3,0.3,0.3,0.3,0.1]
  ν=[0.5,0.5,0.5]

  Random.seed!(seed)
  u = 3
  X,Y = quick_test_data_2d(u=u, scale=Laplace())

  ht_ps = fit_HT_normal_K_EM(X, Y, K=2, seed=2)
  ht_fhat = HT_normal_fhat(ht_ps)
  Y_ = Y .- X
  C = class_of_normal_ps(ht_ps, X,Y,K=2)

  f01s = [x->sigmoid.(x, loc=mean(Y_[C.==1]), scale=std(Y_[C.==1])),
          x->sigmoid.(x, loc=mean(Y[C.==2]), scale=std(Y[C.==2]))]
          #
  # f01s = [x->sigmoid.(x, loc=ht_ps.μ[1], scale=ht_ps.σ[1]),
          # x->sigmoid.(x, loc=ht_ps.μ[2], scale=ht_ps.σ[2])]
  data = hcat(f01s[1].(Y_),
              f01s[2].(Y))
  bern_ps = fit_bernstein_mixture_EM(data, P=1, K=100)

  # ht_ps = fit_HT_normal_K_EM(X, Y, K=2)
  # ht_fhat = HT_normal_fhat(ht_ps)

  bern_fhat = bernstein_πw_fhat(bern_ps, α=1, β=0, f01s=f01s)

  println("BERN")
  println(bern_ps)
  println("NORM")
  println(ht_ps)

  plt = Plots.plot(y->ht_fhat(u, y), label=L"$\hat{f}_{HT}(y|X=%$u)$");
  Plots.plot!(plt,
              x->f_laplace_1_0(x, u, θ, ν),
              label=L"$f(y|X=%$u)$")

  # bern_fhat(u, 1.)
  Plots.plot!(plt, y->bern_fhat(u,y), label=L"$\hat{f}_{bern}(y|X=%$u)$");
  display(plt)
end

function demo_hists()
  u = 6
  X,Y = quick_test_data_2d(u=u, scale=Laplace())

  p1 = histogram(Y,normalize=:pdf,bins=20)
  p2 = histogram(Y .- X,normalize=:pdf,bins=20)
  p3 = histogram((Y .- X) ./ X,normalize=:pdf,bins=20)
  p4 = histogram(Y ./ X,normalize=:pdf,bins=20)

  layout = @layout [ p1 ; p2 ; p3 ; p4 ]
  Plots.plot(p1, p2, p3, p4, layout=layout)
end
