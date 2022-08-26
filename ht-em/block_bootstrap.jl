using Revise
using Distributions
using StatsBase
using LaTeXStrings
includet("../util/util.jl")
includet("../asym-log/gen.jl")
includet("./normal.jl")
includet("./util.jl")

default(thickness_scaling=1, markersize=2, markerstrokewidth=-1,
          legend=false, titlefontsize=12, xguidefontsize=16, yguidefontsize=16)#plot default

default(thickness_scaling=1, markersize=2, markerstrokewidth=-1,
          legend=false, titlefontsize=12, xguidefontsize=12, yguidefontsize=12)#plot default

function block_bootstrap_main()
  # default(thickness_scaling=1, markersize=3, markerstrokewidth=-1,
          # legend=false)
  γσ = 0.05
  K = 2
  N = 15000
  θ=[0.1,0.1,0.1,0.3,0.3,0.3];ν=[0.5,0.5,0.5]
  Xs = gen_Xs(M=1,N=N,θ=θ,ν=ν,q=0, seed=1293483)|>x->vcat(x...)
  Xs = quantile.(Laplace(), cdf.(Frechet(), Xs))
  u = quantile(Xs,0.95)
  X = Xs[1:end-1]; Y = Xs[2:end]

  plt1 = plot(Xs, xticks=[5_000,10_000,15_000], xlab=L"t", ylab=L"$X_{t}$",color=:black)
  Plots.abline!(plt1, 0, u, linestyle=:dash, color=:grey)
  plt2 = scatter(X, Y, xlim=[u,maximum(Xs) + 1], ylim=[-7,7],xlab=L"X_t", ylab=L"$X_{t+1}$",color=:black)
  layout = @layout [ a  b ]
  plt = Plots.plot(plt1,plt2, layout=layout)
  savefig(plt, "./figures/block_bootstrap_Xs.pdf")

  X_exceed = X[X.>u]; Y_exceed = Y[X.>u]
  inits = [0.5,0.9,0.1,0.1,0.1,0.0,0.0,2.0,2.0]
  ps = fit_HT_normal_K_EM(X_exceed,Y_exceed,K=2,inits=inits,maxiter=2000,γσ=γσ)
  quickplot(ps, X_exceed,Y_exceed,u=u, xmax=maximum(X_exceed)+2)

  ps_df =  DataFrame(π=[ps.π[1]],α1=[ps.α[1]],α2=[ps.α[2]],
                     β1=[ps.β[1]],β2=[ps.β[2]],
                     μ1=[ps.μ[1]],μ2=[ps.μ[2]],σ1=[ps.σ[1]],σ2=[ps.σ[2]],
                     ll=[ps.ll])
  table_str = tex_table_string_of_df(ps_df, headers=
                                     [map(x->"\$\\hat{\\$x}\$",
                                         ["pi", "alpha_1", "alpha_2",
                                          "beta_1", "beta_2",
                                          "mu_1", "mu_2", "sigma_1", "sigma_2"]);
                                      "Log likelihood"])
  open("./tex/15000_eg_fit.tex", "w") do io
    write(io, table_str);
  end

  Random.seed!(4485739)
  (Z1, Z2) = sample_Zs(ps, X_exceed, Y_exceed, M=500)
  _fhat_z1 = ecdf(Z1); _fhat_z2 = ecdf(Z2)
  π̂ = ps.π[1]
  Fhat_full_model = ((x,y) -> π̂ * _fhat_z1((y - ps.α[1]*x)/x^ps.β[1]) +
                              (1-π̂) * _fhat_z2((y - ps.α[2]*x)/x^ps.β[2]))

  ## uncertainty in the residuals
  # Random.seed!(102938)
  # M_ = 2000
 # Fhats_resids = Vector{Function}(undef, M_)
  # for m1 in 1:M_
    # C = class_of_normal_ps(ps,X_exceed,Y_exceed;K=2,random=true)
    # z1s = (Y_exceed[C.==1] .- ps.α[1] .* X_exceed[C.==1]) ./ X_exceed[C.==1].^ps.β[1]
    # z2s = (Y_exceed[C.==2] .- ps.α[2] .* X_exceed[C.==2]) ./ X_exceed[C.==2].^ps.β[2]
    # _fhat_z1 = ecdf(z1s); _fhat_z2 = ecdf(z2s)
    # π̂ = length(z1s) / length(C)
    # # π̂ = ps.π[1]
    # Fhats_resids[m1] = ((x,y) -> π̂ * _fhat_z1((y - ps.α[1]*x)/x^ps.β[1]) +
                            # (1-π̂) * _fhat_z2((y - ps.α[2]*x)/x^ps.β[2]))
  # end

  # function get_yFhat_full(x0, x)
    # y_Fhat_bootstrap = map(f->f.(x0,x), Fhats_resids)|>x->hcat(x...)
    # y_Fhat_bootstrap_median = mapslices(x->quantile(x,0.5), y_Fhat_bootstrap,dims=2)[:]
    # y_Fhat_full = y_Fhat_bootstrap_median
    # y_Fhat_full
  # end

  # x = -5:0.1:6.5
  # q0 = 0.99
  # x0 = quantile(Laplace(), q0)
  # y_Fhat_bootstrap = map(f->f.(x0,x), Fhats_resids)|>x->hcat(x...)
  # y_Fhat_bootstrap_q995 = mapslices(x->quantile(x,0.995), y_Fhat_bootstrap,dims=2)[:]
  # y_Fhat_bootstrap_q005 = mapslices(x->quantile(x,0.005), y_Fhat_bootstrap,dims=2)[:]
  # # y_Fhat_bootstrap_median = y_Fhat_bootstrap|>x->mean(x,dims=2)
  # y_Fhat_bootstrap_median = mapslices(x->quantile(x,0.5), y_Fhat_bootstrap,dims=2)[:]
  # y_Fhat_full = y_Fhat_bootstrap_median
  # ribbon = (y_Fhat_bootstrap_median.-y_Fhat_bootstrap_q005,
            # y_Fhat_bootstrap_q995 .- y_Fhat_bootstrap_median)
  # plt1 = plot(x,y_Fhat_bootstrap_median, ribbon=ribbon,xlab=L"x", title=L"\hat{F}_{1|0}(x;F_{0}^{-1}(%$(q0)))")
  # plot!(plt1,x,y_Fhat_full, color="red")
  # Ftrues = map(x->F1_0(x,x0,θ,ν,scale=Laplace()), x)
  # plot!(plt1, x,Ftrues,color="black")
  # savefig(plt1, "./figures/boot-sampling-residuals-only.pdf")

  ## uncertainty in parameters and in residuals
  Random.seed!(839)
  block_size = 50
  bootstrap_iters = 1000
  block_options = eachindex(Xs)[Xs .> u]|>x->filter(x -> x + block_size ≤ N,x)
  num_blocks_req = N / block_size |> Int

  colnames = ["π1","π2","α1","α2","β1","β2","μ1","μ2","σ1","σ2","ll"]
  colnames_and_types = zip(colnames, [Float64 for _ in eachindex(colnames)])
  results_df = DataFrame([i => Vector{T}(undef, bootstrap_iters) for (i, T) in  colnames_and_types],
                      copycols=false)
  _size = length(block_options)

  datasets = Array{Float64}(undef, bootstrap_iters, 15000)
  for iter in 1:bootstrap_iters
    # bootstrap the data
    boots = wsample(block_options,
                    ones(_size) ./ _size,
                    num_blocks_req)
    Xs_boot = vcat([Xs[b:b+block_size-1] for b in boots]...)
    @assert length(Xs_boot) == N
    datasets[iter,:] = Xs_boot
  end

  progr = Progress(bootstrap_iters,1,"bootstrapping...");
  @Threads.threads for iter in 1:bootstrap_iters
    # iter = 1
    # iter += 1
    Xs_boot = datasets[iter,:]
    X_b = Xs_boot[1:end-1]; Y_b = Xs_boot[2:end]
    X_exceed_b = X_b[X_b.>u]; Y_exceed_b = Y_b[X_b.>u]
    # inits = [0.5,0.6,0.4,0.1,0.1,0.1,-0.1,5.0,5.0]
    inits = [0.5,0.95,0.10,0.1,0.1,-1.,1.,2.0,2.0]
    ps_b = fit_HT_normal_K_EM(X_exceed_b,Y_exceed_b,K=2,inits=inits,maxiter=2000,γσ=γσ)
    # quickplot(ps_b, X_exceed_b,Y_exceed_b,u=u, xmax=maximum(X_exceed)+3)

    results = [ps_b.π; ps_b.α; ps_b.β; ps_b.μ; ps_b.σ; ps_b.ll]
    results_df[iter,:] = deepcopy(results)

    ProgressMeter.next!(progr)
  end
  # ProgressMeter.finish!(progr)
  # CSV.write("./bootstrap-data/final-2-1000.csv", results_df)
  results_df = CSV.read("./bootstrap-data/final-2-1000.csv", DataFrame)

  progr = Progress(bootstrap_iters,1,"Fhats_boot...");
  Random.seed!(102938578493)
  Fhats_boot = Vector{Function}(undef, bootstrap_iters)
  @Threads.threads for i in 1:bootstrap_iters
    Xs_boot = datasets[i,:]
    X_b = Xs_boot[1:end-1]; Y_b = Xs_boot[2:end]
    X_exceed_b = X_b[X_b.>u]; Y_exceed_b = Y_b[X_b.>u]
    ps_b = results_df[i,:]|>collect
    ps_b =(π=ps_b[1:K], α=ps_b[K+1:2K], β=ps_b[2K+1:3K],
           μ=ps_b[3K+1:4K], σ=ps_b[4K+1:5K])

    (Z1, Z2) =  sample_Zs(ps_b, X_exceed_b,Y_exceed_b, M=500)
    _fhat_z1 = ecdf(Z1); _fhat_z2 = ecdf(Z2)
    π̂ = ps_b.π[1]
    Fhats_boot[i] = ((x,y) -> π̂ * _fhat_z1((y - ps.α[1]*x)/x^ps.β[1]) +
                              (1-π̂) * _fhat_z2((y - ps.α[2]*x)/x^ps.β[2]))
    ProgressMeter.next!(progr)
  end
  ProgressMeter.finish!(progr)


  # Fhats_boot = Matrix{Function}(undef, M_, bootstrap_iters)
  # N1 = 100; N2 = 150
  # Fhats_boot = Matrix{Function}(undef, N1, N2)
  # @Threads.threads for i in 1:N2
    # Xs_boot = datasets[i,:]
    # X_b = Xs_boot[1:end-1]; Y_b = Xs_boot[2:end]
    # X_exceed_b = X_b[X_b.>u]; Y_exceed_b = Y_b[X_b.>u]
    # ps_b = results_df[i,:]|>collect
    # ps_b =(π=ps_b[1:K], α=ps_b[K+1:2K], β=ps_b[2K+1:3K],
           # μ=ps_b[3K+1:4K], σ=ps_b[4K+1:5K])

    # for m1 in 1:N1
      # C = class_of_normal_ps(ps_b,X_exceed_b,Y_exceed_b;K=2,random=true)
      # z1s = (Y_exceed_b[C.==1] .- ps_b.α[1] .* X_exceed_b[C.==1]) ./ X_exceed_b[C.==1].^ps_b.β[1]
      # z2s = (Y_exceed_b[C.==2] .- ps_b.α[2] .* X_exceed_b[C.==2]) ./ X_exceed_b[C.==2].^ps_b.β[2]
      # @assert length(z1s) + length(z2s) == length(Y_exceed_b) == length(X_exceed_b)
      # _fhat_z1 = ecdf(z1s); _fhat_z2 = ecdf(z2s)
      # ## TODO try both ps.π and #z1 / (#z1 + #z2)
      # # π̂ = length(z1s) / length(C)
      # π̂ = ps.π[1]
      # Fhats_boot[m1,i] = ((x,y) -> π̂ * _fhat_z1((y - ps.α[1]*x)/x^ps.β[1]) +
                                   # (1-π̂) * _fhat_z2((y - ps.α[2]*x)/x^ps.β[2]))
    # end
    # ProgressMeter.next!(progr)
  # end
  # ProgressMeter.finish!(progr)



  default(legend=false)
  alpha=0.6
  p1 = histogram(results_df.α1, label="α1",bins=20,normalize=:pdf, title=L"\alpha_1",xlim=[0.70,1],alpha=alpha,
                 ylim=[0,13])
  vline!(p1,[ps.α[1]],color="red",linestyle=:dash,linewidth=2)
  p2 = histogram(results_df.α2, label="α2",bins=20,normalize=:pdf, title=L"\alpha_2",alpha=alpha,
                 ylim=[0,Inf],xlim=[-0.1,0.3])
  vline!(p2,[ps.α[2]],color="red",linestyle=:dash,linewidth=2)
  p3 = histogram(results_df.β1, label="β1",bins=20,normalize=:pdf, title=L"\beta_1",alpha=alpha,
                 xticks=3,
                 xlim=[0,1e-11], ylim=[0,Inf])
  vline!(p3,[ps.β[1]],color="red",linestyle=:dash,linewidth=2)
  p4 = histogram(results_df.β2, label="β2",bins=20,normalize=:pdf, title=L"\beta_2",alpha=alpha,
                 xticks=3, xlim=[0,6e-11],ylim=[0,Inf])
  vline!(p4,[ps.β[2]],color="red",linestyle=:dash,linewidth=2)
  layout = @layout [ a b ; c d ]
  p = Plots.plot(p1, p2, p3, p4, layout=layout, rightmargin=6Plots.mm)
  savefig(p,"./figures/block_bootstrap_params_final.pdf")

  x = -4:0.1:6.5
  q0 = 0.975
  x0 = quantile(Laplace(), q0)
  y_Fhat_bootstrap = Matrix{Float64}(undef, bootstrap_iters, length(x))
  for i in eachindex(x)
    _x = x[i]
    y_Fhat_bootstrap[:,i] = map(f->f(x0,_x), Fhats_boot)|>x->vcat(x...)
  end
  y_Fhat_bootstrap_q995 = mapslices(x->quantile(x,0.995), y_Fhat_bootstrap,dims=1)[:]
  y_Fhat_bootstrap_q005 = mapslices(x->quantile(x,0.005), y_Fhat_bootstrap,dims=1)[:]
  y_Fhat_bootstrap_median = mapslices(x->quantile(x,0.5), y_Fhat_bootstrap,dims=1)[:]
  # y_Fhat_full = y_Fhat_bootstrap_median
  # y_Fhat_full = get_yFhat_full(x0, x)
  y_Fhat_full = Fhat_full_model.(x0,x)
  ribbon = (y_Fhat_bootstrap_median.-y_Fhat_bootstrap_q005,
            y_Fhat_bootstrap_q995 .- y_Fhat_bootstrap_median)
  plt1 = plot(x,y_Fhat_bootstrap_median, ribbon=ribbon,xlab=L"z",
              linewidth=0,
              title=L"\hat{F}_{1|0}(z;F_{0}^{-1}(%$(q0)))",linecolor=false)
  plot!(plt1,x,y_Fhat_full, color="red")
  Ftrues = map(x->F1_0(x,x0,θ,ν,scale=Laplace()), x)
  plot!(plt1, x,Ftrues,color="black")

  q0 = 0.99
  x0 = quantile(Laplace(), q0)
  y_Fhat_bootstrap = Matrix{Float64}(undef, bootstrap_iters, length(x))
  for i in eachindex(x)
    _x = x[i]
    y_Fhat_bootstrap[:,i] = map(f->f(x0,_x), Fhats_boot)|>x->vcat(x...)
  end
  y_Fhat_bootstrap_q995 = mapslices(x->quantile(x,0.995), y_Fhat_bootstrap,dims=1)[:]
  y_Fhat_bootstrap_q005 = mapslices(x->quantile(x,0.005), y_Fhat_bootstrap,dims=1)[:]
  y_Fhat_bootstrap_median = mapslices(x->quantile(x,0.5), y_Fhat_bootstrap,dims=1)[:]
  # y_Fhat_full = get_yFhat_full(x0, x)
  y_Fhat_full = Fhat_full_model.(x0, x)
  ribbon = (y_Fhat_bootstrap_median.-y_Fhat_bootstrap_q005,
            y_Fhat_bootstrap_q995 .- y_Fhat_bootstrap_median)
  plt2 = plot(x,y_Fhat_bootstrap_median, ribbon=ribbon,xlab=L"z", title=L"\hat{F}_{1|0}(z;F_{0}^{-1}(%$(q0)))",
             linewidth=0,linecolor=false)
  plot!(plt2,x,y_Fhat_full, color="red")
  Ftrues = map(x->F1_0(x,x0,θ,ν,scale=Laplace()), x)
  plot!(plt2, x,Ftrues,color="black")

  layout = @layout [ a ; b ]
  plt = Plots.plot(plt1,plt2, layout = layout)
  savefig(plt,"./figures/block_bootstrap_Fs_final2.pdf")

  x = -4:0.1:15.
  q0 = 0.99999
  x0 = quantile(Laplace(), q0)
  y_Fhat_bootstrap = Matrix{Float64}(undef, bootstrap_iters, length(x))
  for i in eachindex(x)
    _x = x[i]
    y_Fhat_bootstrap[:,i] = map(f->f(x0,_x), Fhats_boot)|>x->vcat(x...)
  end
  y_Fhat_bootstrap_q995 = mapslices(x->quantile(x,0.995), y_Fhat_bootstrap,dims=1)[:]
  y_Fhat_bootstrap_q005 = mapslices(x->quantile(x,0.005), y_Fhat_bootstrap,dims=1)[:]
  y_Fhat_bootstrap_median = mapslices(x->quantile(x,0.5), y_Fhat_bootstrap,dims=1)[:]
  y_Fhat_full = Fhat_full_model.(x0, x)
  ribbon = (y_Fhat_bootstrap_median.-y_Fhat_bootstrap_q005,
            y_Fhat_bootstrap_q995 .- y_Fhat_bootstrap_median)
  plt = plot(x,y_Fhat_bootstrap_median, ribbon=ribbon,xlab=L"z", title=L"\hat{F}_{1|0}(z;F_{0}^{-1}(%$(q0)))",
             linewidth=0, linecolor=false)
  plot!(plt,x,y_Fhat_full, color="red")
  Ftrues = map(x->F1_0(x,x0,θ,ν,scale=Laplace()), x)
  plot!(plt, x,Ftrues,color="black")
  savefig(plt,"./figures/block_bootstrap_Fs_final_bad_large2.pdf")


  # map(f->f(3.,3.), Fhats_boot[1:(M_ ÷ 2),1:(bootstrap_iters ÷ 2)])
  # x = -4:0.1:6.5
  # q0 = 0.975
  # x0 = quantile(Laplace(), q0)
  # y_Fhat_bootstrap = Matrix{Float64}(undef, N1 * N2, length(x))
  # for i in eachindex(x)
    # _x = x[i]
    # y_Fhat_bootstrap[:, i] = map(f->f(x0,_x), Fhats_boot[1:N1,1:N2])|>x->vcat(x...)
  # end
  # y_Fhat_bootstrap_q995 = mapslices(x->quantile(x,0.995), y_Fhat_bootstrap,dims=1)[:]
  # y_Fhat_bootstrap_q005 = mapslices(x->quantile(x,0.005), y_Fhat_bootstrap,dims=1)[:]
  # y_Fhat_bootstrap_median = mapslices(x->quantile(x,0.5), y_Fhat_bootstrap,dims=1)[:]
  # # y_Fhat_full = y_Fhat_bootstrap_median
  # y_Fhat_full = get_yFhat_full(x0, x)
  # ribbon = (y_Fhat_bootstrap_median.-y_Fhat_bootstrap_q005,
            # y_Fhat_bootstrap_q995 .- y_Fhat_bootstrap_median)
  # plt1 = plot(x,y_Fhat_bootstrap_median, ribbon=ribbon,xlab=L"x",
              # linewidth=0,
              # title=L"\hat{F}_{1|0}(x;F_{0}^{-1}(%$(q0)))",linecolor=false)
  # plot!(plt1,x,y_Fhat_full, color="red")
  # Ftrues = map(x->F1_0(x,x0,θ,ν,scale=Laplace()), x)
  # plot!(plt1, x,Ftrues,color="black")

  # q0 = 0.99
  # x0 = quantile(Laplace(), q0)
  # y_Fhat_bootstrap = Matrix{Float64}(undef, N1 * N2, length(x))
  # for i in eachindex(x)
    # _x = x[i]
    # y_Fhat_bootstrap[:, i] = map(f->f(x0,_x), Fhats_boot[1:N1,1:N2])|>x->vcat(x...)
  # end
  # y_Fhat_bootstrap_q995 = mapslices(x->quantile(x,0.995), y_Fhat_bootstrap,dims=1)[:]
  # y_Fhat_bootstrap_q005 = mapslices(x->quantile(x,0.005), y_Fhat_bootstrap,dims=1)[:]
  # y_Fhat_bootstrap_median = mapslices(x->quantile(x,0.5), y_Fhat_bootstrap,dims=1)[:]
  # y_Fhat_full = get_yFhat_full(x0, x)
  # ribbon = (y_Fhat_bootstrap_median.-y_Fhat_bootstrap_q005,
            # y_Fhat_bootstrap_q995 .- y_Fhat_bootstrap_median)
  # plt2 = plot(x,y_Fhat_bootstrap_median, ribbon=ribbon,xlab=L"x", title=L"\hat{F}_{1|0}(x;F_{0}^{-1}(%$(q0)))",
             # linewidth=0,linecolor=false)
  # plot!(plt2,x,y_Fhat_full, color="red")
  # Ftrues = map(x->F1_0(x,x0,θ,ν,scale=Laplace()), x)
  # plot!(plt2, x,Ftrues,color="black")

  # layout = @layout [ a ; b ]
  # plt = Plots.plot(plt1,plt2, layout = layout)
  # savefig(plt,"./figures/block_bootstrap_Fs_final.pdf")

  # x = -4:0.1:15.
  # q0 = 0.99999
  # x0 = quantile(Laplace(), q0)
  # y_Fhat_bootstrap = Matrix{Float64}(undef, N1 * N2, length(x))
  # for i in eachindex(x)
    # _x = x[i]
    # y_Fhat_bootstrap[:, i] = map(f->f(x0,_x), Fhats_boot[1:N1,1:N2])|>x->vcat(x...)
  # end
  # y_Fhat_bootstrap_q995 = mapslices(x->quantile(x,0.995), y_Fhat_bootstrap,dims=1)[:]
  # y_Fhat_bootstrap_q005 = mapslices(x->quantile(x,0.005), y_Fhat_bootstrap,dims=1)[:]
  # y_Fhat_bootstrap_median = mapslices(x->quantile(x,0.5), y_Fhat_bootstrap,dims=1)[:]
  # y_Fhat_full = get_yFhat_full(x0, x)
  # ribbon = (y_Fhat_bootstrap_median.-y_Fhat_bootstrap_q005,
            # y_Fhat_bootstrap_q995 .- y_Fhat_bootstrap_median)
  # plt = plot(x,y_Fhat_bootstrap_median, ribbon=ribbon,xlab=L"x", title=L"\hat{F}_{1|0}(x;F_{0}^{-1}(%$(q0)))",
             # linewidth=0, linecolor=false)
  # plot!(plt,x,y_Fhat_full, color="red")
  # Ftrues = map(x->F1_0(x,x0,θ,ν,scale=Laplace()), x)
  # plot!(plt, x,Ftrues,color="black")
  # savefig(plt,"./figures/block_bootstrap_Fs_final_bad_large.pdf")



  # q0 = 0.999
  # x0 = quantile(Laplace(), q0)
  # y_Fhat_bootstrap = map(f->f.(x0,x), Fhats)|>x->hcat(x...)
  # y_Fhat_bootstrap_q995 = mapslices(x->quantile(x,0.995), y_Fhat_bootstrap,dims=2)
  # y_Fhat_bootstrap_q005 = mapslices(x->quantile(x,0.005), y_Fhat_bootstrap,dims=2)
  # y_Fhat_bootstrap_mean = y_Fhat_bootstrap|>x->mean(x,dims=2)
  # y_Fhat_full = Fhat_resids.(x0,x)
  # ribbon = (y_Fhat_bootstrap_mean.-y_Fhat_bootstrap_q005,
            # y_Fhat_bootstrap_q995 .- y_Fhat_bootstrap_mean)
  # plt2 = plot(x,y_Fhat_bootstrap_mean, ribbon=ribbon, xlab=L"x", title=L"\hat{F}_{1|0}(x;F_{0}^{-1}(0.999))")
  # plot!(plt2,x,y_Fhat_full, color="red")
  # Ftrues = map(x->F1_0(x,x0,θ,ν,scale=Laplace()), x)
  # plot!(plt2, x,Ftrues,color="black")

  # layout = @layout [ a ; b ]
  # plt = Plots.plot(plt1,plt2, layout = layout)
  # savefig(plt,"./figures/block_bootstrap_Fs.pdf")

end




  # Fhats = Vector{Function}(undef, bootstrap_iters)
  # for i in 1:bootstrap_iters
    # K = 2
    # Xs_boot = datasets[i,:]
    # X_b = Xs_boot[1:end-1]; Y_b = Xs_boot[2:end]
    # X_exceed_b = X_b[X_b.>u]; Y_exceed_b = Y_b[X_b.>u]

    # ps_b = results_df[i,:]|>collect
    # ps_b =(π=ps_b[1:K], α=ps_b[K+1:2K], β=ps_b[2K+1:3K],
           # μ=ps_b[3K+1:4K], σ=ps_b[4K+1:5K])

    # C = class_of_normal_ps(ps_b,X_exceed_b,Y_exceed_b;K=2)
    # Z1 = (Y_exceed_b[C.==1] .- ps_b.α[1] .* X_exceed_b[C.==1]) ./ X_exceed_b[C.==1].^ps_b.β[1]
    # Z2 = (Y_exceed_b[C.==2] .- ps_b.α[2] .* X_exceed_b[C.==2]) ./ X_exceed_b[C.==2].^ps_b.β[2]

    # # π̂ = length(Z1)/(length(Z1) + length(Z2))
    # π̂ = ps.π[1]

    # Fhat_z1 = ecdf(Z1); Fhat_z2 = ecdf(Z2)

    # if π̂≈ 1
      # Fhats[i] = ((x,y) ->Fhat_z1((y - ps_b.α[1]*x)/x^ps_b.β[1]))
    # elseif π̂≈0
      # Fhats[i] = ((x,y) ->Fhat_z2((y - ps_b.α[2]*x)/x^ps_b.β[2]))
    # else
      # Fhats[i] = ((x,y) -> π̂ * Fhat_z1((y - ps_b.α[1]*x)/x^ps_b.β[1]) +
                             # (1-π̂) * Fhat_z2((y - ps_b.α[2]*x)/x^ps_b.β[2]))
    # end
  # end
  # df = results_df
  # CSV.write("./bootstrap-data/final.csv", results_df)


  # default(legend=false)
  # alpha=0.6
  # p1 = histogram(df.α1, label="α1",bins=20,normalize=:pdf, title=L"\alpha_1",xlim=[0.6,1],alpha=alpha,
                 # ylim=[0,Inf])
  # vline!(p1,[ps.α[1]],color="red",linestyle=:dash,linewidth=2)
  # p2 = histogram(df.α2, label="α2",bins=20,normalize=:pdf, title=L"\alpha_2",alpha=alpha,
                 # ylim=[0,Inf],xlim=[0,0.07],xticks=3)
  # vline!(p2,[ps.α[2]],color="red",linestyle=:dash,linewidth=2)
  # p3 = histogram(df.β1, label="β1",bins=20,normalize=:pdf, title=L"\beta_1",alpha=alpha,
                 # xticks=2, xlim=[0,3e-14], ylim=[0,Inf])
  # vline!(p3,[ps.β[1]],color="red",linestyle=:dash,linewidth=2)
  # p4 = histogram(df.β2, label="β2",bins=20,normalize=:pdf, title=L"\beta_2",alpha=alpha,
                 # xticks=3, xlim=[0,2*maximum(df.β2)],ylim=[0,Inf])
  # vline!(p4,[ps.β[2]],color="red",linestyle=:dash,linewidth=2)
  # layout = @layout [ a b ; c d ]
  # p = Plots.plot(p1, p2, p3, p4, layout=layout)
  # savefig(p,"./figures/block_bootstrap_params.pdf")
