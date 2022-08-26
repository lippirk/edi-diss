using Plots
using DataFrames
using LaTeXStrings

function plot_basic_realization()
  println("plot_basic_realization...")
  seed = 2
  q = 0.995
  M = 1; N = 30
  θ1=[1,1,1,0,0,0];ν1=[1,1,1]
  Xs1 = gen_Xs(M=M, N=N, θ=θ1,ν=ν1, seed=seed,q=q)

  θ2=[0.3,0.3,0.3,0.3,0.3,0.1];ν2=[0.7,0.7,0.7]
  Xs2 = gen_Xs(M=M, N=N, θ=θ2,ν=ν2, seed=seed,q=q)

  θ3=[0.3,0.3,0.3,0.3,0.3,0.1];ν3=[0.3,0.3,0.3]
  Xs3 = gen_Xs(M=M, N=N, θ=θ3,ν=ν3, seed=seed,q=q)

  θ4=[0,0,0,0.05,0.05,0.9];ν4=[0.3,0.3,0.3]
  Xs4 = gen_Xs(M=M, N=N, θ=θ4,ν=ν4, seed=seed,q=q)

  Xs = vcat(Xs1, Xs2, Xs3, Xs4)|>transpose
  p = Plots.plot(Xs, yaxis=:log, legend=:outertop,
             labels=permutedims([L"\theta_{0}=\theta_{1}=\theta_{2}=1,v=1",
                     L"\theta_0=\theta_1=\theta_2=0.3,θ_{012}=0.1,v=0.7",
                     L"\theta_0=\theta_1=\theta_2=0.3,θ_{012}=0.1,v=0.3",
                     L"\theta_0=\theta_1=\theta_2=0,θ_{012}=0.9,v=0.3"]),
             ylab=L"X_t",xlab=L"t");
  u = quantile(Frechet(), q)
  Plots.abline!(p,0,u, label=L"u=%$(round(u, digits=3))", color="red",
                linestyle=:dash)
  println("plot_basic_realization: saving...")
  savefig(p, "figures/3dim-basic-realization.pdf")
end

function plot_x2_vs_x1_laplace_scale()

  seed = 3; q=0.995; M=100; N=30;
  scale_dist = Laplace()
  # u = quantile(Frechet(), q)
  u = quantile(scale_dist, q)

  d = Dict()
  θ=[0.3,0.3,0.3,0.3,0.3,0.1]
  function go(v)
    ν=[v for _ in 1:3]
    Xs = gen_Xs(M=M, N=N, θ=θ,ν=ν, seed=seed,q=q)
    Xs = quantile.(scale_dist, cdf.(Frechet(), Xs))# move to exp scale
    u_ixs1 = Xs.>u
    u_ixs1[:,end] .= 0 # don't include last point in each series
    u_ixs2 = mapslices(x->x>>1, u_ixs1, dims=2) # shift each row to the right

    @assert sum(u_ixs1) == sum(u_ixs2)
    x2 = Xs[u_ixs2]
    x1 = Xs[u_ixs1]

    d[v] = (x1, x2)
  end

  for v in [0.2,0.4,0.6,0.8]
    go(v)
  end

  default(thickness_scaling=1, markersize=2, markerstrokewidth=-1,
          legend=false, titlefontsize=12, xguidefontsize=12, yguidefontsize=12)#plot default
  p1 = Plots.scatter(d[0.2][1], d[0.2][2], title=L"v=0.2", color="black",
                     ylab=L"X_{t+1}", xticks=5:11)
  Plots.abline!(p1, 0, 0, alpha=0.5, color="grey", linestyle=:dash)
  Plots.abline!(p1, 1, 0, alpha=0.5, color="grey", linestyle=:dash)

  p2 = Plots.scatter(d[0.4][1], d[0.4][2], title=L"v=0.4", color="black",
                     xticks=5:11)
  Plots.abline!(p2, 0, 0, alpha=0.5, color="grey",linestyle=:dash)
  Plots.abline!(p2, 1, 0, alpha=0.5, color="grey",linestyle=:dash)

  p3 = Plots.scatter(d[0.6][1], d[0.6][2], title=L"v=0.6", color="black",
                     xlab=L"X_{t}", ylab=L"X_{t+1}", xticks=5:11)
  Plots.abline!(p3, 0, 0, alpha=0.5, color="grey",linestyle=:dash)
  Plots.abline!(p3, 1, 0, alpha=0.5, color="grey",linestyle=:dash)

  p4 = Plots.scatter(d[0.8][1], d[0.8][2], title=L"v=0.8", color="black",
                     xlab=L"X_{t}", xticks=5:11)
  Plots.abline!(p4, 0, 0, alpha=0.5, color="grey",linestyle=:dash)
  Plots.abline!(p4, 1, 0, alpha=0.5, color="grey",linestyle=:dash)
  l = @layout [ a b ; c d ]
  p = Plots.plot(p1, p2, p3, p4, layout=l)
  savefig(p, "figures/3dim-x2-vs-x1-laplace-scale.pdf")
end
