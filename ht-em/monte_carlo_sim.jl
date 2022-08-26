using QuadGK
using Distributed
using StatsBase
using Statistics
using Distributions
using DataFrame

function analysis_1()
  df = CSV.read("./mc-sim-study/df_1_.csv", DataFrame)
  gdf = groupby(df, [:θ0,:ν01,:u,:scale])
  diff_names = names(df)|>x->filter(x->occursin("diff", x),x)|>x->map(Symbol,x)
  cdf_ = combine(gdf, :absdiffp97 .=> [mean,std],
               :absdiffp98 .=> [mean,std],
               :absdiffp99 .=> [mean,std],
               :absdiffp999 .=> [mean,std],
               :absdiffp9999 .=> [mean,std],
               :abs_sq_diffp97 .=> [mean,std],
               :abs_sq_diffp98 .=> [mean,std],
               :abs_sq_diffp99 .=> [mean,std],
               :abs_sq_diffp999 .=> [mean,std],
               :abs_sq_diffp9999 .=> [mean,std])

  abs_diff_mean = [:absdiffp97_mean,:absdiffp98_mean,:absdiffp99_mean,:absdiffp999_mean,:absdiffp9999_mean]
  abs_diff_std = [:absdiffp97_std,:absdiffp98_std,:absdiffp99_std,:absdiffp999_std,:absdiffp9999_std]
  _xticks = ["0.97","0.98","0.99","0.999","0.9999"]

  p1 = Plots.scatter((1:length(_xticks))|>x->map(x->x-0.1,x), eg_1[1,abs_diff_mean]|>Vector,
                xticks=(1:length(abs_diff_mean),_xticks), label="Logistic",
                yerror=eg_1[1,abs_diff_std]|>Vector,legend=:outerright)
  Plots.scatter!(p1,(1:length(_xticks))|>x->map(x->x+0.1,x),eg_1[2,abs_diff_mean]|>Vector,
                 xticks=(1:length(abs_diff_mean),_xticks), label="Laplace",
                 yerror=eg_1[2,abs_diff_std]|>Vector)
  mean_symbols = diff_names|>x->map(x->Symbol(string(x)*"_mean"),x)
  logistic_df = groupby(cdf_, [:scale])[1][:,mean_symbols]
  laplace_df = groupby(cdf_, [:scale])[2][:,mean_symbols]
  laplace_df_m = laplace_df|>Matrix
  logistic_df_m = logistic_df|>Matrix
  (laplace_df_m .- logistic_df_m) .<= 0

  eg_1 = cdf_[(cdf_.θ0.==0.3) .&& (cdf_.u .== 0.98) .&& (cdf_.ν01 .== 0.5),:]
  p1 = Plots.scatter((1:length(_xticks))|>x->map(x->x-0.1,x), eg_1[1,abs_diff_mean]|>Vector,
                xticks=(1:length(abs_diff_mean),_xticks), label="Logistic",
                yerror=eg_1[1,abs_diff_std]|>Vector,legend=:outerright)
  Plots.scatter!(p1,(1:length(_xticks))|>x->map(x->x+0.1,x),eg_1[2,abs_diff_mean]|>Vector,
                 xticks=(1:length(abs_diff_mean),_xticks), label="Laplace",
                 yerror=eg_1[2,abs_diff_std]|>Vector)

  fdf = cdf_[:,[:θ0,:ν01,:u]]|>Matrix|>x->unique(x,dims=1)
  fdf = fdf|>x->DataFrame(x,:auto)|>x->sort(x,[:x1,:x2])|>x->x[x[:,1].==0.3,:]|>Matrix
  # fdf = fdf[fdf.θ.==0.3,:]

  tmp=[(0.,0.) for _ in 1:size(fdf, 1)]
  fdf = DataFrame(θ=fdf[:,1],ν=fdf[:,2],u=fdf[:,3],
                  q97_lap=tmp, q98_lap=tmp, q99_lap=tmp, q999_lap=tmp, q9999_lap=tmp,
                  q97_log=tmp, q98_log=tmp, q99_log=tmp, q999_log=tmp, q9999_log=tmp)

  for i in 1:nrow(fdf)
    θ = fdf[i,:θ]; ν = fdf[i,:ν]; u = fdf[i,:u]
    lap_d = cdf_[(cdf_.θ0.==θ) .&& (cdf_.ν01.==ν) .&& (cdf_.u.==u) .&& (cdf_.scale .== string(Laplace())),:]
    log_d = cdf_[(cdf_.θ0.==θ) .&& (cdf_.ν01.==ν) .&& (cdf_.u.==u) .&& (cdf_.scale .== string(Logistic())),:]
    @assert nrow(lap_d) == nrow(log_d) == 1
    fdf[i,:q97_lap] = (lap_d[1,:abs_sq_diffp97_mean], lap_d[1,:abs_sq_diffp97_std])
    fdf[i,:q97_log] = (log_d[1,:abs_sq_diffp97_mean], log_d[1,:abs_sq_diffp97_std])
    fdf[i,:q98_lap] = (lap_d[1,:abs_sq_diffp98_mean], lap_d[1,:abs_sq_diffp98_std])
    fdf[i,:q98_log] = (log_d[1,:abs_sq_diffp98_mean], log_d[1,:abs_sq_diffp98_std])
    fdf[i,:q99_lap] = (lap_d[1,:abs_sq_diffp99_mean], lap_d[1,:abs_sq_diffp99_std])
    fdf[i,:q99_log] = (log_d[1,:abs_sq_diffp99_mean], log_d[1,:abs_sq_diffp99_std])
    fdf[i,:q999_lap] = (lap_d[1,:abs_sq_diffp999_mean], lap_d[1,:abs_sq_diffp999_std])
    fdf[i,:q999_log] = (log_d[1,:abs_sq_diffp999_mean], log_d[1,:abs_sq_diffp999_std])
    fdf[i,:q9999_lap] = (lap_d[1,:abs_sq_diffp9999_mean], lap_d[1,:abs_sq_diffp9999_std])
    fdf[i,:q9999_log] = (log_d[1,:abs_sq_diffp9999_mean], log_d[1,:abs_sq_diffp9999_std])
  end

  s = raw"$(v,q_u)$ & Marginal & $100\text{MISE}_{0.97}$ & $100\text{MISE}_{0.98}$& $100\text{MISE}_{0.99}$& $100\text{MISE}_{0.999}$& $100\text{MISE}_{0.9999}$\\"
  s *= raw"\\hline"

  for i in 1:nrow(fdf)
    v = fdf[i,:ν]; u = fdf[i,:u]
    val = []
    opts = [(:q97_log,:q97_lap),
            (:q98_log,:q98_lap),
            (:q99_log,:q99_lap),
            (:q999_log,:q999_lap),
            (:q9999_log,:q9999_lap)]
    for (log_,lap_) in opts
      miaelap = fdf[i,lap_] .* 100
      miaelog = fdf[i,log_] .* 100
      lap_better = miaelap[1] < miaelog[1]
      miaelap = round.(miaelap, digits=1)
      miaelog = round.(miaelog, digits=1)
      v1="";v2=""
      if lap_better
        v1 = "\$\\mathbf{$(miaelap[1]) \\pm $(miaelap[2])}\$"
        # v1 = "\\textbf{$(miaelap[1]) \$\\pm\$ $(miaelap[2])}"
        v2 = "\$$(miaelog[1]) \\pm $(miaelog[2])\$"
        push!(val,"\\textbf{$v1} \\newline $v2")
      else
        v1 = "\$$(miaelap[1]) \\pm $(miaelap[2])\$"
        v2 = "\$\\mathbf{$(miaelog[1]) \\pm $(miaelog[2])}\$"
        push!(val,"$v1 \\newline \\textbf{$v2}")
      end
    end
    s *= "($v, $u) &" *
         "Laplace \\newline Logistic &" *
         "$(val[1]) & " *
         "$(val[2]) &" *
         "$(val[3]) &" *
         "$(val[4]) &" *
         "$(val[5])"
    s *= "\\\\"
    s *= "\\hline"
  end
  open("./tex/lap_vs_log_e.tex","w") do io
    write(io, s)
  end


  fdf = cdf_[:,[:θ0,:ν01,:u]]|>Matrix|>x->unique(x,dims=1)
  fdf = fdf|>x->DataFrame(x,:auto)|>x->sort(x,[:x1,:x2])|>x->x[x[:,1].==0.3,:]|>Matrix

  tmp=[(0.,0.) for _ in 1:size(fdf, 1)]
  fdf = DataFrame(θ=fdf[:,1],ν=fdf[:,2],u=fdf[:,3],
                  q97_lap=tmp, q98_lap=tmp, q99_lap=tmp, q999_lap=tmp, q9999_lap=tmp,
                  q97_log=tmp, q98_log=tmp, q99_log=tmp, q999_log=tmp, q9999_log=tmp)

  for i in 1:nrow(fdf)
    θ = fdf[i,:θ]; ν = fdf[i,:ν]; u = fdf[i,:u]
    lap_d = cdf_[(cdf_.θ0.==θ) .&& (cdf_.ν01.==ν) .&& (cdf_.u.==u) .&& (cdf_.scale .== string(Laplace())),:]
    log_d = cdf_[(cdf_.θ0.==θ) .&& (cdf_.ν01.==ν) .&& (cdf_.u.==u) .&& (cdf_.scale .== string(Logistic())),:]
    @assert nrow(lap_d) == nrow(log_d) == 1
    fdf[i,:q97_lap] = (lap_d[1,:absdiffp97_mean], lap_d[1,:absdiffp97_std])
    fdf[i,:q97_log] = (log_d[1,:absdiffp97_mean], log_d[1,:absdiffp97_std])
    fdf[i,:q98_lap] = (lap_d[1,:absdiffp98_mean], lap_d[1,:absdiffp98_std])
    fdf[i,:q98_log] = (log_d[1,:absdiffp98_mean], log_d[1,:absdiffp98_std])
    fdf[i,:q99_lap] = (lap_d[1,:absdiffp99_mean], lap_d[1,:absdiffp99_std])
    fdf[i,:q99_log] = (log_d[1,:absdiffp99_mean], log_d[1,:absdiffp99_std])
    fdf[i,:q999_lap] = (lap_d[1,:absdiffp999_mean], lap_d[1,:absdiffp999_std])
    fdf[i,:q999_log] = (log_d[1,:absdiffp999_mean], log_d[1,:absdiffp999_std])
    fdf[i,:q9999_lap] = (lap_d[1,:absdiffp9999_mean], lap_d[1,:absdiffp9999_std])
    fdf[i,:q9999_log] = (log_d[1,:absdiffp9999_mean], log_d[1,:absdiffp9999_std])
  end

  s = raw"$(v,q_u)$ & Marginal & $100\text{MIAE}_{0.97}$ & $100\text{MIAE}_{0.98}$& $100\text{MIAE}_{0.99}$& $100\text{MIAE}_{0.999}$& $100\text{MIAE}_{0.9999}$\\"
  s *= raw"\\hline"

  for i in 1:nrow(fdf)
    v = fdf[i,:ν]; u = fdf[i,:u]
    val = []
    opts = [(:q97_log,:q97_lap),
            (:q98_log,:q98_lap),
            (:q99_log,:q99_lap),
            (:q999_log,:q999_lap),
            (:q9999_log,:q9999_lap)]
    for (log_,lap_) in opts
      miaelap = fdf[i,lap_] .* 100
      miaelog = fdf[i,log_] .* 100
      lap_better = miaelap[1] < miaelog[1]
      miaelap = round.(miaelap, digits=1)
      miaelog = round.(miaelog, digits=1)
      v1="";v2=""
      if lap_better
        v1 = "\$\\mathbf{$(miaelap[1]) \\pm $(miaelap[2])}\$"
        # v1 = "\\textbf{$(miaelap[1]) \$\\pm\$ $(miaelap[2])}"
        v2 = "\$$(miaelog[1]) \\pm $(miaelog[2])\$"
        push!(val,"\\textbf{$v1} \\newline $v2")
      else
        v1 = "\$$(miaelap[1]) \\pm $(miaelap[2])\$"
        v2 = "\$\\mathbf{$(miaelog[1]) \\pm $(miaelog[2])}\$"
        push!(val,"$v1 \\newline \\textbf{$v2}")
      end
    end
    s *= "($v, $u) &" *
         "Laplace \\newline Logistic &" *
         "$(val[1]) & " *
         "$(val[2]) &" *
         "$(val[3]) &" *
         "$(val[4]) &" *
         "$(val[5])"
    s *= "\\\\"
    s *= "\\hline"
  end
  open("./tex/lap_vs_log_e_miae.tex","w") do io
    write(io, s)
  end


end

function debug(x)
  println(x)
end


function X_Y_of_Xs(Xs; u)
  u_ixs1 = Xs.>u
  u_ixs1[:,end] .= 0
  u_ixs2 = mapslices(x->x>>1, u_ixs1, dims=2)

  X = Xs[u_ixs1][:]; Y = Xs[u_ixs2][:]
  return (X,Y)
end
