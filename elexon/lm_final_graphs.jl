using Plots
using CSV
using DataFrames
using StatsBase
using Distributions
using LaTeXStrings

default(thickness_scaling=1, markersize=4, markerstrokewidth=-1,
        legend=false, titlefontsize=12, xguidefontsize=12, yguidefontsize=12)
function get_resids(;margin=Laplace())
  xs = CSV.read("./residuals/lm-final.csv", DataFrame)|>Matrix
  xs = xs[:,2]

  n = length(xs)
  xs = n/(n+1) .* ecdf(xs).(xs)

  xs = quantile.(margin, xs)
  xs
end

function get_jan22_true_prices()
  ys = CSV.read("./residuals/jan-22-true-prices.csv", DataFrame)|>Matrix
  ys[:,2]
end

function get_ys_ecdf()
  ys = CSV.read("./residuals/jan-22.csv", DataFrame)|>Matrix
  ys = ys[:,2]

  n = length(ys)
  ecdf(ys)
end

function get_jan22_resids(;margin=Laplace())
  # train_resids = CSV.read("./residuals/lm-final.csv", DataFrame)|>Matrix
  # train_resids = train_resids[:,2]

  ys = CSV.read("./residuals/jan-22.csv", DataFrame)|>Matrix
  ys = ys[:,2]

  n = length(ys)
  ys = n/(n+1) .* ecdf(ys).(ys)

  ys = quantile.(margin, ys)
  ys
end


u = quantile(Laplace(), 0.95)
function plot_extreme_dep_first_order(resids; u=u)
  default(thickness_scaling=1, markersize=3, markerstrokewidth=-1,
          legend=false)
  xs = resids

  u_ixs = xs .> u
  u_ixs[end] = 0
  p = Plots.scatter(xs[u_ixs], xs[u_ixs>>1], xlab=L"y_{t}", ylab=L"y_{t+1}");
  Plots.abline!(p, -0.9, 0, color="black", linestyle=:dash);
  Plots.abline!(p, 0.7, 0, color="black", linestyle=:dash);
  savefig(p, "figures/first-order-extreme-dep.pdf")
end

function plot_raw_first_order(resids; u = u)
  default(thickness_scaling=1, markersize=3, markerstrokewidth=-1,
          legend=false)
  u = 2.8
  xs = resids

  p1 = Plots.scatter(xs[1:end-1], xs[2:end], xlab=L"y_{t}", ylab=L"y_{t+1}",
                     title="(a)", color=:black);
  Plots.vline!(p1, [u], color="grey", linestyle=:dash);

  u_ixs = xs .≥ u
  u_ixs[end] = 0
  p2 = Plots.scatter(xs[u_ixs], xs[u_ixs>>1], xlab=L"y_{t}", ylab=L"y_{t+1}",
                     title="(b)", color=:black);
  l = @layout [ a  b ]
  p = Plots.plot(p1, p2, layout=l)
  savefig(p, "figures/first-order-raw.pdf")
end
