using MarketData
using CSV
using DataFrames
using Plots
using Statistics
using LaTeXStrings
using StatsBase
using Distributions

default(thickness_scaling=1, markersize=4, markerstrokewidth=-1,
        legend=false, titlefontsize=12, xguidefontsize=8, yguidefontsize=8)#plot default

# includet("../util/util.jl")


skipna(xs) = filter(x->!isnan(x), xs)

log_returns(xs) = log.(1 .+ diff(xs) ./ xs[2:end])

function to_laplace(xs)
 quantile.(Laplace(), (ecdf(xs).(xs) .* (n-1) ./ n))
end

function fetch_spy()
  # s_and_p_500 = "^GSPC"
  # data = yahoo(s_and_p_500)
  # data
  # CSV.write("./data/spy.csv", data)
  df = CSV.read("./data/spy.csv", DataFrame)
  df
end

function fetch_btc()
  df = CSV.read("data/binance-btc-usdt.csv", DataFrame, header=2)
  df.date = df.date|>collect|>x->map(x->x[1:length("yyyy-mm-dd")],x)|>x->map(Date,x)
  df = sort(df, :date)
  df
end

function fetch_elexon_2021_resids()
  df = CSV.read("../elexon/residuals/mgcv-3.csv", DataFrame)
  data = df.x
  data
end

function plot_spy_tail_switch(data=fetch_spy())
  dates = data.timestamp
  _log_returns = [NaN; data.Close|>values|>log_returns]
  n = length(_log_returns|>skipna)
  Fhat(x) = ecdf(_log_returns|>skipna).(x) .* (n-1) ./ n
  _log_returns = map(x->isnan(x) ? x : quantile(Laplace(),Fhat(x)),_log_returns)

  @assert length(_log_returns) == length(dates)
  q95 = quantile(Laplace(), 0.95)

  excess_ixs = (_log_returns .> q95) .|| ((_log_returns .> q95) << 1)

  # ixs = [all(excess_ixs[i:(i+5)]) for i in 1:length(excess_ixs) - 5]
  # findall(ixs)

  rotat = 0
  # xticks = 4
  interesting_ixs = 7985:8005
  _xticks = ([5, 10, 15,20], dates[7985 .+ [5,10,15,20]])
  dates = map(string,dates)
  markers = [_log_returns[i-1] > q95 && _log_returns[i] < q95 ? :star : :o for
             i in interesting_ixs]
  p1 = Plots.plot(1:length(interesting_ixs), _log_returns[interesting_ixs],
                  ylab="Log returns\n(Laplace scale)", label=false, markershape=markers,
                  legend=false, xrotation=rotat,xticks=_xticks, color="black",
                  xguidefontsize=4
                 )
                 # legend=:topleft)
  # Plots.abline!(p1, 0, q95, label=L"$u$ = 95% quantile", color="red")
  Plots.abline!(p1, 0, q95, color="grey",label=false,linestyle=:dash)

  p2 = Plots.plot(1:length(interesting_ixs), (data.Close|>values)[interesting_ixs],
                  ylab="Price (USD)", label=false, xrotation=rotat,xticks=_xticks, color="black",
                 xguidefontsize=4)

  layout = @layout [ a ; b ]
  # p = Plots.plot(p2, p1, layout=layout, plot_title="S&P 500");
  p = Plots.plot(p2, p1, layout=layout, plot_title="")
  savefig(p, "figures/spy_tail_switch.pdf")
end

function plot_btc_tail_switch(data=fetch_btc())
  dates = data.date
  _log_returns = [NaN; data.close|>collect|>log_returns]

  @assert length(_log_returns) == length(dates)
  q95 = quantile(_log_returns|>skipna, 0.95)

  excess_ixs = (_log_returns .> q95) .|| ((_log_returns .> q95) << 1)
  x1 = _log_returns[excess_ixs];
  x2 = _log_returns[excess_ixs>>1];
  x3 = _log_returns[excess_ixs>>2]

  # ixs = [all(excess_ixs[i:(i+3)]) for i in 1:length(excess_ixs) - 3]
  # findall(ixs)

  interesting_ixs = 160:200

  markers = [_log_returns[i-1] > q95 && _log_returns[i] < q95 ? :star : :o for
             i in interesting_ixs]
  p1 = Plots.plot(dates[interesting_ixs], _log_returns[interesting_ixs],
                 ylab="Log returns", label=false, markershape=markers,
                 legend=:bottomright)
  Plots.abline!(p1, 0, q95, label=L"$u$ = 95% quantile", color="red")

  p2 = Plots.plot(dates[interesting_ixs], data.close[interesting_ixs],
                  ylab="Price (USD)", label=false)

  layout = @layout [ a ; b ]
  p = Plots.plot(p2, p1, layout=layout, plot_title="");
  savefig(p, "figures/btc_tail_switch.pdf")
end

function plot_btc_modes(data=fetch_btc())
  dates = data.date
  _log_returns = data.close|>collect|>log_returns
  xs = empirical_transform_copula(Logistic(), _log_returns)
  q95 = quantile(xs, 0.95)

  excess_ixs = (xs .> q95) .|| ((xs .> q95) << 1)
  u_ixs = (xs .> q95)
  # Plots.scatter(xs[u_ixs], xs[u_ixs >> 1])
  Plots.scatter(xs[u_ixs], xs[u_ixs >> 1])
end


function plot_spy_modes(data=fetch_spy())
  dates = data|>timestamp
  _log_returns = data.Close|>values|>log_returns
  xs = empirical_transform_copula(Logistic(), _log_returns)
  q95 = quantile(xs, 0.99)

  excess_ixs = (xs .> q95) .|| ((xs .> q95) << 1)
  u_ixs = (xs .> q95)
  # Plots.scatter(xs[u_ixs], xs[u_ixs >> 1])
  Plots.scatter(xs[u_ixs], xs[u_ixs >> 2])
end

function plot_elexon_2021_resids_modes(data=fetch_elexon_2021_resids())
  xs = data
  xs = empirical_transform_copula(Logistic(), _log_returns)
  q95 = quantile(xs, 0.95)

  excess_ixs = (xs .> q95) .|| ((xs .> q95) << 1)
  u_ixs = (xs .> q95)
  Plots.scatter(xs[u_ixs], xs[u_ixs >> 1])
  # Plots.scatter(xs[excess_ixs], xs[excess_ixs >> 1], legend=false)
end
