using Printf
using Roots

function empirical_transform_copula(dist::UnivariateDistribution,
                                    data::Matrix{<:Real};
                                    minq=0, maxq=length(data)/(length(data)+1))
  # maps data along each dimension to supp(dist)^d
  n,d = size(data)
  res = deepcopy(data);
  for j in 1:d
    res[:,j] = minq .+ (maxq - minq) .* ecdf(data[:,j]).(data[:,j])
  end
  res = quantile.(dist, res)
  res
end
empirical_transform_copula(dist::UnivariateDistribution,
                           data::Vector{<:Real}; kwargs...) =
  empirical_transform_copula(dist, hcat(data); kwargs...)[:,1]

sigmoid(x; loc=0, scale=1) = 1/(1 + exp(-(x - loc)/scale))
# logit(p) = log(p) - log(1-p)
logit(p; loc=0, scale=1) = loc + scale*log(p/(1-p))

"""
logit_div(x,y) = logit(x/y)
"""
logit_div(x,y) = log(x) - log(y) - log(1-x/y)

function assess_convergence(θold, θnew, fθold, fθnew;
                            atol=1e-6, rtol=1e-6,
                            θatol=1e-6, θrtol=1e-6)
  convergence_failed = any(isnan.(θnew)) || isinf(fθnew)
  θ_converged = abs.(θold .- θnew) .<= max.(θatol, max.(abs.(θold),abs.(θnew)) .* θrtol)
  f_converged = abs(fθnew-fθold) < max(atol, max(abs(fθold),abs(fθnew))*rtol)

  (convergence_failed=convergence_failed,
   θ_converged=θ_converged,
   f_converged=f_converged)
end

"""
Convert vector of vectors to a matrix

If input vectors do not all have the same length, then
we pad them with zeros on the end

```
[[1,2],[1,2,3]] |> m_of_vv

3×2 Matrix{Int64}:
 1  1
 2  2
 0  3
```
"""
function m_of_vv(vv)
  if all(x -> length(x) == length(vv[1]), vv)
    reduce(hcat, vv)
  else
    maxK = maximum(map(length, vv))
    _vv = map(x -> [x..., [0 for _ in 1:(maxK - length(x))]...], vv)
    reduce(hcat, _vv)
  end
end

function tex_table_string_of_df(df; headers, sigfig=4, sigfigpercol=nothing, printf=:f)
  @assert printf == :f || printf == :e
  df_new = string.(df)
  @assert length(headers) == ncol(df)
  if isnothing(sigfigpercol)
    sigfigpercol=[sigfig for _ in 1:ncol(df)]
  end
  for i in 1:nrow(df)
    r = df[i,:]|>Vector
    ixs = isa.(r,AbstractFloat)
    for j in (1:ncol(df))[ixs]
      if printf == :f
        df_new[i,j] = Printf.format.(Ref(Printf.Format("%.$(sigfigpercol[j])f")), r[j])
      elseif printf == :e
        df_new[i,j] = Printf.format.(Ref(Printf.Format("%.$(sigfigpercol[j])e")), r[j])
      end
    end
  end
  df = df_new
  o  = "\\begin{tabular}{" * "c |" ^ (ncol(df)-1) * "c" * "}\n";
  o *= headers[1]
  for i in 2:length(headers)
    o *= " & " * headers[i]
  end; o *= " \\\\\n"
  o *= "    \\hline\n";

  for row in 1:nrow(df)
    o *= "  "
    o *= string(df[row,1])
    for col in 2:ncol(df)
      o *= " & " * df[row,col]
    end; o *= " \\\\\n";
  end
  o *= "\\end{tabular}\n";
  o
end

function solve_ecdf_eq_u_in_R(func, u)
  ## solve func(x) = u
  ## we use [0,1] as the search space, so that bisection methods
  ## can be used
  function g(x)
    if x≈0
      -u
    elseif x≈1
      1-u
    else
      y = logit(x)
      func(y) - u
    end
  end
  res = fzero(g, (0,1))
  logit(res)
end
