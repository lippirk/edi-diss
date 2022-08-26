function X_Y_of_Xs(Xs; u)
  u_ixs1 = Xs.>u
  u_ixs1[:,end] .= 0 # don't include last point in each series
  u_ixs2 = mapslices(x->x>>1, u_ixs1, dims=2) # shift each row to the right

  X = Xs[u_ixs1][:]; Y = Xs[u_ixs2][:]
  return (X,Y)
end

function perturb_inits_K_2(inits; _typ=:conservative)
  @assert _typ == :conservative || _typ == :aggressive
  if _typ == :conservative
    return rand.([TruncatedNormal.(inits[1:5],0.01,0,1);
                  TruncatedNormal.(inits[6:7],0.5,-Inf,Inf);
                  TruncatedNormal.(inits[8:9],0.5,0,Inf);
                  ])
  elseif _typ == :aggressive
    return rand.([TruncatedNormal.(inits[1:5],0.1,0,1);
                  TruncatedNormal.(inits[6:7],0.8,-Inf,Inf);
                  TruncatedNormal.(ifelse.(inits[8:9] .≥ 1, inits[8:9], 1.),0.8,1.,Inf);
                  ])

  end
end

function weighted_ht_ecdf_K_2(X,Y;ps)
  K = 2
  n = length(X)
  @assert all(X .≥ 0)
  π = ps.π;
  α = ps.α; β = ps.β; μ = ps.μ; σ = ps.σ
  π̃ = zeros(n,K)
  Z = zeros(n,K)
  for i in 1:n
    Z[i,:] = (Y[i] .- α .* X[i]) ./ X[i] .^ β
    π̃[i,:] = π .* pdf.(Normal.(μ,σ), Z[i,:])
    π̃[i,1] = π̃[i,1] ./ sum(π̃[i,:])
    π̃[i,2] = 1 .- π̃[i,1]
    @assert isprobvec(π̃[i,:])
  end

  return function (x,y)
    res = 0
    z = (y .- α .* x) ./ x .^ β
    for i = 1:n
      if Z[i,1] ≤ z[1]
        res += π̃[i,1]
      end
      if Z[i,2] ≤ z[2]
        res += π̃[i,2]
      end
    end
    res ./ n
  end
end
