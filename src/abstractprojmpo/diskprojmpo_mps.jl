mutable struct DiskProjMPO_MPS
  PH::DiskProjMPO
  pm::Vector{ProjMPS}
  weight::Float64
end

copy(P::DiskProjMPO_MPS) = DiskProjMPO_MPS(copy(P.PH), copy.(P.pm), P.weight)

function DiskProjMPO_MPS(H::MPO, mpsv::Vector{MPS}; weight=1.0)
  return DiskProjMPO_MPS(DiskProjMPO(H), [ProjMPS(m) for m in mpsv], weight)
end

DiskProjMPO_MPS(H::MPO, Ms::MPS...; weight=1.0) = DiskProjMPO_MPS(H, [Ms...], weight)

nsite(P::DiskProjMPO_MPS) = nsite(P.PH)

disk(Ps::ProjMPO_MPS; kwargs...) = DiskProjMPO_MPS(disk(Ps.PH; kwargs...), Ps.pm, Ps.weight)
disk(Ps::DiskProjMPO_MPS; kwargs...) = Ps

function set_nsite!(Ps::DiskProjMPO_MPS, nsite)
  set_nsite!(Ps.PH, nsite)
  for P in Ps.pm
    set_nsite!(P, nsite)
  end
  return Ps
end

Base.length(P::DiskProjMPO_MPS) = length(P.PH)

function site_range(P::DiskProjMPO_MPS)
  r = site_range(P.PH)
  @assert all(m -> site_range(m) == r, P.pm)
  return r
end

function product(P::DiskProjMPO_MPS, v::ITensor)::ITensor
  Pv = product(P.PH, v)
  for p in P.pm
    Pv += P.weight * product(p, v)
  end
  return Pv
end

function Base.eltype(P::DiskProjMPO_MPS)
  elT = eltype(P.PH)
  for p in P.pm
    elT = promote_type(elT, eltype(p))
  end
  return elT
end

(P::DiskProjMPO_MPS)(v::ITensor) = product(P, v)

Base.size(P::DiskProjMPO_MPS) = size(P.H)

function position!(P::DiskProjMPO_MPS, psi::MPS, pos::Int)
  position!(P.PH, psi, pos)
  for p in P.pm
    position!(p, psi, pos)
  end
  return P
end

noiseterm(P::DiskProjMPO_MPS, phi::ITensor, dir::String) = noiseterm(P.PH, phi, dir)

function checkflux(P::DiskProjMPO_MPS)
  checkflux(P.PH)
  foreach(checkflux, P.pm)
  return nothing
end
