module QuasiArraysDomainSetsExt

using QuasiArrays, DomainSets
import QuasiArrays: cardinality
import DomainSets: ClosedInterval

cardinality(d::UnionDomain) = sum(cardinality, d.domains)
cardinality(d::VcatDomain) = prod(cardinality, d.domains)

# use UnionDomain to support intervals
_union(a::ClosedInterval...) = UnionDomain(a...)

end
