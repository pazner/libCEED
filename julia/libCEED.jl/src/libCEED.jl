module libCEED

using CUDA, StaticArrays, UnsafeArrays

# import low-level C interface
include("C.jl")
import .C

# types and functions
export CeedScalar, CeedInt
export Ceed, getresource, isdeterministic, iscuda, get_preferred_memtype
export Basis, BasisCollocated, create_tensor_h1_lagrange_basis, create_tensor_h1_basis, create_h1_basis, apply!, apply, getdimension, gettopology, getnumcomponents, getnumnodes, getnumnodes1d, getnumqpts, getnumqpts1d
export ElemRestriction, ElemRestrictionNone, create_elem_restriction, create_elem_restriction_strided
export CeedVector, CeedVectorActive, CeedVectorNone, setvalue!, reciprocal!, witharray, witharray_read, @witharray, @witharray_read, setarray!, sync_array!, take_array!, norm
export gauss_quadrature, lobatto_quadrature, Abscissa, AbscissaAndWeights
export UserQFunction, QFunction, QFunctionNone, create_interior_qfunction, create_identity_qfunction, add_input!, add_output!, set_context!, apply!
export extract_context, extract_array, @interior_qf
export Operator, set_field!, apply!
export Context, set_data!
export RequestImmediate, RequestOrdered
export CeedDim, det, setvoigt, setvoigt!, getvoigt, getvoigt!
# CUDA
export set_cufunction!
# enums and globals
export QuadMode, GAUSS, GAUSS_LOBATTO
export MemType, MEM_HOST, MEM_DEVICE
export CopyMode, COPY_VALUES, USE_POINTER, OWN_POINTER
export EvalMode, EVAL_NONE, EVAL_INTERP, EVAL_GRAD, EVAL_DIV, EVAL_CURL, EVAL_WEIGHT
export NormType, NORM_1, NORM_2, NORM_MAX
export TransposeMode, NOTRANSPOSE, TRANSPOSE
export Topology, LINE, TRIANGLE, QUAD, TET, PYRAMIC, PRISM, HEX
export STRIDES_BACKEND

include("Globals.jl")
include("Ceed.jl")
include("CeedVector.jl")
include("Basis.jl")
include("ElemRestriction.jl")
include("Quadrature.jl")
include("Context.jl")
include("UserQFunction.jl")
include("QFunction.jl")
include("Request.jl")
include("Operator.jl")
include("Misc.jl")
include("Cuda.jl")

function __init__()
    set_globals()
end

end # module
