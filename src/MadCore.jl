module MadCore

import Pkg.TOML: parsefile
import Printf: @sprintf
import LinearAlgebra:
    BLAS, LAPACK, Adjoint, Symmetric, Diagonal,
    mul!, ldiv!, rdiv!, lmul!, rmul!, norm, dot, diagind,
    transpose!, issuccess, BlasReal,
    bunchkaufman, cholesky, qr, lu,
    bunchkaufman!, cholesky!, axpy!, LowerTriangular
import LinearAlgebra.BLAS: libblastrampoline, BlasInt, @blasfunc
import SparseArrays:
    SparseArrays, AbstractSparseMatrix, SparseMatrixCSC, sparse,
    getcolptr, rowvals, nnz, nonzeros
import Base: string, show, print, size, getindex, copyto!, @kwdef
import NLPModels
import NLPModels:
    finalize, AbstractNLPModel, obj, grad!, cons!,
    jac_coord!, hess_coord!, hess_structure!, jac_structure!,
    hess_dense!, jac_dense!, NLPModelMeta,
    get_nvar, get_ncon, get_minimize, get_x0, get_y0,
    get_nnzj, get_nnzh, get_lvar, get_uvar, get_lcon, get_ucon
import SolverCore:
    getStatus, AbstractOptimizationSolver, AbstractExecutionStats, solve!
import OpenBLAS32_jll

function __init__()
    config = BLAS.lbt_get_config()
    if !any(lib -> lib.interface == :lp64, config.loaded_libs)
        BLAS.lbt_forward(OpenBLAS32_jll.libopenblas_path)
    end
end

include("enums.jl")
include("utils.jl")
include("matrixtools.jl")
include(joinpath("Callbacks", "nlpmodels.jl"))
include(joinpath("Callbacks", "wrappers.jl"))
include("quasi_newton.jl")
include(joinpath("KKT", "KKTsystem.jl"))
include(joinpath("LinearSolvers", "linearsolvers.jl"))

end # module
