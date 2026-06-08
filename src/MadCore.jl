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

# Public API. MadCore owns the solver-agnostic surface of MadNLP (KKT systems,
# core linear solvers, options, logger, quasi-Newton, callbacks, status). MadNLP
# `@reexport using MadCore`s these so the historical `MadNLP.*` names resolve.
# Explicit list (per REFACTOR_PLAN.md §4.1) — preferred over names()-introspection
# so the exported surface is reviewable and stable.
export
    # KKT systems
    AbstractKKTSystem, AbstractReducedKKTSystem, AbstractCondensedKKTSystem,
    AbstractUnreducedKKTSystem,
    SparseKKTSystem, SparseUnreducedKKTSystem, SparseCondensedKKTSystem,
    ScaledSparseKKTSystem, SchurComplementKKTSystem,
    DenseKKTSystem, DenseCondensedKKTSystem,
    UnreducedKKTVector,
    # Linear solvers (core; backend solvers live in lib/* subpackages)
    AbstractLinearSolver, LapackCPUSolver, LapackOptions,
    # Options / logging
    MadNLPLogger, AbstractOptions,
    # Matrix tools
    SparseMatrixCOO, coo_to_csc,
    # Quasi-Newton
    AbstractHessian, ExactHessian, BFGS, DampedBFGS, CompactLBFGS,
    QuasiNewtonOptions,
    # Status / enums
    Status, LogLevels, get_status_output,
    SOLVE_SUCCEEDED, SOLVED_TO_ACCEPTABLE_LEVEL,
    SEARCH_DIRECTION_BECOMES_TOO_SMALL, DIVERGING_ITERATES,
    INFEASIBLE_PROBLEM_DETECTED, MAXIMUM_ITERATIONS_EXCEEDED,
    MAXIMUM_WALLTIME_EXCEEDED, INITIAL, REGULAR, RESTORE, ROBUST,
    LINESEARCH_SUCCEEDED, RESTORATION_FAILED, INVALID_NUMBER_DETECTED,
    ERROR_IN_STEP_COMPUTATION, NOT_ENOUGH_DEGREES_OF_FREEDOM,
    USER_REQUESTED_STOP, INTERNAL_ERROR, INVALID_NUMBER_OBJECTIVE,
    INVALID_NUMBER_GRADIENT, INVALID_NUMBER_CONSTRAINTS,
    INVALID_NUMBER_JACOBIAN, INVALID_NUMBER_HESSIAN_LAGRANGIAN,
    # SolverCore re-export (historically exported by MadNLP)
    solve!

end # module
