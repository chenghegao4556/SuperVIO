//
// Created by chenghe on 4/2/20.
//

#include <optimization/optimizer.h>

#include <utility>

namespace SuperVIO::Optimization
{

    ///////////////////////////////////////////////////////////////////////////////////
    Optimizer::Options::
    Options(std::string  _trust_region_strategy,
           std::string  _linear_solver_type,
           int _max_iteration,
           int _num_threads,
           double _max_solver_time,
           bool _verbose):
            trust_region_strategy(std::move(_trust_region_strategy)),
            linear_solver_type(std::move(_linear_solver_type)),
            max_iteration(_max_iteration),
            num_threads(_num_threads),
            max_solver_time(_max_solver_time),
            verbose(_verbose)

    {

    }

    ///////////////////////////////////////////////////////////////////////////////////
    ceres::Solver::Options Optimizer::Options::
    ToCeres() const
    {
        ceres::Solver::Options options;
        if(trust_region_strategy == "lm")
        {
            options.trust_region_strategy_type = ceres::TrustRegionStrategyType::LEVENBERG_MARQUARDT;
        }
        else if(trust_region_strategy == "dogleg")
        {
            options.trust_region_strategy_type = ceres::TrustRegionStrategyType::DOGLEG;
        }
        else
        {
            ROS_ERROR_STREAM("unsupport trust region strategy: "<<trust_region_strategy);
            throw std::runtime_error("unsupport trust region strategy");
        }

        if(linear_solver_type == "iterative_schur")
        {
            options.linear_solver_type = ceres::LinearSolverType::ITERATIVE_SCHUR;
            options.use_explicit_schur_complement = true;
            options.preconditioner_type = ceres::PreconditionerType::SCHUR_JACOBI;
        }
        else if(linear_solver_type == "dense_schur")
        {
            options.use_explicit_schur_complement = true;
            options.linear_solver_type = ceres::LinearSolverType::DENSE_SCHUR;
        }
        else if(linear_solver_type == "sparse_normal_cholesky")
        {
            options.linear_solver_type = ceres::LinearSolverType::SPARSE_NORMAL_CHOLESKY;
            options.sparse_linear_algebra_library_type = ceres::SparseLinearAlgebraLibraryType::SUITE_SPARSE;
        }
        else
        {
            ROS_ERROR_STREAM("unsupport linear solver type: "<<linear_solver_type);
            throw std::runtime_error("unsupport linear solver type");
        }
        options.max_solver_time_in_seconds = max_solver_time;
        options.num_threads = num_threads;
        options.max_num_iterations = max_iteration;
        options.minimizer_progress_to_stdout = verbose;

        return options;
    }

    ///////////////////////////////////////////////////////////////////////////////////
    void Optimizer::
    Construct(const Options& options,
              const std::vector<ParameterBlockPtr>& parameter_blocks,
              const std::vector<ResidualBlockPtr>&  residual_blocks)
    {
        ceres::Problem problem;
        for(const auto& parameter_block: parameter_blocks)
        {
            problem.AddParameterBlock(parameter_block->GetData(),
                                      parameter_block->GetGlobalSize(),
                                      parameter_block->GetLocalParameterization());

            if(parameter_block->IsFixed())
            {
                problem.SetParameterBlockConstant(parameter_block->GetData());
            }
        }
        for(const auto& residual_block: residual_blocks)
        {
            problem.AddResidualBlock(residual_block->GetCostFunction(),
                                     residual_block->GetLostFunction(),
                                     residual_block->GetData());
        }

        ceres::Solver::Summary summary;
        ceres::Solve(options.ToCeres(), &problem , &summary);

        if(options.verbose)
        {
            std::cout << summary.FullReport() << "\n";
        }
    }


}//end of SuperVIO