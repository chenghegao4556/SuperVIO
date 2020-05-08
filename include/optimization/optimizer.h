//
// Created by chenghe on 4/2/20.
//

#ifndef SUPER_VIO_OPTIMIZER_H
#define SUPER_VIO_OPTIMIZER_H
#include <optimization/residual_blocks/residual_block_factory.h>
#include <optimization/parameter_blocks/parameter_block_factory.h>
namespace SuperVIO::Optimization
{
    class Optimizer
    {
    public:

        typedef ResidualBlockFactory::Ptr ResidualBlockPtr;
        typedef ParameterBlockFactory::Ptr ParameterBlockPtr;

        class Options
        {
        public:
            explicit Options(std::string  _trust_region_strategy = "dogleg",
                            std::string  _linear_solver_type = "dense_schur",
                            int _max_iteration = 10,
                            int _num_threads = 4,
                            double _max_solver_time = 1e32,
                            bool _verbose = false);
            [[nodiscard]] ceres::Solver::Options ToCeres() const;

            std::string trust_region_strategy;
            std::string linear_solver_type;

            int max_iteration;
            int num_threads;

            double max_solver_time;

            bool verbose;
        };

        static void Construct(const Options& options,
                              const std::vector<ParameterBlockPtr>& parameter_blocks,
                              const std::vector<ResidualBlockPtr>&  residual_blocks);
    };//end of Optimizer
}//end of SuperVIO

#endif //SUPER_VIO_OPTIMIZER_H
