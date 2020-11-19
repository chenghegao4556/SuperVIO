//
// Created by chenghe on 3/30/20.
//

#ifndef SUPER_VIO_MARGINALIZER_H
#define SUPER_VIO_MARGINALIZER_H

#include <optimization/residual_blocks/residual_block_factory.h>
#include <optimization/parameter_blocks/parameter_block_factory.h>
#include <utility/matrix_operation.h>
namespace SuperVIO::Optimization
{
    class Marginalizer
    {
    public:
        typedef ResidualBlockFactory::Ptr ResidualBlockPtr;
        typedef ParameterBlockFactory::Ptr ParameterBlockPtr;

        class OrderedParameters
        {
        public:
            OrderedParameters();

            size_t drop_landmark_size;
            size_t drop_pose_size;
            size_t drop_jacobian_size;
            size_t keep_jacobian_size;
            size_t total_jacobian_size;
            std::vector<ParameterBlockPtr> drop_landmarks;
            std::vector<ParameterBlockPtr> drop_poses;//! drop poses and speed biases;
            std::vector<ParameterBlockPtr> keep_poses;//! keep poses and speed biases
        };

        static MarginalizationInformation
        Construct(const std::vector<ParameterBlockPtr>& parameter_blocks,
                  const std::vector<ResidualBlockPtr>&  residual_blocks,
                  const std::set<size_t>& drop_set,
                  const size_t& num_threads = 4);

    protected:
        class ConstructHessianThread
        {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
            void Evaluate();

            size_t jacobian_size;
            std::vector<ResidualBlockPtr> residual_per_thread;
            MatrixX information_matrix;
            VectorX information_vector;
        };
        static OrderedParameters
        SetOrder(const std::vector<ParameterBlockPtr>& parameter_blocks,
                 const std::set<size_t>& drop_set);

        static std::pair<MatrixX, VectorX>
        ConstructHessian(const OrderedParameters& ordered_parameters,
                         const std::vector<ResidualBlockPtr>& residual_blocks,
                         const size_t& num_threads);

        static MarginalizationInformation
        CreatMarginalizationInformation(const MatrixX& linearized_jacobians,
                                        const VectorX& linearized_residuals,
                                        const OrderedParameters& ordered_parameters);

    };

}//end of SuperVIO
#endif //SUPER_VIO_MARGINALIZER_H
