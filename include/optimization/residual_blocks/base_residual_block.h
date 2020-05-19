//
// Created by chenghe on 3/31/20.
//

#ifndef SUPER_VIO_RESIDUAL_BLOCK_H
#define SUPER_VIO_RESIDUAL_BLOCK_H

#include <optimization/factors/cost_function_factory.h>
#include <optimization/parameter_blocks/parameter_block_factory.h>
namespace SuperVIO::Optimization
{
    class BaseResidualBlock
    {
    public:
        typedef ParameterBlockFactory::Ptr ParameterBlockPtr;
        typedef std::shared_ptr<BaseResidualBlock> Ptr;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        enum class Type
        {
            PreIntegration,
            Reprojection,
            Marginalization,
            RelativePose,
        };

        virtual ~BaseResidualBlock();
        [[nodiscard]] virtual Type GetType() const = 0;

        void Evaluate();

        ceres::CostFunction* GetCostFunction();
        ceres::LossFunction* GetLostFunction();
        const std::vector<ParameterBlockPtr>& GetParameterBlock();
        std::vector<double*>& GetData();

        [[nodiscard]] const VectorX& GetResiduals() const;

        [[nodiscard]] const std::vector<DynamicMatrix, Eigen::aligned_allocator<DynamicMatrix>>&
        GetJacobians() const;

    protected:
        explicit BaseResidualBlock(ceres::CostFunction* _cost_function,
                                   ceres::LossFunction* _loss_function,
                                   std::vector<ParameterBlockPtr>   _parameter_blocks);

    private:
        ceres::CostFunction* cost_function_;
        ceres::LossFunction* loss_function_;
        std::vector<double*> data_;
        std::vector<ParameterBlockPtr> parameter_blocks_;
        double **raw_jacobians_;
        VectorX residuals_;
        std::vector<DynamicMatrix, Eigen::aligned_allocator<DynamicMatrix>> jacobians_;
    };
}//end of SuperVIO

#endif //SUPER_VIO_RESIDUAL_BLOCK_H
