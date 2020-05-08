//
// Created by chenghe on 3/31/20.
//
#include <optimization/residual_blocks/base_residual_block.h>

#include <utility>
namespace SuperVIO::Optimization
{

    typedef BaseResidualBlock::ParameterBlockPtr ParameterBlockPtr;

    ///////////////////////////////////////////////////////////////////////////////////
    BaseResidualBlock::
    BaseResidualBlock(ceres::CostFunction* _cost_function,
                      ceres::LossFunction* _loss_function,
                      std::vector<ParameterBlockPtr>  _parameter_blocks):
                  cost_function_(_cost_function),
                  loss_function_(_loss_function),
                  parameter_blocks_(std::move(_parameter_blocks)),
                  raw_jacobians_(nullptr)
    {
        for(const auto& parameter_block: parameter_blocks_)
        {
            data_.push_back(parameter_block->GetData());
        }
    }

    ///////////////////////////////////////////////////////////////////////////////////
    BaseResidualBlock::
    ~BaseResidualBlock()
    {
        delete []raw_jacobians_;
    }

    ///////////////////////////////////////////////////////////////////////////////////
    void BaseResidualBlock::
    Evaluate()
    {
        residuals_.resize(cost_function_->num_residuals());
        std::vector<int> block_sizes = cost_function_->parameter_block_sizes();
        raw_jacobians_ = new double *[block_sizes.size()];
        jacobians_.resize(block_sizes.size());

        for (size_t i = 0; i < block_sizes.size(); ++i)
        {
            jacobians_[i].resize(cost_function_->num_residuals(), block_sizes[i]);
            raw_jacobians_[i] = jacobians_[i].data();
        }

        //! compute jacobian
        cost_function_->Evaluate(data_.data(), residuals_.data(), raw_jacobians_);
        //! apply loss function
        if (loss_function_)
        {
            double residual_scaling, alpha_sq_norm;
            double sq_norm, rho[3];
            sq_norm = residuals_.squaredNorm();
            loss_function_->Evaluate(sq_norm, rho);

            double sqrt_rho1 = sqrt(rho[1]);
            if ((sq_norm == 0.0) || (rho[2] <= 0.0))
            {
                residual_scaling = sqrt_rho1;
                alpha_sq_norm = 0.0;
            }
            else
            {
                const double D = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];
                const double alpha = 1.0 - sqrt(D);
                residual_scaling = sqrt_rho1 / (1 - alpha);
                alpha_sq_norm = alpha / sq_norm;
            }

            for (size_t i = 0; i < parameter_blocks_.size(); i++)
            {
                jacobians_[i] = sqrt_rho1 * (jacobians_[i] -
                        alpha_sq_norm * residuals_ * (residuals_.transpose() * jacobians_[i]));
            }

            residuals_ *= residual_scaling;
        }
    }

    ///////////////////////////////////////////////////////////////////////////////////
    ceres::CostFunction* BaseResidualBlock::
    GetCostFunction()
    {
        return cost_function_;
    }

    ///////////////////////////////////////////////////////////////////////////////////
    ceres::LossFunction* BaseResidualBlock::
    GetLostFunction()
    {
        return loss_function_;
    }

    ///////////////////////////////////////////////////////////////////////////////////
    std::vector<double*>& BaseResidualBlock::
    GetData()
    {
        return data_;
    }

    ///////////////////////////////////////////////////////////////////////////////////
    const std::vector<ParameterBlockPtr>& BaseResidualBlock::
    GetParameterBlock()
    {
        return parameter_blocks_;
    }

    ///////////////////////////////////////////////////////////////////////////////////
    const VectorX& BaseResidualBlock::
    GetResiduals() const
    {
        return residuals_;
    }

    ///////////////////////////////////////////////////////////////////////////////////
    const std::vector<DynamicMatrix, Eigen::aligned_allocator<DynamicMatrix>>& BaseResidualBlock::
    GetJacobians() const
    {
        return jacobians_;
    }
}//end of SuperVIO