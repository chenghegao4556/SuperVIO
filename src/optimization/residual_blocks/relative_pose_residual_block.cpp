//
// Created by chenghe on 5/17/20.
//
#include <optimization/residual_blocks/relative_pose_residual_block.h>
namespace SuperVIO::Optimization
{
    typedef RelativePoseResidualBlock::Ptr Ptr;
    typedef RelativePoseResidualBlock::Type Type;

    /////////////////////////////////////////////////////////////////////////////////////////////////
    Ptr RelativePoseResidualBlock::
    Creat(const Quaternion& q_ij, const Vector3& p_ij,
          const double& position_var, const double& rotation_var,
          const std::vector<ParameterBlockPtr>& parameter_blocks)
    {
        Ptr p(new RelativePoseResidualBlock(q_ij, p_ij, position_var, rotation_var, parameter_blocks));

        return p;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    Type RelativePoseResidualBlock::
    GetType() const
    {
        return Type::RelativePose;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    RelativePoseResidualBlock::
    RelativePoseResidualBlock(const Quaternion& q_ij, const Vector3& p_ij,
                              const double& position_var, const double& rotation_var,
                              const std::vector<ParameterBlockPtr>& parameter_blocks):
            BaseResidualBlock(CostFunctionFactory::CreatRelativePoseFactor(q_ij, p_ij, position_var, rotation_var),
                              new ceres::HuberLoss(0.1),
                              parameter_blocks)
    {

    }
}//end of SuperVIO::Optimization