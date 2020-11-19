//
// Created by chenghe on 4/2/20.
//

#include <optimization/residual_blocks/reprojection_residual_block.h>
namespace SuperVIO::Optimization
{
    typedef ReprojectionResidualBlock::Ptr Ptr;
    typedef ReprojectionResidualBlock::Type Type;

    ///////////////////////////////////////////////////////////////////////////////////
    Ptr ReprojectionResidualBlock::
    Creat(const Matrix3& intrinsic,
          const Quaternion& r_i_c,
          const Vector3& t_i_c,
          const Vector2& measurement_i,
          const Vector2& measurement_j,
          const double& sigma_x,
          const double& sigma_y,
          const std::vector<ParameterBlockPtr>& parameter_blocks)
    {
        Ptr p(new ReprojectionResidualBlock(intrinsic,
                                            r_i_c,
                                            t_i_c,
                                            measurement_i,
                                            measurement_j,
                                            sigma_x,
                                            sigma_y,
                                            parameter_blocks));
        return p;
    }

    ///////////////////////////////////////////////////////////////////////////////////
    Type ReprojectionResidualBlock::
    GetType() const
    {
        return Type::Reprojection;
    }

    ///////////////////////////////////////////////////////////////////////////////////
    ReprojectionResidualBlock::
    ReprojectionResidualBlock(const Matrix3& intrinsic,
                              const Quaternion& r_i_c,
                              const Vector3& t_i_c,
                              const Vector2& measurement_i,
                              const Vector2& measurement_j,
                              const double& sigma_x,
                              const double& sigma_y,
                              const std::vector<ParameterBlockPtr>& parameter_blocks):
            BaseResidualBlock(CostFunctionFactory::CreatReprojectionCostFunction(intrinsic,
                                                                                 r_i_c,
                                                                                 t_i_c,
                                                                                 measurement_i,
                                                                                 measurement_j,
                                                                                 sigma_x,
                                                                                 sigma_y),
                              new ceres::CauchyLoss(1.0),
                              parameter_blocks)
    {
        ROS_ASSERT(parameter_blocks.size() == 3);
        ROS_ASSERT(parameter_blocks[0]->GetType() == BaseParametersBlock::Type::Pose);
        ROS_ASSERT(parameter_blocks[1]->GetType() == BaseParametersBlock::Type::Pose);
        ROS_ASSERT(parameter_blocks[2]->GetType() == BaseParametersBlock::Type::InverseDepth);
    }
}//end of SuperVIO
