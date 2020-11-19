//
// Created by chenghe on 4/2/20.
//
#include <optimization/residual_blocks/residual_block_factory.h>
namespace SuperVIO::Optimization
{
    typedef ResidualBlockFactory::Ptr Ptr;

    ///////////////////////////////////////////////////////////////////////////////////
    Ptr ResidualBlockFactory::
    CreatPreIntegration(const IMU::IMUPreIntegrationMeasurement& pre_integration_measurement,
                        const Vector3& gravity_vector,
                        const std::vector<ParameterBlockPtr>& parameter_blocks)
    {
        return PreIntegrationResidualBlock::Creat(pre_integration_measurement,
                                                  gravity_vector,
                                                  parameter_blocks);
    }

    ///////////////////////////////////////////////////////////////////////////////////
    Ptr ResidualBlockFactory::
    CreatReprojection(const Matrix3& intrinsic,
                      const Quaternion& r_i_c,
                      const Vector3& t_i_c,
                      const Vector2& measurement_i,
                      const Vector2& measurement_j,
                      const double& sigma_x,
                      const double& sigma_y,
                      const std::vector<ParameterBlockPtr>& parameter_blocks)
    {
        return ReprojectionResidualBlock::Creat(intrinsic,
                                                r_i_c,
                                                t_i_c,
                                                measurement_i,
                                                measurement_j,
                                                sigma_x,
                                                sigma_y,
                                                parameter_blocks);
    }

    ///////////////////////////////////////////////////////////////////////////////////
    Ptr ResidualBlockFactory::
    CreatMarginalization(const MarginalizationInformation& marginalization_information)
    {
        return MarginalizationResidualBlock::Creat(marginalization_information);
    }

    ///////////////////////////////////////////////////////////////////////////////////
    Ptr ResidualBlockFactory::
    CreatRelativePose(const Quaternion& q_ij, const Vector3& p_ij,
                      const double& position_var, const double& rotation_var,
                      const std::vector<ParameterBlockPtr>& parameter_blocks)
    {
        return RelativePoseResidualBlock::Creat(q_ij, p_ij, position_var, rotation_var, parameter_blocks);
    }
}//end of SuperVIO