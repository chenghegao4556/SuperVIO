//
// Created by chenghe on 4/2/20.
//

#ifndef SUPER_VIO_RESIDUAL_BLOCK_FACTORY_H
#define SUPER_VIO_RESIDUAL_BLOCK_FACTORY_H

#include <optimization/residual_blocks/pre_integration_residual_block.h>
#include <optimization/residual_blocks/reprojection_residual_block.h>
#include <optimization/residual_blocks/marginalization_residual_block.h>
#include <optimization/residual_blocks/relative_pose_residual_block.h>
namespace SuperVIO::Optimization
{
    class ResidualBlockFactory
    {
    public:
        typedef BaseResidualBlock::Ptr Ptr;
        typedef ParameterBlockFactory::Ptr ParameterBlockPtr;


        static Ptr CreatPreIntegration(const IMU::IMUPreIntegrationMeasurement& pre_integration_measurement,
                                       const Vector3& gravity_vector,
                                       const std::vector<ParameterBlockPtr>& parameter_blocks);

        static Ptr CreatReprojection(const Matrix3& intrinsic,
                                     const Quaternion& r_i_c,
                                     const Vector3& t_i_c,
                                     const Vector2& measurement_i,
                                     const Vector2& measurement_j,
                                     const double& sigma_x,
                                     const double& sigma_y,
                                     const std::vector<ParameterBlockPtr>& parameter_blocks);


        static Ptr CreatMarginalization(const MarginalizationInformation& marginalization_information);

        static Ptr CreatRelativePose(const Quaternion& q_ij, const Vector3& p_ij,
                                     const double& position_var, const double& rotation_var,
                                     const std::vector<ParameterBlockPtr>& parameter_blocks);
    };//end of ResidualBlockFactory
}//end of SuperVIO

#endif //SUPER_VIO_RESIDUAL_BLOCK_FACTORY_H
