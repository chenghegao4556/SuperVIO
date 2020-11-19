//
// Created by chenghe on 3/31/20.
//

#ifndef SUPER_VIO_COST_FUNCTION_FACTORY_H
#define SUPER_VIO_COST_FUNCTION_FACTORY_H

#include <optimization/factors/imu_factor.h>
#include <optimization/factors/inverse_depth_factor.h>
#include <optimization/factors/marginalization_factor.h>
#include <optimization/factors/relative_pose_factor.h>
namespace SuperVIO::Optimization
{
    class CostFunctionFactory
    {
    public:

        static ceres::CostFunction*
        CreatReprojectionCostFunction(const Matrix3& intrinsic,
                                      const Quaternion& r_i_c,
                                      const Vector3& t_i_c,
                                      const Vector2& measurement_i,
                                      const Vector2& measurement_j,
                                      const double& sigma_x,
                                      const double& sigma_y);

        static ceres::CostFunction*
        CreatPreIntegrationCostFunction(const IMU::IMUPreIntegrationMeasurement& pre_integration_measurement,
                                        const Vector3& gravity_vector);

        static ceres::CostFunction*
        CreatMarginalizationCostFunction(const MarginalizationInformation& marginalization_information);

        static ceres::CostFunction*
        CreatRelativePoseFactor(const Quaternion& q_ij, const Vector3& p_ij, const double& position_var,
                                const double& rotation_var);


    };//end of CostFunctionFactory
}//end of SuperVIO

#endif //SUPER_VIO_COST_FUNCTION_FACTORY_H
