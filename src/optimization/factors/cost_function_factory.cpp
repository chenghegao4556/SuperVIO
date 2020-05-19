//
// Created by chenghe on 3/31/20.
//

#include <optimization/factors/cost_function_factory.h>

#include <utility>

namespace SuperVIO::Optimization
{

    ///////////////////////////////////////////////////////////////////////////////////////////////////
    ceres::CostFunction* CostFunctionFactory::
    CreatReprojectionCostFunction(const Matrix3& intrinsic,
                                  const Quaternion& r_i_c,
                                  const Vector3& t_i_c,
                                  const Vector2& measurement_i,
                                  const Vector2& measurement_j,
                                  const double& sigma_x,
                                  const double& sigma_y)
    {
        auto* cost_function = new InverseDepthFactor(intrinsic,
                                                     r_i_c,
                                                     t_i_c,
                                                     measurement_i,
                                                     measurement_j,
                                                     sigma_x,
                                                     sigma_y);

        return cost_function;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////
    ceres::CostFunction* CostFunctionFactory::
    CreatPreIntegrationCostFunction(const IMU::IMUPreIntegrationMeasurement& pre_integration_measurement,
                                    const Vector3& gravity_vector)
    {
        auto* cost_function = new IMUFactor(pre_integration_measurement,
                                            gravity_vector);

        return cost_function;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////
    ceres::CostFunction* CostFunctionFactory::
    CreatMarginalizationCostFunction(const MarginalizationInformation& marginalization_information)
    {
        auto* cost_function = new MarginalizationFactor(marginalization_information);

        return cost_function;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////
    ceres::CostFunction* CostFunctionFactory::
    CreatRelativePoseFactor(const Quaternion& q_ij, const Vector3& p_ij, const double& position_var,
                            const double& rotation_var)
    {
        auto* cost_function = new RelativePoseFactor(q_ij, p_ij, position_var, rotation_var);

        return cost_function;
    }

}//end of SuperVIO