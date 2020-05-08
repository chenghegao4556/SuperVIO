//
// Created by chenghe on 3/28/20.
//

#ifndef SUPER_VIO_IMU_FACTOR_H
#define SUPER_VIO_IMU_FACTOR_H

#include <ceres/ceres.h>
#include <imu/imu_states_measurements.h>
#include <imu/imu_pre_integrator.h>
namespace SuperVIO::Optimization
{
    class IMUFactor : public ceres::SizedCostFunction<15, 7, 9, 7, 9>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef IMU::PreIntegrator::StateJacobianOrder StateOrder;


        explicit IMUFactor(const IMU::IMUPreIntegrationMeasurement& pre_integration_measurement,
                           Vector3 gravity_vector);

        bool Evaluate(double const *const *parameters,
                      double *residuals,
                      double **jacobians) const override;

    private:
        double sum_dt_;
        Vector3 gravity_vector_;
        Quaternion delta_q_;
        Vector3 delta_v_;
        Vector3 delta_p_;
        Vector3 linearized_ba_;
        Vector3 linearized_bg_;
        Matrix15 jacobian_;
        Matrix15 sqrt_information_;
    };//end of IMUFactor
}//end of IMUFactor

#endif //SUPER_VIO_IMU_FACTOR_H
