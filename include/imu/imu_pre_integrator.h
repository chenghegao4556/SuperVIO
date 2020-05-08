//
// Created by chenghe on 3/6/20.
//

#ifndef SRC_IMU_PRE_INTEGRATOR_H
#define SRC_IMU_PRE_INTEGRATOR_H

#include <utility/eigen_type.h>
#include <utility/eigen_base.h>
#include <imu/imu_states_measurements.h>
namespace SuperVIO::IMU
{

    class PreIntegrator
    {
    public:
        enum StateJacobianOrder
        {
            O_P = 0,
            O_R = 3,
            O_V = 6,
            O_BA = 9,
            O_BG = 12
        };
        enum NoiseJacobianOrder
        {
            O_OA = 0,
            O_OG = 3,
            O_NA = 6,
            O_NG = 9,
            O_OBA = 12,
            O_OBG = 15
        };


        static IMUPreIntegrationMeasurement
        ComputeAll(const IMUPreIntegrationMeasurement& last_pre_integration,
                   const Vector3& acceleration_0,
                   const Vector3& angular_velocity_0,
                   const Matrix18& imu_noise_matrix,
                   const IMURawMeasurements& imu_raw_measurements);

        static IMUPreIntegrationMeasurement
        ComputeOnce(const IMUPreIntegrationMeasurement& last_pre_integration,
                    const Vector3& acceleration_0,
                    const Vector3& angular_velocity_0,
                    const Matrix18& imu_noise_matrix,
                    const IMURawMeasurement& imu_raw_measurement);

    };//end of PreIntegrator
}//end of SuperVIO::IMU

#endif //SRC_IMU_PRE_INTEGRATOR_H
