//
// Created by chenghe on 3/5/20.
//

#ifndef SUPER_VIO_IMU_INTEGRATOR_H
#define SUPER_VIO_IMU_INTEGRATOR_H

#include <utility/eigen_type.h>
#include <utility/eigen_base.h>
#include <imu/imu_states_measurements.h>
namespace SuperVIO::IMU
{

    class Integrator
    {
    public:
        /**
         * @brief integrate all measurements
         */
        static IMUState
        ComputeAll(const IMUState& old_state,
                   const Vector3& acceleration_0,
                   const Vector3& angular_velocity_0,
                   const Vector3& gravity_vector,
                   const IMURawMeasurements& imu_raw_measurements);

        /**
         * @brief integrate once
         */
        static IMUState
        ComputeOnce(const IMUState& old_state,
                    const Vector3& acceleration_0,
                    const Vector3& angular_velocity_0,
                    const Vector3& gravity_vector,
                    const IMURawMeasurement& imu_raw_measurement);
    };//end of IMUIntegrator
}//end of SuperVIO::IMU

#endif //SUPER_VIO_IMU_INTEGRATOR_H
