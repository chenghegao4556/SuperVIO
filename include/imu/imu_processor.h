//
// Created by chenghe on 3/5/20.
//

#ifndef SUPER_VIO_IMU_PROCESSOR_H
#define SUPER_VIO_IMU_PROCESSOR_H

#include <numeric>
#include <imu/imu_integrator.h>
#include <imu/imu_pre_integrator.h>
namespace SuperVIO::IMU
{
    class IMUProcessor
    {
    public:
        //! used for normally propagate
        static IMUStateAndMeasurements
        Propagate(const IMUState& last_state,
                  const Vector3& acceleration_0,
                  const Vector3& angular_velocity_0,
                  const Vector3& gravity_vector,
                  const Matrix18& imu_noise_matrix,
                  const IMURawMeasurements& imu_raw_measurements);

        ///! propagate all measurements(used for second new frame is not key frame)
        static IMUStateAndMeasurements
        Propagate(const IMUStateAndMeasurements& last_state_and_measurements,
                  const Vector3& acceleration_0,
                  const Vector3& angular_velocity_0,
                  const Vector3& gravity_vector,
                  const Matrix18& imu_noise_matrix,
                  const IMURawMeasurements& imu_raw_measurements);

        ///! repropagate all measurements when biases are changed
        static IMUPreIntegrationMeasurement
        Repropagate(const Vector3& new_linearized_ba,
                    const Vector3& new_linearized_bg,
                    const Vector3& acceleration_0,
                    const Vector3& angular_velocity_0,
                    const Matrix18& imu_noise_matrix,
                    const IMURawMeasurements& imu_raw_measurements);

    };//end of IMUProcessor
}//SuperVIO::IMU

#endif //SRC_IMU_PROCESSOR_H
