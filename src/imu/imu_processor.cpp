//
// Created by chenghe on 3/5/20.
//

#include <imu/imu_processor.h>
#include <utility/eigen_base.h>
namespace SuperVIO::IMU
{
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    IMUStateAndMeasurements IMUProcessor::
    Propagate(const IMUState& last_state,
              const Vector3& acceleration_0,
              const Vector3& angular_velocity_0,
              const Vector3& gravity_vector,
              const Matrix18& imu_noise_matrix,
              const IMURawMeasurements& imu_raw_measurements)
    {
        return Propagate(std::make_pair(
                         last_state,
                         IMUPreIntegrationMeasurement(acceleration_0,
                                                      angular_velocity_0,
                                                      last_state.linearized_ba,
                                                      last_state.linearized_bg)),
                         acceleration_0,
                         angular_velocity_0,
                         gravity_vector,
                         imu_noise_matrix,
                         imu_raw_measurements);
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////
    IMUStateAndMeasurements IMUProcessor::
    Propagate(const IMUStateAndMeasurements& last_state_and_measurements,
              const Vector3& acceleration_0,
              const Vector3& angular_velocity_0,
              const Vector3& gravity_vector,
              const Matrix18& imu_noise_matrix,
              const IMURawMeasurements& imu_raw_measurements)
    {
        auto new_state = Integrator::ComputeAll(last_state_and_measurements.first,
                                                acceleration_0,
                                                angular_velocity_0,
                                                gravity_vector,
                                                imu_raw_measurements);

        auto new_pre_integration = PreIntegrator::ComputeAll(last_state_and_measurements.second,
                                                             acceleration_0,
                                                             angular_velocity_0,
                                                             imu_noise_matrix,
                                                             imu_raw_measurements);

        return std::make_pair(new_state, new_pre_integration);
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////
    IMUPreIntegrationMeasurement IMUProcessor::
    Repropagate(const Vector3& new_linearized_ba,
                const Vector3& new_linearized_bg,
                const Vector3& acceleration_0,
                const Vector3& angular_velocity_0,
                const Matrix18& imu_noise_matrix,
                const IMURawMeasurements& imu_raw_measurements)
    {
        auto pre_integration = PreIntegrator::ComputeAll(IMUPreIntegrationMeasurement(
                                                                                      acceleration_0,
                                                                                      angular_velocity_0,
                                                                                      new_linearized_ba,
                                                                                      new_linearized_bg),
                                                         acceleration_0,
                                                         angular_velocity_0,
                                                         imu_noise_matrix,
                                                         imu_raw_measurements);

        return pre_integration;
    }

}//end of SuperVIO