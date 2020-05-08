//
// Created by chenghe on 3/5/20.
//

#include <imu/imu_integrator.h>

namespace SuperVIO::IMU
{


    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    IMUState Integrator::
    ComputeAll(const IMUState& old_state,
               const Vector3& acceleration_0,
               const Vector3& angular_velocity_0,
               const Vector3& gravity_vector,
               const IMURawMeasurements& imu_raw_measurements)
    {
        if(imu_raw_measurements.empty())
        {
            ROS_ERROR_STREAM("Buffer is empty");
            throw std::runtime_error("Empty Buffer");
        }

        auto new_state = old_state;
        Vector3 acc_0 = acceleration_0;
        Vector3 gyr_0 = angular_velocity_0;

        for(const auto& imu_raw_measurement: imu_raw_measurements)
        {
            new_state = ComputeOnce(new_state,
                                    acc_0,
                                    gyr_0,
                                    gravity_vector,
                                    imu_raw_measurement);

            acc_0 = imu_raw_measurement.acceleration;
            gyr_0 = imu_raw_measurement.angular_velocity;
        }

        return new_state;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    IMUState Integrator::
    ComputeOnce(const IMUState& old_state,
                const Vector3& acceleration_0,
                const Vector3& angular_velocity_0,
                const Vector3& gravity_vector,
                const IMURawMeasurement& imu_raw_measurement)
    {

        const auto& dt = imu_raw_measurement.dt;
        const auto& acceleration_1     = imu_raw_measurement.acceleration;
        const auto& angular_velocity_1 = imu_raw_measurement.angular_velocity;

        const auto& linearized_ba = old_state.linearized_ba;
        const auto& linearized_bg = old_state.linearized_bg;

        const Vector3 average_angular_velocity = 0.5 * (angular_velocity_0 + angular_velocity_1) - linearized_bg;
        const Quaternion new_rotation = old_state.rotation * Utility::EigenBase::DeltaQ(average_angular_velocity * dt);

        const Vector3 world_acceleration_0 = old_state.rotation * (acceleration_0 - linearized_ba) - gravity_vector;
        const Vector3 world_acceleration_1 = new_rotation * (acceleration_1 - linearized_ba) - gravity_vector;
        const Vector3 average_world_acceleration = 0.5 * (world_acceleration_0 + world_acceleration_1);

        const Vector3 new_position = old_state.position + dt * old_state.velocity + 0.5 * dt * dt * average_world_acceleration;

        const Vector3 new_velocity = old_state.velocity + dt * average_world_acceleration;

        return IMUState(new_rotation,
                        new_position,
                        new_velocity,
                        linearized_ba,
                        linearized_bg);
    }

}//end of SuperVIO