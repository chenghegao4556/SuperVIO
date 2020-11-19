//
// Created by chenghe on 3/6/20.
//
#include <imu/imu_pre_integrator.h>

namespace SuperVIO::IMU
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    IMUPreIntegrationMeasurement  PreIntegrator::
    ComputeAll(const IMUPreIntegrationMeasurement& last_pre_integration,
               const Vector3& acceleration_0,
               const Vector3& angular_velocity_0,
               const Matrix18& imu_noise_matrix,
               const IMURawMeasurements& imu_raw_measurements)
    {
        if(imu_raw_measurements.empty())
        {
            ROS_ERROR_STREAM("Buffer is empty");
            throw std::runtime_error("Empty Buffer");
        }
        auto new_pre_integration = last_pre_integration;
        Vector3 acc_0 = acceleration_0;
        Vector3 gyr_0 = angular_velocity_0;
        for(const auto& imu_raw_measurement: imu_raw_measurements)
        {
            new_pre_integration = ComputeOnce(new_pre_integration,
                                              acc_0,
                                              gyr_0,
                                              imu_noise_matrix,
                                              imu_raw_measurement);

            acc_0 = imu_raw_measurement.acceleration;
            gyr_0 = imu_raw_measurement.angular_velocity;
        }

        return new_pre_integration;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    IMUPreIntegrationMeasurement  PreIntegrator::
    ComputeOnce(const IMUPreIntegrationMeasurement& last_pre_integration,
                const Vector3& acceleration_0,
                const Vector3& angular_velocity_0,
                const Matrix18& imu_noise_matrix,
                const IMURawMeasurement& imu_raw_measurement)
    {
        const double& dt = imu_raw_measurement.dt;
        const Vector3& acceleration_1     = imu_raw_measurement.acceleration;
        const Vector3& angular_velocity_1 = imu_raw_measurement.angular_velocity;

        const Vector3& linearized_ba = last_pre_integration.linearized_ba;
        const Vector3& linearized_bg = last_pre_integration.linearized_bg;

        const Quaternion& delta_q = last_pre_integration.delta_q;
        const Vector3& delta_v = last_pre_integration.delta_v;
        const Vector3& delta_p = last_pre_integration.delta_p;

        const Matrix15& covariance = last_pre_integration.covariance;
        const Matrix15& jacobian   = last_pre_integration.jacobian;

        const double new_sum_dt = last_pre_integration.sum_dt + dt;

        const Vector3 average_angular_velocity = 0.5 * (angular_velocity_0 + angular_velocity_1) - linearized_bg;
        //!new delta rotation
        const Quaternion new_delta_q = delta_q * Utility::EigenBase::DeltaQ(average_angular_velocity * dt);

        const Vector3 un_acceleration_0 = delta_q * (acceleration_0 - linearized_ba);
        const Vector3 un_acceleration_1 = new_delta_q * (acceleration_1 - linearized_ba);
        const Vector3 average_acceleration = 0.5 * (un_acceleration_0 + un_acceleration_1);

        //!new delta position
        const Vector3 new_delta_p = delta_p + dt * delta_v + 0.5 * dt * dt * average_acceleration;
        //! new delta velocity
        const Vector3 new_delta_v = delta_v + dt * average_acceleration;

        Matrix3 skew_gyr = Utility::EigenBase::SkewSymmetric(average_angular_velocity);
        Matrix3 skew_acc_0 = Utility::EigenBase::SkewSymmetric(acceleration_0 - linearized_ba);
        Matrix3 skew_acc_1 = Utility::EigenBase::SkewSymmetric(acceleration_1 - linearized_ba);

        Matrix15 F = Matrix15::Zero();
        //! partial delta_p partial delta_p
        F.block<3, 3>(StateJacobianOrder::O_P, StateJacobianOrder::O_P) = Matrix3::Identity();
        //! partial delta_p partial delta_q
        F.block<3, 3>(StateJacobianOrder::O_P, StateJacobianOrder::O_R) =
                -0.25 *  delta_q.toRotationMatrix() * skew_acc_0 * dt * dt +
                -0.25 *  new_delta_q.toRotationMatrix() * skew_acc_1 *
                (Matrix3::Identity() - skew_gyr * dt) * dt * dt;
        //! partial delta_p partial delta_v
        F.block<3, 3>(StateJacobianOrder::O_P, StateJacobianOrder::O_V) = Matrix3::Identity() * dt;
        //! partial delta_p partial delta_ba
        F.block<3, 3>(StateJacobianOrder::O_P, StateJacobianOrder::O_BA) =
                -0.25 * (delta_q.toRotationMatrix() + new_delta_q.toRotationMatrix()) * dt * dt;
        //! partial delta_p partial delta_bg
        F.block<3, 3>(StateJacobianOrder::O_P, StateJacobianOrder::O_BG) =
                 0.25 * new_delta_q.toRotationMatrix() * skew_acc_1 * dt * dt * dt;

        //! partial delta_q partial delta_q
        F.block<3, 3>(StateJacobianOrder::O_R,  StateJacobianOrder::O_R) = Matrix3::Identity() - skew_gyr * dt;
        //! partial delta_q partial delta_bg
        F.block<3, 3>(StateJacobianOrder::O_R,  StateJacobianOrder::O_BG) = -1.0 * Matrix3::Identity() * dt;

        //! partial delta_v partial delta_r
        F.block<3, 3>(StateJacobianOrder::O_V,  StateJacobianOrder::O_R) =
                -0.5 * delta_q.toRotationMatrix() * skew_acc_0 * dt +
                -0.5 * new_delta_q.toRotationMatrix() * skew_acc_1 *
                (Matrix3::Identity() - skew_gyr * dt) * dt;
        //! partial delta_v partial delta_v
        F.block<3, 3>(StateJacobianOrder::O_V,  StateJacobianOrder::O_V) = Matrix3::Identity();
        //! partial delta_v partial delta_ba
        F.block<3, 3>(StateJacobianOrder::O_V,  StateJacobianOrder::O_BA) =
                -0.5 * (delta_q.toRotationMatrix() + new_delta_q.toRotationMatrix()) * dt;
        //! partial delta_v partial delta_bg
        F.block<3, 3>(StateJacobianOrder::O_V,  StateJacobianOrder::O_BG) =
                0.5 * new_delta_q.toRotationMatrix() * skew_acc_1 * dt * dt;

        //! partial delta_ba partial delta_ba
        F.block<3, 3>(StateJacobianOrder::O_BA, StateJacobianOrder::O_BA) = Matrix3::Identity();
        //! partial delta_bg partial delta_bg
        F.block<3, 3>(StateJacobianOrder::O_BG, StateJacobianOrder::O_BG) = Matrix3::Identity();

        MatrixX V = MatrixX::Zero(15,18);

        //! partial delta_p partial old acc noise
        V.block<3, 3>(StateJacobianOrder::O_P, NoiseJacobianOrder::O_OA) =
                0.25 * delta_q.toRotationMatrix() * dt * dt;
        //! partial delta_p partial old gyr noise
        V.block<3, 3>(StateJacobianOrder::O_P, NoiseJacobianOrder::O_OG) =
                -0.125 * new_delta_q.toRotationMatrix() * skew_acc_1 * dt * dt * dt;
        //! partial delta_p partial new acc noise
        V.block<3, 3>(StateJacobianOrder::O_P, NoiseJacobianOrder::O_NA) =
                  0.25 * new_delta_q.toRotationMatrix() * dt * dt;
        //! partial delta_p partial new acc noise
        V.block<3, 3>(StateJacobianOrder::O_P, NoiseJacobianOrder::O_NG) =
                V.block<3, 3>(StateJacobianOrder::O_P, NoiseJacobianOrder::O_OG);

        //! partial delta_q partial old gyr noise
        V.block<3, 3>(StateJacobianOrder::O_R, NoiseJacobianOrder::O_OG) = 0.5 * Matrix3::Identity() * dt;
        //! partial delta_q partial new gyr noise
        V.block<3, 3>(StateJacobianOrder::O_R, NoiseJacobianOrder::O_NG) = 0.5 * Matrix3::Identity() * dt;

        //! partial delta_v partial old acc noise
        V.block<3, 3>(StateJacobianOrder::O_V, NoiseJacobianOrder::O_OA) = 0.5 * delta_q.toRotationMatrix() * dt;
        //! partial delta_v partial old gyr noise
        V.block<3, 3>(StateJacobianOrder::O_V, NoiseJacobianOrder::O_OG) =
                -0.25 * new_delta_q.toRotationMatrix() * skew_acc_1 * dt * dt;
        //! partial delta_v partial new acc noise
        V.block<3, 3>(StateJacobianOrder::O_V, NoiseJacobianOrder::O_NA) =
                  0.5 * new_delta_q.toRotationMatrix() * dt;
        //! partial delta_v partial new gyr noise
        V.block<3, 3>(StateJacobianOrder::O_V, NoiseJacobianOrder::O_NG) =
                V.block<3, 3>(StateJacobianOrder::O_V, NoiseJacobianOrder::O_OG);

        //! partial delta_ba partial old gyr noise
        V.block<3, 3>(StateJacobianOrder::O_BA,  NoiseJacobianOrder::O_OBA) = Matrix3::Identity() * dt;
        //! partial delta_bg partial old gyr noise
        V.block<3, 3>(StateJacobianOrder::O_BG,  NoiseJacobianOrder::O_OBG) = Matrix3::Identity() * dt;

        const auto new_jacobian = F * jacobian;
        const auto new_covariance = F * covariance * F.transpose() + V * imu_noise_matrix * V.transpose();

        return IMUPreIntegrationMeasurement(last_pre_integration.acceleration_0,
                                            last_pre_integration.angular_velocity_0,
                                            linearized_ba,
                                            linearized_bg,
                                            new_sum_dt,
                                            new_jacobian,
                                            new_covariance,
                                            new_delta_q,
                                            new_delta_p,
                                            new_delta_v);
    }

}//end of SuperVIO::IMU
