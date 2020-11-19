//
// Created by chenghe on 4/4/20.
//
#include <optimization/factors/imu_factor.h>
namespace SuperVIO::Optimization
{
    ///////////////////////////////////////////////////////////////////////////////////////
    IMUFactor::
    IMUFactor(const IMU::IMUPreIntegrationMeasurement& pre_integration_measurement,
              Vector3 gravity_vector):
            sum_dt_(pre_integration_measurement.sum_dt),
            gravity_vector_(std::move(gravity_vector)),
            delta_q_(pre_integration_measurement.delta_q),
            delta_v_(pre_integration_measurement.delta_v),
            delta_p_(pre_integration_measurement.delta_p),
            linearized_ba_(pre_integration_measurement.linearized_ba),
            linearized_bg_(pre_integration_measurement.linearized_bg),
            jacobian_(pre_integration_measurement.jacobian),
            sqrt_information_(Eigen::LLT<Matrix15>(pre_integration_measurement.covariance.inverse()).matrixL().transpose())
    {

    }

    ///////////////////////////////////////////////////////////////////////////////////////
    bool IMUFactor::
    Evaluate(double const *const *parameters,
             double *residuals,
             double **jacobians) const
    {
        const ConstMapVector3 p_i(parameters[0]);
        const ConstMapQuaternion q_i(parameters[0] + 3);
        const ConstMapVector3 v_i(parameters[1]);
        const ConstMapVector3 ba_i(parameters[1] + 3);
        const ConstMapVector3 bg_i(parameters[1] + 6);

        const ConstMapVector3 p_j(parameters[2]);
        const ConstMapQuaternion q_j(parameters[2] + 3);
        const ConstMapVector3 v_j(parameters[3]);
        const ConstMapVector3 ba_j(parameters[3] + 3);
        const ConstMapVector3 bg_j(parameters[3] + 6);

        const Matrix3 dp_dba = jacobian_.block<3, 3>(StateOrder::O_P, StateOrder::O_BA);
        const Matrix3 dp_dbg = jacobian_.block<3, 3>(StateOrder::O_P, StateOrder::O_BG);
        const Matrix3 dq_dbg = jacobian_.block<3, 3>(StateOrder::O_R, StateOrder::O_BG);

        const Matrix3 dv_dba = jacobian_.block<3, 3>(StateOrder::O_V, StateOrder::O_BA);
        const Matrix3 dv_dbg = jacobian_.block<3, 3>(StateOrder::O_V, StateOrder::O_BG);

        const Vector3 dba = ba_i - linearized_ba_;
        const Vector3 dbg = bg_i - linearized_bg_;

        const Quaternion corrected_delta_q = delta_q_ * Utility::EigenBase::DeltaQ(dq_dbg * dbg);
        const Vector3    corrected_delta_v = delta_v_ + dv_dba * dba + dv_dbg * dbg;
        const Vector3    corrected_delta_p = delta_p_ + dp_dba * dba + dp_dbg * dbg;

        MapVector15 residual_vector(residuals);

        residual_vector.block<3, 1>(StateOrder::O_P, 0) =
                q_i.inverse() * (0.5 * gravity_vector_ * sum_dt_ * sum_dt_ + p_j - p_i - v_i * sum_dt_)
                - corrected_delta_p;
        residual_vector.block<3, 1>(StateOrder::O_R, 0) =
                2.0 * (corrected_delta_q.inverse() * (q_i.inverse() * q_j)).vec();
        residual_vector.block<3, 1>(StateOrder::O_V, 0) =
                q_i.inverse() * (gravity_vector_ * sum_dt_ + v_j - v_i) - corrected_delta_v;

        residual_vector.block<3, 1>(StateOrder::O_BA, 0) = ba_j - ba_i;
        residual_vector.block<3, 1>(StateOrder::O_BG, 0) = bg_j - bg_i;

        residual_vector = sqrt_information_ * residual_vector;
        if (jacobians)
        {
            if (jacobians[0])
            {
                MapMatrix15_7 jacobian_pose_i(jacobians[0]);
                jacobian_pose_i.setZero();

                jacobian_pose_i.block<3, 3>(StateOrder::O_P, StateOrder::O_P) =
                        -q_i.inverse().toRotationMatrix();

                jacobian_pose_i.block<3, 3>(StateOrder::O_P, StateOrder::O_R) =
                        Utility::EigenBase::SkewSymmetric(
                                q_i.inverse() *
                                (0.5 * gravity_vector_ * sum_dt_ * sum_dt_ +
                                 p_j - p_i - v_i * sum_dt_));

                jacobian_pose_i.block<3, 3>(StateOrder::O_R, StateOrder::O_R) =
                        -(Utility::EigenBase::Qleft (q_j.inverse() * q_i) *
                          Utility::EigenBase::Qright(corrected_delta_q)).bottomRightCorner<3, 3>();

                jacobian_pose_i.block<3, 3>(StateOrder::O_V, StateOrder::O_R) =
                        Utility::EigenBase::SkewSymmetric(
                                q_i.inverse() * (gravity_vector_ * sum_dt_ + v_j - v_i));

                jacobian_pose_i = sqrt_information_ * jacobian_pose_i;
            }
            if (jacobians[1])
            {
                MapMatrix15_9 jacobian_speed_bias_i(jacobians[1]);
                jacobian_speed_bias_i.setZero();

                jacobian_speed_bias_i.block<3, 3>(StateOrder::O_P, StateOrder::O_V - StateOrder::O_V) =
                        -q_i.inverse().toRotationMatrix() * sum_dt_;
                jacobian_speed_bias_i.block<3, 3>(StateOrder::O_P, StateOrder::O_BA - StateOrder::O_V) =
                        -dp_dba;
                jacobian_speed_bias_i.block<3, 3>(StateOrder::O_P, StateOrder::O_BG - StateOrder::O_V) =
                        -dp_dbg;
                jacobian_speed_bias_i.block<3, 3>(StateOrder::O_R, StateOrder::O_BG - StateOrder::O_V) =
                        -Utility::EigenBase::Qleft(q_j.inverse() * q_i * delta_q_).bottomRightCorner<3, 3>() * dq_dbg;

                jacobian_speed_bias_i.block<3, 3>(StateOrder::O_V, StateOrder::O_V - StateOrder::O_V) =
                        -q_i.inverse().toRotationMatrix();
                jacobian_speed_bias_i.block<3, 3>(StateOrder::O_V, StateOrder::O_BA - StateOrder::O_V) =
                        -dv_dba;
                jacobian_speed_bias_i.block<3, 3>(StateOrder::O_V, StateOrder::O_BG - StateOrder::O_V) =
                        -dv_dbg;
                jacobian_speed_bias_i.block<3, 3>(StateOrder::O_BA, StateOrder::O_BA - StateOrder::O_V) =
                        -Matrix3::Identity();

                jacobian_speed_bias_i.block<3, 3>(StateOrder::O_BG, StateOrder::O_BG - StateOrder::O_V) =
                        -Matrix3::Identity();

                jacobian_speed_bias_i = sqrt_information_ * jacobian_speed_bias_i;
            }
            if (jacobians[2])
            {
                MapMatrix15_7 jacobian_pose_j(jacobians[2]);
                jacobian_pose_j.setZero();

                jacobian_pose_j.block<3, 3>(StateOrder::O_P, StateOrder::O_P) =
                        q_i.inverse().toRotationMatrix();

                jacobian_pose_j.block<3, 3>(StateOrder::O_R, StateOrder::O_R) =
                        Utility::EigenBase::Qleft(corrected_delta_q.inverse() * q_i.inverse() * q_j).bottomRightCorner<3, 3>();

                jacobian_pose_j = sqrt_information_ * jacobian_pose_j;
            }
            if (jacobians[3])
            {
                MapMatrix15_9 jacobian_speed_bias_j(jacobians[3]);
                jacobian_speed_bias_j.setZero();

                jacobian_speed_bias_j.block<3, 3>(StateOrder::O_V,  StateOrder::O_V -  StateOrder::O_V) =
                        q_i.inverse().toRotationMatrix();

                jacobian_speed_bias_j.block<3, 3>(StateOrder::O_BA, StateOrder::O_BA - StateOrder::O_V) =
                        Matrix3::Identity();

                jacobian_speed_bias_j.block<3, 3>(StateOrder::O_BG, StateOrder::O_BG - StateOrder::O_V) =
                        Matrix3::Identity();

                jacobian_speed_bias_j = sqrt_information_* jacobian_speed_bias_j;
            }
        }

        return true;
    }
}//end of SuperVIO