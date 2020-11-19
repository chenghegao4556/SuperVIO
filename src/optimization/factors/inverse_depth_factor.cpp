//
// Created by chenghe on 4/4/20.
//

#include <optimization/factors/inverse_depth_factor.h>
namespace SuperVIO::Optimization
{
    ///////////////////////////////////////////////////////////////////////////////////////
    InverseDepthFactor::
    InverseDepthFactor(const Matrix3& _intrinsic,
                       const Quaternion& q_i_c,
                       Vector3  t_i_c,
                       Vector2 _measurement_i,
                       Vector2 _measurement_j,
                       const double  &sigma_x,
                       const double  &sigma_y):
            intrinsic_(_intrinsic),
            intrinsic_inv_(_intrinsic.inverse()),
            fx_(_intrinsic(0, 0)),
            fy_(_intrinsic(1, 1)),
            q_i_c_(q_i_c),
            t_i_c_(std::move(t_i_c)),
            measurement_i_(std::move(_measurement_i)),
            measurement_j_(std::move(_measurement_j))
    {
        sqrt_information_<<1.0 / sigma_x, 0.0, 0.0, 1.0/sigma_y;
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    bool InverseDepthFactor::
    Evaluate(double const *const *parameters,
             double *residuals,
             double **jacobians) const
    {
        const ConstMapVector3 p_i(parameters[0]);
        const ConstMapQuaternion q_i(parameters[0]+3);

        const ConstMapVector3 p_j(parameters[1]);
        const ConstMapQuaternion q_j(parameters[1]+3);

        const double inverse_depth_i = parameters[2][0];

        const Vector3 point_camera_i = (intrinsic_inv_ * measurement_i_.homogeneous()) / inverse_depth_i;

        const Vector3 point_imu_i    = q_i_c_ * point_camera_i + t_i_c_;

        const Vector3 point_world    = q_i * point_imu_i + p_i;

        const Vector3 point_imu_j    = q_j.inverse() * (point_world - p_j);

        const Vector3 point_camera_j = q_i_c_.inverse() * (point_imu_j - t_i_c_);

        const Vector2 prediction_j   = (intrinsic_ * (point_camera_j / point_camera_j(2))).head<2>();


        MapVector2 residual_vector(residuals);
        residual_vector = prediction_j - measurement_j_;
        residual_vector = sqrt_information_ * residual_vector;

        if (jacobians)
        {
            const Matrix3 r_i = q_i.toRotationMatrix();
            const Matrix3 r_j = q_j.toRotationMatrix();
            const Matrix3 ric = q_i_c_.toRotationMatrix();

            const double x_j = point_camera_j(0);
            const double y_j = point_camera_j(1);
            const double z_j = point_camera_j(2);

            Matrix23 dr_dpts_j;

            dr_dpts_j<< fx_ * 1.0 / z_j,               0,  -fx_ * x_j /(z_j * z_j),
                        0,               fy_ * 1.0 / z_j,  -fy_ * y_j /(z_j * z_j);
            dr_dpts_j = sqrt_information_ * dr_dpts_j;

            if (jacobians[0])
            {
                MapMatrix27 jacobian_pose_i(jacobians[0]);

                Matrix36 dpts_j_dpose_i;

                dpts_j_dpose_i.leftCols<3>()  =  ric.transpose() * r_j.transpose();
                dpts_j_dpose_i.rightCols<3>() = -ric.transpose() * r_j.transpose() *
                        r_i * Utility::EigenBase::SkewSymmetric(point_imu_i);

                jacobian_pose_i.leftCols<6>() = dr_dpts_j * dpts_j_dpose_i;
                jacobian_pose_i.rightCols<1>().setZero();
            }

            if (jacobians[1])
            {
                MapMatrix27 jacobian_pose_j(jacobians[1]);

                Matrix36 dpts_j_dpose_j;
                dpts_j_dpose_j.leftCols<3>()  = -ric.transpose() * r_j.transpose();
                dpts_j_dpose_j.rightCols<3>() =  ric.transpose() *
                        Utility::EigenBase::SkewSymmetric(point_imu_j);

                jacobian_pose_j.leftCols<6>() = dr_dpts_j * dpts_j_dpose_j;
                jacobian_pose_j.rightCols<1>().setZero();
            }

            if (jacobians[2])
            {
                MapVector2 jacobian_inverse_depth_i(jacobians[2]);
                jacobian_inverse_depth_i = dr_dpts_j * ric.transpose() * r_j.transpose() *
                        r_i * ric * (intrinsic_inv_ * measurement_i_.homogeneous()) * -1.0 / (inverse_depth_i * inverse_depth_i);
            }
        }

        return true;
    }
}//end of SuperVIO
