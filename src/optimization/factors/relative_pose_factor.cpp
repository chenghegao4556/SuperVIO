//
// Created by chenghe on 5/17/20.
//
#include <optimization/factors/relative_pose_factor.h>
namespace SuperVIO::Optimization
{
    ////////////////////////////////////////////////////////////////////////////
    RelativePoseFactor::
    RelativePoseFactor(const Quaternion& q_ij,
                       Vector3 p_ij,
                       const double& position_var,
                       const double& rotation_var):
                        q_ij_(q_ij),
                        p_ij_(std::move(p_ij))
    {
        sqrt_info_ = Matrix6::Identity();
        sqrt_info_(0, 0) = 1.0 / position_var;
        sqrt_info_(1, 1) = 1.0 / position_var;
        sqrt_info_(2, 2) = 1.0 / position_var;
        sqrt_info_(3, 3) = 1.0 / rotation_var;
        sqrt_info_(4, 4) = 1.0 / rotation_var;
        sqrt_info_(5, 5) = 1.0 / rotation_var;
    }

    ////////////////////////////////////////////////////////////////////////////
    bool RelativePoseFactor::
    Evaluate(double const *const *parameters,
             double *residuals,
             double **jacobians) const
    {
        const ConstMapVector3 p_i(parameters[0]);
        const ConstMapQuaternion q_i(parameters[0]+3);

        const ConstMapVector3 p_j(parameters[1]);
        const ConstMapQuaternion q_j(parameters[1]+3);


        MapVector6 residual_vector(residuals);
        residual_vector.head<3>() = q_i.inverse() * (p_j - p_i);
        residual_vector.tail<3>() = 2.0 * (q_ij_.inverse() * (q_i.inverse() * q_j)).vec();
        residual_vector = sqrt_info_ * residual_vector;

        if (jacobians)
        {
            if (jacobians[0])
            {
                MapMatrix6 jacobian_pose_i(jacobians[0]);
                jacobian_pose_i.setZero();
                jacobian_pose_i.block<3, 3>(0, 0) = -q_i.inverse().toRotationMatrix();
                jacobian_pose_i.block<3, 3>(3, 3) =
                        -(Utility::EigenBase::Qleft (q_j.inverse() * q_i) *
                          Utility::EigenBase::Qright(q_ij_)).bottomRightCorner<3, 3>();

                jacobian_pose_i = sqrt_info_ * jacobian_pose_i;
            }

            if (jacobians[1])
            {
                MapMatrix6 jacobian_pose_j(jacobians[0]);
                jacobian_pose_j.setZero();
                jacobian_pose_j.block<3, 3>(0, 0) = q_j.inverse().toRotationMatrix();
                jacobian_pose_j.block<3, 3>(3, 3) =
                        Utility::EigenBase::Qleft(q_ij_.inverse() * q_i.inverse() * q_j).bottomRightCorner<3, 3>();
                jacobian_pose_j = sqrt_info_ * jacobian_pose_j;
            }
        }

        return true;
    }
}//end of SuperVIO::Optimization
