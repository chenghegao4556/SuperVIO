//
// Created by chenghe on 5/17/20.
//

#ifndef SUPER_VIO_RELATIVE_POSE_FACTOR_H
#define SUPER_VIO_RELATIVE_POSE_FACTOR_H
#include <ceres/ceres.h>
#include <utility/eigen_type.h>
#include <utility/eigen_base.h>
namespace SuperVIO::Optimization
{
    class RelativePoseFactor: public ceres::SizedCostFunction<6, 7, 7>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        RelativePoseFactor(const Quaternion& q_ij,
                           Vector3 p_ij,
                           const double& position_var,
                           const double& rotation_var);

        bool Evaluate(double const *const *parameters,
                      double *residuals,
                      double **jacobians) const override;

    private:
        const Quaternion q_ij_;
        const Vector3    p_ij_;
        Matrix6    sqrt_info_;
    };//end of RelativePoseFactor
}//end of SuperVIO::Optimization

#endif //SUPER_VIO_RELATIVE_POSE_FACTOR_H
