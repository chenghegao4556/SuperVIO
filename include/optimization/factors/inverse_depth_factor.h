//
// Created by chenghegao on 19/3/20.
//

#ifndef SUPER_VIO_INVERSE_DEPTH_FACTOR_H
#define SUPER_VIO_INVERSE_DEPTH_FACTOR_H

#include <utility/eigen_type.h>
#include <ceres/ceres.h>
#include <utility/eigen_base.h>

namespace SuperVIO::Optimization
{
    class InverseDepthFactor: public ceres::SizedCostFunction<2, 7, 7, 1>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        InverseDepthFactor(const Matrix3& _intrinsic,
                           const Quaternion& q_i_c,
                           Vector3  t_i_c,
                           Vector2 _measurement_i,
                           Vector2 _measurement_j,
                           const double  &sigma_x = 1.0,
                           const double  &sigma_y = 1.0);

        bool Evaluate(double const *const *parameters,
                      double *residuals,
                      double **jacobians) const override;

    private:
        Matrix3 intrinsic_;
        Matrix3 intrinsic_inv_;

        double fx_;
        double fy_;

        Quaternion q_i_c_;
        Vector3    t_i_c_;

        Vector2 measurement_i_;
        Vector2 measurement_j_;
        Matrix2 sqrt_information_;
    };//end of InverseDepthFactor
}//end of SuperVIO


#endif //SUPER_VIO_INVERSE_DEPTH_FACTOR_H
