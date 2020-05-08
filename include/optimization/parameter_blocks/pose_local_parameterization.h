
//
// Created by chenghe on 19/3/20.
//

#ifndef SUPER_VIO_POSE_LOCAL_PARAMETERIZATION_H
#define SUPER_VIO_POSE_LOCAL_PARAMETERIZATION_H

#include <utility/eigen_type.h>
#include <ceres/ceres.h>

namespace SuperVIO::Optimization
{
    class PoseLocalParameterization : public ceres::LocalParameterization
    {
    public:
        /**
         * @brief destructor
         */
        ~PoseLocalParameterization() override = default;

        /**
         * @brief define operator "+" for pose in tangent plane
         */
        bool Plus(const double* x,
                  const double* delta_x,
                  double* x_plus_delta_x) const override;

        /**
         * @brief compute jacobian 7 x 6 matrix
         */
        bool ComputeJacobian(const double* x, double* jacobian) const override;

        /**
         * @brief return global size pose parameters
         */
        [[nodiscard]] int GlobalSize() const override;

        /**
         * @brief return global size pose parameters
         */
        [[nodiscard]] int LocalSize() const override;

    };// PoseLocalParameterization

}

#endif //SUPER_VIO_POSE_LOCAL_PARAMETERIZATION_H
