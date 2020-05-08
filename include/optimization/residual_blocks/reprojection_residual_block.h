//
// Created by chenghe on 4/2/20.
//

#ifndef SUPER_VIO_REPROJECTION_RESIDUAL_BLOCK_H
#define SUPER_VIO_REPROJECTION_RESIDUAL_BLOCK_H

#include <optimization/factors/cost_function_factory.h>
#include <optimization/residual_blocks/base_residual_block.h>

namespace SuperVIO::Optimization
{

    class ReprojectionResidualBlock: public BaseResidualBlock
    {
    public:



        static Ptr Creat(const Matrix3& intrinsic,
                         const Quaternion& r_i_c,
                         const Vector3& t_i_c,
                         const Vector2& measurement_i,
                         const Vector2& measurement_j,
                         const double& sigma_x,
                         const double& sigma_y,
                         const std::vector<ParameterBlockPtr>& parameter_blocks);

        [[nodiscard]] Type GetType() const override;

    protected:
        explicit ReprojectionResidualBlock(const Matrix3& intrinsic,
                                           const Quaternion& r_i_c,
                                           const Vector3& t_i_c,
                                           const Vector2& measurement_i,
                                           const Vector2& measurement_j,
                                           const double& sigma_x,
                                           const double& sigma_y,
                                           const std::vector<ParameterBlockPtr>& parameter_blocks);
    };
}//end of SuperVIO

#endif //SUPER_VIO_REPROJECTION_RESIDUAL_BLOCK_H
