//
// Created by chenghe on 5/17/20.
//

#ifndef SUPER_VIO_RELATIVE_POSE_RESIDUAL_BLOCK_H
#define SUPER_VIO_RELATIVE_POSE_RESIDUAL_BLOCK_H

#include <optimization/factors/cost_function_factory.h>
#include <optimization/residual_blocks/base_residual_block.h>
namespace SuperVIO::Optimization
{
    class RelativePoseResidualBlock: public BaseResidualBlock
    {
    public:
        static Ptr Creat(const Quaternion& q_ij, const Vector3& p_ij,
                         const double& position_var, const double& rotation_var,
                         const std::vector<ParameterBlockPtr>& parameter_blocks);

        [[nodiscard]] Type GetType() const override;

    protected:
        explicit RelativePoseResidualBlock(const Quaternion& q_ij, const Vector3& p_ij,
                                           const double& position_var, const double& rotation_var,
                                           const std::vector<ParameterBlockPtr>& parameter_blocks);
    };//end of RelativePoseResidualBlock
}//end of SuperVIO::Optimization

#endif //SRC_RELATIVE_POSE_RESIDUAL_BLOCK_H
