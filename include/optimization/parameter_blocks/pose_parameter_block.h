//
// Created by chenghe on 3/31/20.
//

#ifndef SUPER_VIO_POSE_PARAMETER_BLOCK_H
#define SUPER_VIO_POSE_PARAMETER_BLOCK_H

#include <optimization/parameter_blocks/pose_local_parameterization.h>
#include <optimization/parameter_blocks/base_parameter_block.h>
namespace SuperVIO::Optimization
{

    class PoseParameterBlock: public BaseParametersBlock
    {
    public:
        static Ptr Creat(const Quaternion& q,
                         const Vector3& t);

        [[nodiscard]] Type GetType() const override;
        [[nodiscard]] size_t GetGlobalSize() const override;
        [[nodiscard]] size_t GetLocalSize() const override;
        [[nodiscard]] ceres::LocalParameterization* GetLocalParameterization() const override;

    protected:
        void SetData(const Quaternion& q, const Vector3& t);
        explicit PoseParameterBlock(const Quaternion& q,
                                    const Vector3& t);

    };
}//end of SuperVIO

#endif //SUPER_VIO_POSE_PARAMETER_BLOCK_H
