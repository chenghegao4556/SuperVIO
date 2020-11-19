//
// Created by chenghe on 3/31/20.
//

#ifndef SUPER_VIO_SPEED_BIAS_PARAMETER_BLOCK_H
#define SUPER_VIO_SPEED_BIAS_PARAMETER_BLOCK_H
#include <optimization/parameter_blocks/base_parameter_block.h>
namespace SuperVIO::Optimization
{
    class SpeedBiasParameterBlock: public BaseParametersBlock
    {
    public:
        static Ptr Creat(const Vector3& speed,
                         const Vector3& ba,
                         const Vector3& bg);

        [[nodiscard]] Type GetType() const override;
        [[nodiscard]] size_t GetGlobalSize() const override;
        [[nodiscard]] size_t GetLocalSize() const override;
        [[nodiscard]] ceres::LocalParameterization* GetLocalParameterization() const override;

    protected:
        void SetData(const Vector3& speed,
                     const Vector3& ba,
                     const Vector3& bg);

        explicit SpeedBiasParameterBlock(const Vector3& speed,
                                         const Vector3& ba,
                                         const Vector3& bg);
    };//end of InverseDepthParameterBlock
}//end of SuperVIO

#endif //SRC_SPEED_BIAS_PARAMETER_BLOCK_H
