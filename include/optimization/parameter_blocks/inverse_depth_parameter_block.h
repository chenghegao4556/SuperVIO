//
// Created by chenghe on 3/31/20.
//

#ifndef SUPER_VIO_INVERSE_DEPTH_PARAMETER_BLOCK_H
#define SUPER_VIO_INVERSE_DEPTH_PARAMETER_BLOCK_H
#include <optimization/parameter_blocks/base_parameter_block.h>
namespace SuperVIO::Optimization
{
    class InverseDepthParameterBlock: public BaseParametersBlock
    {
    public:
        static Ptr Creat(double depth);

        [[nodiscard]] Type GetType() const override;
        [[nodiscard]] size_t GetGlobalSize() const override;
        [[nodiscard]] size_t GetLocalSize() const override;
        [[nodiscard]] ceres::LocalParameterization* GetLocalParameterization() const override;
    protected:
        void SetData(double depth);

        explicit InverseDepthParameterBlock(double depth);
    };//end of InverseDepthParameterBlock
}//end of SuperVIO

#endif //SUPER_VIO_INVERSE_DEPTH_PARAMETER_BLOCK_H
