//
// Created by chenghe on 4/2/20.
//

#ifndef SUPER_VIO_MARGINALIZATION_RESIDUAL_BLOCK_H
#define SUPER_VIO_MARGINALIZATION_RESIDUAL_BLOCK_H

#include <optimization/factors/cost_function_factory.h>
#include <optimization/residual_blocks/base_residual_block.h>
namespace SuperVIO::Optimization
{
    class MarginalizationResidualBlock: public BaseResidualBlock
    {
    public:
        static Ptr Creat(const MarginalizationInformation& marginalization_information);

        [[nodiscard]] Type GetType() const override;

    protected:
        explicit MarginalizationResidualBlock(const MarginalizationInformation& marginalization_information);
    };
}//end of SuperVIO

#endif //SRC_MARGINALIZATION_RESIDUAL_BLOCK_H
