//
// Created by chenghe on 4/2/20.
//
#include <optimization/residual_blocks/marginalization_residual_block.h>

namespace SuperVIO::Optimization
{
    typedef MarginalizationResidualBlock::Ptr Ptr;
    typedef MarginalizationResidualBlock::Type Type;

    ///////////////////////////////////////////////////////////////////////////////////
    Ptr MarginalizationResidualBlock::
    Creat(const MarginalizationInformation& marginalization_information)
    {
        Ptr p(new MarginalizationResidualBlock(marginalization_information));
        return p;
    }

    ///////////////////////////////////////////////////////////////////////////////////
    Type MarginalizationResidualBlock::
    GetType() const
    {
        return Type::Marginalization;
    }

    ///////////////////////////////////////////////////////////////////////////////////
    MarginalizationResidualBlock::
    MarginalizationResidualBlock(const MarginalizationInformation& marginalization_information):
            BaseResidualBlock(CostFunctionFactory::
            CreatMarginalizationCostFunction(marginalization_information),
                              nullptr,
                              marginalization_information.parameter_blocks)
    {

    }
}//end of SuperVIO
