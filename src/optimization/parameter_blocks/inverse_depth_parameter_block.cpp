//
// Created by chenghe on 3/31/20.
//
#include <optimization/parameter_blocks/inverse_depth_parameter_block.h>
namespace SuperVIO::Optimization
{
    ///////////////////////////////////////////////////////////////////////////////////
    InverseDepthParameterBlock::Ptr InverseDepthParameterBlock::
    Creat(double depth)
    {
        Ptr p(new InverseDepthParameterBlock(depth));
        return p;
    }

    ///////////////////////////////////////////////////////////////////////////////////
    void InverseDepthParameterBlock::
    SetData(double depth)
    {
        this->SetValid(true);

        data_[0] = 1.0 / depth;
    }

    ///////////////////////////////////////////////////////////////////////////////////
    InverseDepthParameterBlock::
    InverseDepthParameterBlock(double depth):
            BaseParametersBlock(static_cast<size_t>(GlobalSize::InverseDepth))
    {
        this->SetData(depth);
    }

    ///////////////////////////////////////////////////////////////////////////////////
    InverseDepthParameterBlock::Type InverseDepthParameterBlock::
    GetType() const
    {
        return Type::InverseDepth;
    }

    ///////////////////////////////////////////////////////////////////////////////////
    size_t InverseDepthParameterBlock::
    GetGlobalSize() const
    {
        return static_cast<size_t>(GlobalSize::InverseDepth);
    }

    ///////////////////////////////////////////////////////////////////////////////////
    size_t InverseDepthParameterBlock::
    GetLocalSize() const
    {
        return static_cast<size_t>(LocalSize::InverseDepth);
    }

    ///////////////////////////////////////////////////////////////////////////////////
    ceres::LocalParameterization* InverseDepthParameterBlock::
    GetLocalParameterization() const
    {
        return nullptr;
    }
}//end of SuperVIO