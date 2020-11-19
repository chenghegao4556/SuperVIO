//
// Created by chenghe on 3/31/20.
//
#include <optimization/parameter_blocks/speed_bias_parameter_block.h>
namespace SuperVIO
{
namespace Optimization
{
    ///////////////////////////////////////////////////////////////////////////////////
    SpeedBiasParameterBlock::Ptr SpeedBiasParameterBlock::
    Creat(const Vector3& speed,
          const Vector3& ba,
          const Vector3& bg)
    {
        Ptr p(new SpeedBiasParameterBlock(speed, ba, bg));
        return p;
    }

    ///////////////////////////////////////////////////////////////////////////////////
    void SpeedBiasParameterBlock::
    SetData(const Vector3& speed,
            const Vector3& ba,
            const Vector3& bg)
    {
        this->SetValid(true);

        data_[0] = speed(0);
        data_[1] = speed(1);
        data_[2] = speed(2);

        data_[3] = ba(0);
        data_[4] = ba(1);
        data_[5] = ba(2);

        data_[6] = bg(0);
        data_[7] = bg(1);
        data_[8] = bg(2);
    }

    ///////////////////////////////////////////////////////////////////////////////////
    SpeedBiasParameterBlock::
    SpeedBiasParameterBlock(const Vector3& speed,
                            const Vector3& ba,
                            const Vector3& bg):
            BaseParametersBlock(static_cast<size_t>(GlobalSize::SpeedBias))
    {

    }

    ///////////////////////////////////////////////////////////////////////////////////
    SpeedBiasParameterBlock::Type SpeedBiasParameterBlock::
    GetType() const
    {
        return Type::SpeedBias;
    }

    ///////////////////////////////////////////////////////////////////////////////////
    size_t SpeedBiasParameterBlock::
    GetGlobalSize() const
    {
        return static_cast<size_t>(GlobalSize::SpeedBias);
    }

    ///////////////////////////////////////////////////////////////////////////////////
    size_t SpeedBiasParameterBlock::
    GetLocalSize() const
    {
        return static_cast<size_t>(LocalSize::SpeedBias);
    }

    ///////////////////////////////////////////////////////////////////////////////////
    ceres::LocalParameterization* SpeedBiasParameterBlock::
    GetLocalParameterization() const
    {
        return nullptr;
    }
}//end of Optimization
}//end of SuperVIO
