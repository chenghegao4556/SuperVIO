//
// Created by chenghe on 4/2/20.
//
#include <optimization/parameter_blocks/parameter_block_factory.h>

namespace SuperVIO::Optimization
{

    typedef ParameterBlockFactory::Ptr Ptr;

    ///////////////////////////////////////////////////////////////////////////////////
    Ptr ParameterBlockFactory::
    CreatInverseDepth(double depth)
    {
        return InverseDepthParameterBlock::Creat(depth);
    }

    ///////////////////////////////////////////////////////////////////////////////////
    Ptr ParameterBlockFactory::
    CreatPose(const Quaternion& q, const Vector3& t)
    {
        return PoseParameterBlock::Creat(q, t);
    }

    ///////////////////////////////////////////////////////////////////////////////////
    Ptr ParameterBlockFactory::
    CreatSpeedBias(const Vector3& speed, const Vector3& ba, const Vector3& bg)
    {
        return SpeedBiasParameterBlock::Creat(speed, ba, bg);
    }
}//end of SuperVIO