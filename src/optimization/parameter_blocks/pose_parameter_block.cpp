//
// Created by chenghe on 3/31/20.
//
#include <optimization/parameter_blocks/pose_parameter_block.h>
namespace SuperVIO::Optimization
{
    ///////////////////////////////////////////////////////////////////////////////////
    PoseParameterBlock::Ptr PoseParameterBlock::
    Creat(const Quaternion& q,
          const Vector3& t)
    {
        Ptr p(new PoseParameterBlock(q, t));
        return p;
    }

    ///////////////////////////////////////////////////////////////////////////////////
    void PoseParameterBlock::
    SetData(const Quaternion& q, const Vector3& t)
    {
        this->SetValid(true);

        data_[0] = t(0);
        data_[1] = t(1);
        data_[2] = t(2);

        data_[3] = q.x();
        data_[4] = q.y();
        data_[5] = q.z();
        data_[6] = q.w();
    }

    ///////////////////////////////////////////////////////////////////////////////////
    PoseParameterBlock::
    PoseParameterBlock(const Quaternion& q,
                       const Vector3& t):
        BaseParametersBlock(static_cast<size_t>(GlobalSize::Pose))
    {
        this->SetData(q, t);
    }

    ///////////////////////////////////////////////////////////////////////////////////
    PoseParameterBlock::Type PoseParameterBlock::
    GetType() const
    {
        return Type::Pose;
    }

    ///////////////////////////////////////////////////////////////////////////////////
    size_t PoseParameterBlock::
    GetGlobalSize() const
    {
        return static_cast<size_t>(GlobalSize::Pose);
    }

    ///////////////////////////////////////////////////////////////////////////////////
    size_t PoseParameterBlock::
    GetLocalSize() const
    {
        return static_cast<size_t>(LocalSize::Pose);
    }

    ///////////////////////////////////////////////////////////////////////////////////
    ceres::LocalParameterization* PoseParameterBlock::
    GetLocalParameterization() const
    {
        return new PoseLocalParameterization;
    }

}//end of SuperVIO