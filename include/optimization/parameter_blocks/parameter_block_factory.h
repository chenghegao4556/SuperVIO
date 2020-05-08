//
// Created by chenghe on 4/2/20.
//

#ifndef SUPER_VIO_PARAMETER_BLOCK_FACTORY_H
#define SUPER_VIO_PARAMETER_BLOCK_FACTORY_H

#include <optimization/parameter_blocks/inverse_depth_parameter_block.h>
#include <optimization/parameter_blocks/speed_bias_parameter_block.h>
#include <optimization/parameter_blocks/pose_parameter_block.h>

#include <imu/imu_states_measurements.h>
namespace SuperVIO::Optimization
{
    class ParameterBlockFactory
    {
    public:
        typedef BaseParametersBlock::Ptr Ptr;
        static Ptr CreatInverseDepth(double depth);

        static Ptr CreatPose(const Quaternion& q,
                             const Vector3& t);

        static Ptr CreatSpeedBias(const Vector3& speed,
                                  const Vector3& ba,
                                  const Vector3& bg);
    };//end of ParameterBlockFactory
}//end of SuperVIO

#endif //SUPER_VIO_PARAMETER_BLOCK_FACTORY_H
