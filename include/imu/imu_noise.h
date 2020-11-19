//
// Created by chenghe on 3/5/20.
//

#ifndef SUPER_VIO_IMU_NOISE_H
#define SUPER_VIO_IMU_NOISE_H

#include <memory>
#include <ros/ros.h>
#include <utility/eigen_type.h>
namespace SuperVIO::IMU
{
    class IMUNoise
    {
    public:
        //! initialize Singleton
        static Matrix18
        CreatNoiseMatrix(double accelerator_noise = 0.1,
                         double gyroscope_noise = 0.01,
                         double accelerator_random_walk_noise = 0.001,
                         double gyroscope_random_walk_noise = 0.0001);
    };//end of IMUNoise
}//end of SuperVIO

#endif //SUPER_VIO_IMU_NOISE_H
