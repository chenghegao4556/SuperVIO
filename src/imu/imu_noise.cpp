//
// Created by chenghe on 3/5/20.
//

#include <imu/imu_noise.h>

namespace SuperVIO::IMU
{
    //////////////////////////////////////////////////////////////////////////////////////////
    Matrix18 IMUNoise::
    CreatNoiseMatrix(double accelerator_noise,
                     double gyroscope_noise,
                     double accelerator_random_walk_noise,
                     double gyroscope_random_walk_noise)
    {
        Matrix18 noise = Matrix18::Zero();

        noise.block<3, 3>(0, 0) =  (accelerator_noise * accelerator_noise) * Matrix3::Identity();
        noise.block<3, 3>(3, 3) =  (gyroscope_noise *   gyroscope_noise)   * Matrix3::Identity();
        noise.block<3, 3>(6, 6) =  (accelerator_noise * accelerator_noise) * Matrix3::Identity();
        noise.block<3, 3>(9, 9) =  (gyroscope_noise *   gyroscope_noise)   * Matrix3::Identity();

        noise.block<3, 3>(12, 12) = (accelerator_random_walk_noise * accelerator_random_walk_noise) * Matrix3::Identity();
        noise.block<3, 3>(15, 15) = (gyroscope_random_walk_noise * gyroscope_random_walk_noise) *     Matrix3::Identity();

        return noise;
    }
}//end of SuperVIO