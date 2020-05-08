//
// Created by chenghe on 3/5/20.
//
#include <imu/imu_checker.h>

namespace SuperVIO::IMU
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    bool IMUCheck::
    IsFullyExcited(const IMUPreIntegrationMeasurementMap& imu_pre_integration_measurement_map)
    {
        /**
         * delta_v has information about acceleration and rotation,
         * <<VINS on Wheels>> proofs that if only constant acceleration or no
         * rotation during initialization, the estimated scale / pitch and roll are inaccurate.
         */
        Vector3 sum_delta_v = Vector3::Zero();
        auto size = (double)imu_pre_integration_measurement_map.size();
        for(const auto& pre: imu_pre_integration_measurement_map)
        {
            sum_delta_v += pre.second.delta_v / pre.second.sum_dt;
        }
        Vector3 average_delta_v = sum_delta_v * 1.0 / size;

        double variance = 0;
        for(const auto& pre: imu_pre_integration_measurement_map)
        {
            Vector3 tmp_delta_v = pre.second.delta_v / pre.second.sum_dt;
            variance += (tmp_delta_v - average_delta_v).transpose() * (tmp_delta_v - average_delta_v);
        }
        double standard_variance = std::sqrt(variance / size);
        ROS_INFO_STREAM("IMU variance"<<standard_variance);
        if(standard_variance < 0.25)
        {
            ROS_INFO("IMU excitation not enough!");
            return false;
        }
        else
        {
            return true;
        }
    }

}//end of SuperVIO
