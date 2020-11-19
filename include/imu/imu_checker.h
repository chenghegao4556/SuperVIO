//
// Created by chenghe on 3/5/20.
//

#ifndef SUPER_VIO_IMU_CHECKER_H
#define SUPER_VIO_IMU_CHECKER_H

#include <imu/imu_states_measurements.h>

namespace SuperVIO::IMU
{
    class IMUCheck
    {
    public:

        static bool IsFullyExcited(const IMUPreIntegrationMeasurementMap& imu_pre_integration_measurement_map);

    };//end of IMUCheck
}//end of SuperVIO

#endif //SUPER_VIO_IMU_CHECKER_H
