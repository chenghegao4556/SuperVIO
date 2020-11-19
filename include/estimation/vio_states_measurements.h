//
// Created by chenghe on 4/12/20.
//

#ifndef SRC_VIO_STATES_MEASUREMENTS_H
#define SRC_VIO_STATES_MEASUREMENTS_H

#include <vision/vision_measurements.h>
#include <imu/imu_states_measurements.h>
#include <optimization/parameter_blocks/parameter_block_factory.h>
#include <optimization/residual_blocks/residual_block_factory.h>

namespace SuperVIO::Estimation
{
    class VIOStatesMeasurements
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef Optimization::BaseParametersBlock::Ptr ParameterBlockPtr;


        VIOStatesMeasurements();

        bool lost;

        bool initialized;

        /*******************************IMU STATES & MEASUREMENTS******************************************/
        //! imu states: key: time stamp(double)
        IMU::IMUStateMap imu_state_map;

        //! imu raw measurements: dt buffer; acceleration buffer; angular velocity buffer
        //! key: <start time stamp, end time stamp>
        IMU::IMURawMeasurementsMap imu_raw_measurements_map;

        //!  imu pre-integration measurements: jacobians, covariance, delta_q, delta_v, delta_p, etc..
        //! key: <start time stamp, end time stamp>
        IMU::IMUPreIntegrationMeasurementMap imu_pre_integration_measurements_map;


        /*******************************VISION STATES & MEASUREMENTS******************************************/
        //! feature states: point cloud and depth of first measurement(in camera coordinate)
        //! key: track id(size_t)
        Vision::FeatureStateMap feature_state_map;

        //! feature measurements: key point position and corresponding frame id(imu state id)
        //! key: track id(size_t)
        Vision::TrackMap track_map;

        //! vision measurements per frame: descriptors, key points
        //! key: time stamp(double)
        Vision::FrameMeasurementMap frame_measurement_map;

        /*****************************OPTIMIZABLE PARAMETERS(Ceres)******************************************/

        //! parameter block map
        std::map<IMU::StateKey,    ParameterBlockPtr> pose_block_map;
        std::map<IMU::StateKey,    ParameterBlockPtr> speed_bias_block_map;

        //! hessian prior
        Optimization::MarginalizationInformation marginalization_information;

        Vector3 gravity_vector;

        Vector3  acceleration_0;
        Vector3  angular_velocity_0;
    };

}//end of SuperVIO::Estimation

#endif //SRC_VIO_STATES_MEASUREMENTS_H
