//
// Created by chenghe on 4/2/20.
//

#ifndef SUPER_VIO_HELPER_H
#define SUPER_VIO_HELPER_H

#include <optimization/parameter_blocks/parameter_block_factory.h>
#include <optimization/residual_blocks/residual_block_factory.h>
#include <vision/feature_tracker.h>
#include <imu/imu_processor.h>
#include <utility/cv_base.h>
#include <vision/camera.h>
namespace SuperVIO::Optimization
{
    class Helper
    {
    public:

        typedef ResidualBlockFactory::Ptr ResidualBlockPtr;
        typedef ParameterBlockFactory::Ptr ParameterBlockPtr;

        struct SpeedBiases
        {
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            SpeedBiases(Vector3 _speed, Vector3 _ba, Vector3 _bg);
            Vector3 speed;
            Vector3 ba;
            Vector3 bg;
        };//end of SpeedBiases

        static double
        GetDepthFromParameterBlock(const ParameterBlockPtr& ptr);

        static std::pair<Quaternion, Vector3>
        GetPoseFromParameterBlock(const ParameterBlockPtr& ptr);

        static SpeedBiases
        GetSpeedBiasFromParameterBlock(const ParameterBlockPtr& ptr);

        static std::vector<ResidualBlockPtr>
        CreatReprojectionResidualBlocks(const std::map<IMU::StateKey, ParameterBlockPtr>& pose_block_map,
                                        const std::map<Vision::TrackKey, ParameterBlockPtr>& feature_block_map,
                                        const Vision::TrackMap& track_map,
                                        const Vision::FrameMeasurementMap& frame_measurement_map,
                                        const Vision::Camera::ConstPtr& camera_ptr,
                                        const Quaternion& q_i_c,
                                        const Vector3& p_i_c);

        static ResidualBlockPtr
        CreatIMUResidualBlock(const std::map<IMU::StateKey, ParameterBlockPtr>& pose_block_map,
                              const std::map<IMU::StateKey, ParameterBlockPtr>& speed_bias_block_map,
                              const IMU::IMUPreIntegrationMeasurementMap& pre_integration_map,
                              const Vector3& gravity_vector,
                              const IMU::StatePairKey& pair_key);

        static Vision::FeatureState
        UpdateFeatureState(const Vision::TrackMap& track_map,
                           const Vision::FrameMeasurementMap& frame_measurement_map,
                           const IMU::IMUStateMap& imu_state_map,
                           const Vision::TrackKey& track_key,
                           const ParameterBlockPtr& depth_ptr,
                           const Vision::Camera::ConstPtr& camera_ptr,
                           const Quaternion& q_i_c,
                           const Vector3& p_i_c);

        static double
        ResetFeatureDepth(const Vision::TrackMap& track_map,
                          const Vision::FeatureStateMap& feature_state_map,
                          const IMU::IMUStateMap& imu_state_map,
                          const Vision::TrackKey& track_key,
                          const Vision::Camera::ConstPtr& camera_ptr,
                          const Quaternion& q_i_c,
                          const Vector3& p_i_c);

    };
}//end of SuperVIO
#endif //SUPER_VIO_HELPER_H
