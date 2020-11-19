//
// Created by chenghe on 4/2/20.
//
#include <optimization/helper.h>
namespace SuperVIO::Optimization
{

    typedef Helper::ParameterBlockPtr ParameterBlockPtr;
    typedef Helper::ResidualBlockPtr  ResidualBlockPtr;
    //////////////////////////////////////////////////////////////////////////////////////////////////
    Helper::SpeedBiases::
    SpeedBiases(Vector3 _speed, Vector3 _ba, Vector3 _bg):
        speed(std::move(_speed)),
        ba(std::move(_ba)),
        bg(std::move(_bg))
    {

    }


    //////////////////////////////////////////////////////////////////////////////////////////////////
    double Helper::
    GetDepthFromParameterBlock(const ParameterBlockPtr& ptr)
    {
        ROS_ASSERT(ptr->GetType() == BaseParametersBlock::Type::InverseDepth);
        double inverse_depth = *ptr->GetData();
        double depth = 1.0 / inverse_depth;
        return depth;
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////
    std::pair<Quaternion, Vector3> Helper::
    GetPoseFromParameterBlock(const ParameterBlockPtr& ptr)
    {
        ROS_ASSERT(ptr->GetType() == BaseParametersBlock::Type::Pose);
        Vector3 t(&ptr->GetData()[0]);
        Quaternion q(&ptr->GetData()[3]);

        return std::make_pair(q, t);
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////
    Helper::SpeedBiases Helper::
    GetSpeedBiasFromParameterBlock(const ParameterBlockPtr& ptr)
    {
        ROS_ASSERT(ptr->GetType() == BaseParametersBlock::Type::SpeedBias);
        Vector3 speed(&ptr->GetData()[0]);
        Vector3 ba(&ptr->GetData()[3]);
        Vector3 bg(&ptr->GetData()[6]);

        return SpeedBiases(speed, ba, bg);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    std::vector<ResidualBlockPtr> Helper::
    CreatReprojectionResidualBlocks(const std::map<IMU::StateKey, ParameterBlockPtr>& pose_block_map,
                                    const std::map<Vision::TrackKey, ParameterBlockPtr>& feature_block_map,
                                    const Vision::TrackMap& track_map,
                                    const Vision::FrameMeasurementMap& frame_measurement_map,
                                    const Vision::Camera::ConstPtr& camera_ptr,
                                    const Quaternion& q_i_c,
                                    const Vector3& p_i_c)
    {
        std::vector<ResidualBlockPtr> residual_blocks;

        for(const auto& feature_ptr: feature_block_map)
        {
            const auto& measurements  = track_map.at(feature_ptr.first).measurements;
            const auto& measurement_i = measurements.front();
            const auto& pose_block_i = pose_block_map.at(measurement_i.state_id);
            const auto& frame_i = frame_measurement_map.at(measurement_i.state_id);
            const auto& point_i = frame_i.key_points[measurement_i.point_id];

            for(size_t j = 1; j < measurements.size(); ++j)
            {
                const auto& measurement_j = measurements[j];
                const auto& pose_block_j = pose_block_map.at(measurement_j.state_id);
                const auto& frame_j = frame_measurement_map.at(measurement_j.state_id);
                const auto& point_j = frame_j.key_points[measurement_j.point_id];

                std::vector<ParameterBlockPtr> parameter_blocks{
                        pose_block_i, pose_block_j, feature_ptr.second};
                auto reprojection_ptr = Optimization::ResidualBlockFactory::CreatReprojection(
                        camera_ptr->GetIntrinsicMatrixEigen(),
                        q_i_c, p_i_c,
                        Utility::CVBase::Point2fToVector2(point_i.point),
                        Utility::CVBase::Point2fToVector2(point_j.point),
                        point_j.sigma_x, point_j.sigma_y, parameter_blocks);

                residual_blocks.push_back(reprojection_ptr);
            }

        }

        return residual_blocks;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ResidualBlockPtr Helper::
    CreatIMUResidualBlock(const std::map<IMU::StateKey, ParameterBlockPtr>& pose_block_map,
                          const std::map<IMU::StateKey, ParameterBlockPtr>& speed_bias_block_map,
                          const IMU::IMUPreIntegrationMeasurementMap& pre_integration_map,
                          const Vector3& gravity_vector,
                          const IMU::StatePairKey& pair_key)
    {
        const auto& pose_block_i = pose_block_map.at(pair_key.first);
        const auto& pose_block_j = pose_block_map.at(pair_key.second);
        const auto& speed_bias_block_i = speed_bias_block_map.at(pair_key.first);
        const auto& speed_bias_block_j = speed_bias_block_map.at(pair_key.second);

        std::vector<ParameterBlockPtr> parameter_blocks{
                pose_block_i, speed_bias_block_i, pose_block_j, speed_bias_block_j};

        const auto& pre_integration_measurement = pre_integration_map.at(pair_key);

        auto imu_residual_block = Optimization::ResidualBlockFactory::CreatPreIntegration(
                pre_integration_measurement, gravity_vector, parameter_blocks);

        return imu_residual_block;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    Vision::FeatureState Helper::
    UpdateFeatureState(const Vision::TrackMap& track_map,
                       const Vision::FrameMeasurementMap& frame_measurement_map,
                       const IMU::IMUStateMap& imu_state_map,
                       const Vision::TrackKey& track_key,
                       const ParameterBlockPtr& depth_ptr,
                       const Vision::Camera::ConstPtr& camera_ptr,
                       const Quaternion& q_i_c,
                       const Vector3& p_i_c)
    {
        auto depth = Optimization::Helper::GetDepthFromParameterBlock(depth_ptr);
        const auto& track = track_map.find(track_key)->second;
        const auto& measurements = track.measurements;
        const auto& first_state_id = measurements.begin()->state_id;
        const auto& first_point_id = measurements.begin()->point_id;
        const auto& frame_i = frame_measurement_map.find(first_state_id)->second;
        const Vector2 measurement_i = Utility::CVBase::Point2fToVector2(
                frame_i.key_points[first_point_id].point);

        const Vector3 camera_i_point = camera_ptr->BackProject(measurement_i) * depth;
        const Vector3 imu_i_point = q_i_c * camera_i_point + p_i_c;
        const Vector3 world_point = imu_state_map.at(first_state_id).rotation * imu_i_point +
                                    imu_state_map.at(first_state_id).position;

        return Vision::FeatureState(depth, world_point);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    double Helper::
    ResetFeatureDepth(const Vision::TrackMap& track_map,
                      const Vision::FeatureStateMap& feature_state_map,
                      const IMU::IMUStateMap& imu_state_map,
                      const Vision::TrackKey& track_key,
                      const Vision::Camera::ConstPtr& camera_ptr,
                      const Quaternion& q_i_c,
                      const Vector3& p_i_c)
    {
        const auto& world_point = feature_state_map.at(track_key).world_point;
        const auto& track = track_map.find(track_key)->second;
        const auto& measurements = track.measurements;
        const auto& first_state_id = measurements.begin()->state_id;
        const auto& q_w_i = imu_state_map.at(first_state_id).rotation;
        const auto& p_w_i = imu_state_map.at(first_state_id).position;

        const Vector3 imu_point    = q_w_i.inverse() *  (world_point - p_w_i);
        const Vector3 camera_point = q_i_c.inverse() * (imu_point - p_i_c);

        const double depth = camera_point(2);

        return depth;
    }
}//end of SuperVIO
