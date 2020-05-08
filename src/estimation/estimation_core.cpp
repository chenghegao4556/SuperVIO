//
// Created by chenghe on 4/12/20.
//
#include <estimation/estimation_core.h>

#include <utility>
#include <random>

namespace SuperVIO::Estimation
{

    double RandomDepth(double init_depth)
    {
        std::random_device rd;
        std::default_random_engine generator(rd());
        std::normal_distribution<double> depth(init_depth, 1.0);

        return depth(generator);
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    typedef EstimationCore::ParameterBlockPtr ParameterBlockPtr;
    typedef EstimationCore::ResidualBlockPtr  ResidualBlockPtr;


    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    EstimationCore::
    EstimationCore(Parameters  parameters):
        parameters_(std::move(parameters))
    {
        feature_extractor_ = Vision::FeatureExtractor::Creat(parameters_.feature_extractor_parameters,
                parameters_.super_point_weight_path);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    EstimationCore::TrackSummery::
    TrackSummery(bool _valid_tracking,
                 bool _is_key_frame,
                 Vision::TrackMap  _track_map,
                 Vision::FrameMeasurement  _frame_measurement):
                 valid_tracking(_valid_tracking),
                 is_key_frame(_is_key_frame),
                 track_map(std::move(_track_map)),
                 frame_measurement(std::move(_frame_measurement))
    {

    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    VIOStatesMeasurements EstimationCore::
    Estimate(const VIOStatesMeasurements& last_states_measurement,
             const IMU::StateKey& state_key,
             const Vision::Image& image,
             const IMU::IMURawMeasurements& raw_measurements) const
    {
        auto new_states_measurements = last_states_measurement;

        //! 1. track image
        auto track_summery = this->TrackImage(new_states_measurements, state_key, image);

        //! 1.1 detect if lost
        if(!track_summery.valid_tracking)
        {
            ROS_ERROR_STREAM("feature tracking lost!!!");
            new_states_measurements.lost = true;
            return new_states_measurements;
        }
        else
        {
            //! 1.3 add new measurement

            //! release old descriptors to save memory
            new_states_measurements.lost = false;
            new_states_measurements.frame_measurement_map.rbegin()->second.descriptors.release();
            new_states_measurements.frame_measurement_map.insert(std::make_pair(state_key,
                    track_summery.frame_measurement));
            new_states_measurements.track_map = track_summery.track_map;
        }
        //! 2. process imu
        new_states_measurements = this->ProcessIMURawMeasurements(new_states_measurements, state_key, raw_measurements);

        //! 3. initial alignment
        if( !new_states_measurements.initialized &&
             static_cast<int>(new_states_measurements.imu_state_map.size()) >= parameters_.sliding_window_size)
        {
            auto initialized_states_measurements = this->InitialAlignment(new_states_measurements);
            if(initialized_states_measurements.first)
            {
                new_states_measurements = initialized_states_measurements.second;
            }
        }

        if(new_states_measurements.initialized)
        {
            //! 4. triangulate features
            new_states_measurements.feature_state_map = this->Triangulate(new_states_measurements);
            //! 5. optimize states
            new_states_measurements = this->Optimize(new_states_measurements);
            //! 6. marginalize
            new_states_measurements.marginalization_information = this->Marginalize(new_states_measurements,
                    track_summery.is_key_frame);
        }

        //! 7. remove frame outside sliding windows
//        if(new_states_measurements.imu_state_map.size() >= parameters_.sliding_window_size)
//        {
//
//        }
        new_states_measurements = this->RemoveMeasurementsAndStates(new_states_measurements,
                                                                    track_summery.is_key_frame);

        return new_states_measurements;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    VIOStatesMeasurements EstimationCore::
    InitializeVIOStatesMeasurements(const IMU::StateKey& state_key,
                                    const Vision::Image& image,
                                    const Vector3& acceleration_0,
                                    const Vector3& angular_velocity_0) const
    {
        auto frame_measurement = feature_extractor_->Compute(image);


        return this->InitializeVIOStatesMeasurements(state_key, IMU::IMUState(), frame_measurement,
                acceleration_0, angular_velocity_0, Vector3{0, 0, parameters_.gravity_norm});

    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    VIOStatesMeasurements EstimationCore::
    InitializeVIOStatesMeasurements(const IMU::StateKey& state_key,
                                    const IMU::IMUState& imu_state,
                                    const Vision::FrameMeasurement& frame_measurement,
                                    const Vector3& acceleration_0,
                                    const Vector3& angular_velocity_0,
                                    const Vector3& gravity_vector) const
    {
        VIOStatesMeasurements vio_states_measurements;
        vio_states_measurements.track_map = Vision::FeatureTracker::CreatEmptyTrack(frame_measurement, state_key);
        vio_states_measurements.frame_measurement_map.insert(std::make_pair(state_key, frame_measurement));

        vio_states_measurements.imu_state_map.insert(std::make_pair(state_key, imu_state));

        vio_states_measurements.acceleration_0     = acceleration_0;
        vio_states_measurements.angular_velocity_0 = angular_velocity_0;

        vio_states_measurements.gravity_vector = gravity_vector;

        return vio_states_measurements;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    EstimationCore::TrackSummery EstimationCore::
    TrackImage(const VIOStatesMeasurements& last_states_measurement,
               const IMU::StateKey& state_key,
               const Vision::Image& image) const
    {
        ROS_ASSERT(!last_states_measurement.frame_measurement_map.empty());

        //! 1. extract features
        auto frame_measurements = feature_extractor_->Compute(image);

        //! 2. track image
        auto track_result = Vision::FeatureTracker::Tracking(
                last_states_measurement.track_map,
                last_states_measurement.frame_measurement_map.rbegin()->second,
                frame_measurements,
                last_states_measurement.frame_measurement_map.rbegin()->first,
                state_key,
                parameters_.feature_tracker_parameters);

        //! 3. check validity
        size_t num_long_tracks = 0;
        for(const auto& track: track_result.track_map)
        {
            if(track.second.measurements.size() >= 4)
            {
                num_long_tracks ++;
            }
        }

        bool valid_tracking = last_states_measurement.frame_measurement_map.size() <= 4 ||
                             (static_cast<int>(track_result.num_matches) >= parameters_.min_num_match &&
                              num_long_tracks >= 20);

        bool is_key_frame = last_states_measurement.frame_measurement_map.size() < 2 ||
                             track_result.parallax > parameters_.parallax_thresh;

        return TrackSummery(valid_tracking, is_key_frame, track_result.track_map, frame_measurements);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    std::pair<bool, VIOStatesMeasurements> EstimationCore::
    InitialAlignment(const VIOStatesMeasurements& uninitialized_states_measurement) const
    {

        auto initialized_states_measurements = uninitialized_states_measurement;

        //! check imu measurement is fully excited?
        if(!IMU::IMUCheck::IsFullyExcited(initialized_states_measurements.imu_pre_integration_measurements_map))
        {
            return std::make_pair(false, VIOStatesMeasurements());
        }

        //! initial structure from motion
        auto sfm_result = SFM::InitialSFM::Construct(initialized_states_measurements.track_map,
            initialized_states_measurements.frame_measurement_map, parameters_.camera_ptr);

        if(!sfm_result.success)
        {
            return std::make_pair(false, VIOStatesMeasurements());
        }

        //! convert q_c0_ci to q_c0_bi
        for(auto& pose: sfm_result.frame_pose_map)
        {
            pose.second.rotation = pose.second.rotation * parameters_.q_i_c.inverse();
        }

        //! estimate gyroscope bias
        const Vector3 delta_bg = VisualIMUAligner::SolveGyroscopeBias(initialized_states_measurements,
                sfm_result.frame_pose_map);
        for(auto& imu_state: initialized_states_measurements.imu_state_map)
        {
            imu_state.second.linearized_bg += delta_bg;
        }

        //! repropagate raw measurements since guroscope bias is changed
        for(auto& pre: initialized_states_measurements.imu_pre_integration_measurements_map)
        {
            auto imu_state = initialized_states_measurements.imu_state_map.at(pre.first.first);
            const auto& imu_raw_measurements = initialized_states_measurements.imu_raw_measurements_map.at(pre.first);
            pre.second = IMU::IMUProcessor::Repropagate(imu_state.linearized_ba,
                                                        imu_state.linearized_bg,
                                                        pre.second.acceleration_0, pre.second.angular_velocity_0,
                                                        parameters_.noise, imu_raw_measurements);
        }
        auto aligned_result =  VisualIMUAligner::SolveGravityVectorAndVelocities(initialized_states_measurements,
                sfm_result.frame_pose_map, parameters_.p_i_c, parameters_.gravity_norm);

        if(!aligned_result.success)
        {
            return std::make_pair(false, VIOStatesMeasurements());
        }
        const double scale = aligned_result.scale;
        for(auto& imu_state: initialized_states_measurements.imu_state_map)
        {
            //! q_c0_bi
            imu_state.second.rotation = sfm_result.frame_pose_map.at(imu_state.first).rotation;
            //! p_c0_ci
            imu_state.second.position = sfm_result.frame_pose_map.at(imu_state.first).position;
        }
        const Quaternion& q_0 = initialized_states_measurements.imu_state_map.begin()->second.rotation;
        const Vector3& p_0 = initialized_states_measurements.imu_state_map.begin()->second.position;
        for(auto& imu_state: initialized_states_measurements.imu_state_map)
        {
            imu_state.second.position = scale * imu_state.second.position - imu_state.second.rotation * parameters_.p_i_c -
                    (scale * p_0 - q_0 * parameters_.p_i_c);
            //! v_c0_i = q_c0_bi * v_bi
            imu_state.second.velocity = imu_state.second.rotation * aligned_result.velocity_map.at(imu_state.first);
        }

        Quaternion q_world = Utility::EigenBase::GravityVector2Quaternion(aligned_result.gravity_vector);
        //yaw: R_w1_cl * R_cl_b0 = R_w1_b0.yaw = R_w1_w2
        const double yaw = Utility::EigenBase::Quaternion2Euler(q_world * q_0).x();
        //R0: R_w1_w2^-1 * R_w1_cl
        //R0: R_w2_cl
        q_world = Utility::EigenBase::EulerToQuaternion(Vector3{-yaw, 0, 0}) * q_world;
        //g = R_w2_cl * g_cl = g_w2
        initialized_states_measurements.gravity_vector = q_world * aligned_result.gravity_vector;
        for(auto& imu_state: initialized_states_measurements.imu_state_map)
        {
            imu_state.second.rotation = q_world * imu_state.second.rotation;
            imu_state.second.position = q_world * imu_state.second.position;
            imu_state.second.velocity = q_world * imu_state.second.velocity;
        }

        initialized_states_measurements.initialized = true;

        return std::make_pair(true, initialized_states_measurements);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    VIOStatesMeasurements EstimationCore::
    ProcessIMURawMeasurements(const VIOStatesMeasurements& last_states_measurement,
                              const IMU::StateKey& state_key,
                              const IMU::IMURawMeasurements& raw_measurements) const
    {
        auto new_states_measurements = last_states_measurement;

        const auto& last_state_key  = new_states_measurements.imu_state_map.rbegin()->first;
        const auto& last_imu_state  = new_states_measurements.imu_state_map.rbegin()->second;

        const Vector3 acceleration_0   = new_states_measurements.acceleration_0;
        const Vector3 angular_velocity_0 = new_states_measurements.angular_velocity_0;

        auto result = IMU::IMUProcessor::Propagate(last_imu_state,
                                                   acceleration_0,
                                                   angular_velocity_0,
                                                   last_states_measurement.gravity_vector,
                                                   parameters_.noise,
                                                   raw_measurements);

        auto state_pair_key = std::make_pair(last_state_key, state_key);

        new_states_measurements.imu_raw_measurements_map.insert(std::make_pair(state_pair_key, raw_measurements));
        new_states_measurements.imu_pre_integration_measurements_map.insert(std::make_pair(state_pair_key, result.second));
        new_states_measurements.imu_state_map.insert(std::make_pair(state_key, result.first));

        new_states_measurements.acceleration_0     = raw_measurements.rbegin()->acceleration;
        new_states_measurements.angular_velocity_0 = raw_measurements.rbegin()->angular_velocity;

        return new_states_measurements;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    Vision::FeatureStateMap  EstimationCore::
    Triangulate(const VIOStatesMeasurements& last_states_measurements) const
    {
        auto feature_state_map = last_states_measurements.feature_state_map;
        for(const auto& track: last_states_measurements.track_map)
        {
            auto iter = last_states_measurements.feature_state_map.find(track.first);
            if(track.second.measurements.size() >= 4 &&
               iter == last_states_measurements.feature_state_map.end())
            {
                Vector3s positions;
                Quaternions rotations;
                std::vector<cv::Point2f> image_points;

                for(const auto& measurement: track.second.measurements)
                {
                    rotations.push_back(last_states_measurements.imu_state_map.at(measurement.state_id).rotation);
                    positions.push_back(last_states_measurements.imu_state_map.at(measurement.state_id).position);
                    image_points.push_back(last_states_measurements.frame_measurement_map.at(measurement.state_id).
                            key_points[measurement.point_id].point);
                }

                auto result = Vision::Triangulator::TriangulatePoints(positions,
                        rotations, image_points, parameters_.q_i_c, parameters_.p_i_c,
                        parameters_.camera_ptr, parameters_.triangulator_parameters);

                if(result.status == Vision::Triangulator::Status::Success &&
                   result.depth > 0.1)
                {
                    Vision::FeatureState feature(result.depth, result.world_point);
                    feature_state_map.insert(std::make_pair(track.first, feature));
                }
                else
                {
                    const double depth = RandomDepth(5.0);
                    const Vector2 pt0{image_points.front().x, image_points.front().y};
                    const Vector3 camera_point = parameters_.camera_ptr->BackProject(pt0) * depth;
                    const Vector3 imu_point = parameters_.q_i_c * camera_point + parameters_.p_i_c;
                    const Vector3 world_point = rotations.front() * imu_point + positions.front();
                    Vision::FeatureState feature(depth, world_point);
                    feature_state_map.insert(std::make_pair(track.first, feature));
                }
            }
        }

        return feature_state_map;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    VIOStatesMeasurements EstimationCore::
    Optimize(const VIOStatesMeasurements& unoptimized_states_measurement) const
    {
        std::vector<ParameterBlockPtr> parameter_blocks;
        std::map<Vision::TrackKey, ParameterBlockPtr> feature_block_map;

        auto optimized_states_measurement = unoptimized_states_measurement;

//        std::map<Vision::TrackKey, double> reprojection_error_map;
//        for(const auto& feature_state: optimized_states_measurement.feature_state_map)
//        {
//            const Vector3& world_point = feature_state.second.world_point;
//            const auto& measurements = optimized_states_measurement.track_map.at(feature_state.first).measurements;
//            double error = 0.0;
//            for(const auto& measurement: measurements)
//            {
//                const Vector3& p_w_i = optimized_states_measurement.imu_state_map.at(measurement.state_id).position;
//                const Quaternion& q_w_i = optimized_states_measurement.imu_state_map.at(measurement.state_id).rotation;
//                const auto& pt = optimized_states_measurement.frame_measurement_map.at(measurement.state_id).
//                        key_points[measurement.point_id].point;
//                const Vector3 imu_point = q_w_i.inverse() * (world_point - p_w_i);
//                const Vector3 camera_point = parameters_.q_i_c.inverse() * (imu_point - parameters_.p_i_c);
//                error += (parameters_.camera_ptr->Project(camera_point) - Vector2{pt.x, pt.y}).norm();
//            }
//            error /= static_cast<double>(measurements.size());
//            reprojection_error_map.insert(std::make_pair(feature_state.first, error));
//        }


        //! 1. creat imu state parameter blocks
        for(const auto& imu_state: optimized_states_measurement.imu_state_map)
        {
            auto iter = optimized_states_measurement.pose_block_map.find(imu_state.first);
            if(iter == optimized_states_measurement.pose_block_map.end())
            {
                auto pose_ptr = Optimization::ParameterBlockFactory::CreatPose(
                        imu_state.second.rotation, imu_state.second.position);

                auto speed_bias_ptr = Optimization::ParameterBlockFactory::CreatSpeedBias(
                        imu_state.second.velocity, imu_state.second.linearized_ba, imu_state.second.linearized_bg);

                optimized_states_measurement.pose_block_map.insert(std::make_pair(imu_state.first, pose_ptr));
                optimized_states_measurement.speed_bias_block_map.insert(std::make_pair(imu_state.first, speed_bias_ptr));
            }
            parameter_blocks.push_back(optimized_states_measurement.pose_block_map.at(imu_state.first));
            parameter_blocks.push_back(optimized_states_measurement.speed_bias_block_map.at(imu_state.first));
        }

        //! 2. creat feature state parameter blocks
        for(auto& feature_state: optimized_states_measurement.feature_state_map)
        {
            feature_state.second.depth = Optimization::Helper::ResetFeatureDepth(
                    optimized_states_measurement.track_map, optimized_states_measurement.feature_state_map,
                    optimized_states_measurement.imu_state_map, feature_state.first,
                    parameters_.camera_ptr, parameters_.q_i_c, parameters_.p_i_c);
            auto feature_ptr = Optimization::ParameterBlockFactory::CreatInverseDepth(
                        feature_state.second.depth);
            feature_block_map.insert(std::make_pair(feature_state.first, feature_ptr));
            parameter_blocks.push_back(feature_ptr);
        }

        //! 3. creat reprojection residual blocks
        auto residual_blocks = Optimization::Helper::CreatReprojectionResidualBlocks(
                optimized_states_measurement.pose_block_map, feature_block_map,
                optimized_states_measurement.track_map, optimized_states_measurement.frame_measurement_map,
                parameters_.camera_ptr, parameters_.q_i_c, parameters_.p_i_c);

        //! 4. creat imu residual blocks
        for(const auto& pre: optimized_states_measurement.imu_pre_integration_measurements_map)
        {
            if(pre.second.sum_dt > parameters_.max_preintegration_time)
            {
                continue;
            }
            auto imu_residual_block = Optimization::Helper::CreatIMUResidualBlock(
                    optimized_states_measurement.pose_block_map,
                    optimized_states_measurement.speed_bias_block_map,
                    optimized_states_measurement.imu_pre_integration_measurements_map,
                    optimized_states_measurement.gravity_vector, pre.first);

            residual_blocks.push_back(imu_residual_block);
        }

        //! 5 add marginalization residual blocks
        if(optimized_states_measurement.marginalization_information.valid)
        {
            residual_blocks.push_back(Optimization::ResidualBlockFactory::CreatMarginalization(
                    optimized_states_measurement.marginalization_information));
        }
        //optimized_states_measurement.pose_block_map.begin()->second->SetFixed();

        //! 6. optimize
        Optimization::Optimizer::Construct(parameters_.optimizer_options,
                                           parameter_blocks, residual_blocks);
        const auto origin_first_rotation = optimized_states_measurement.imu_state_map.begin()->second.rotation;
        const auto origin_first_position = optimized_states_measurement.imu_state_map.begin()->second.position;
        const auto origin_first_rotation_euler = Utility::EigenBase::Quaternion2Euler(origin_first_rotation);

        const auto optimized_first_rotation = Optimization::Helper::GetPoseFromParameterBlock(
                optimized_states_measurement.pose_block_map.begin()->second).first;
        const auto optimized_first_position = Optimization::Helper::GetPoseFromParameterBlock(
                optimized_states_measurement.pose_block_map.begin()->second).second;
        const auto optimized_first_rotation_euler = Utility::EigenBase::Quaternion2Euler(optimized_first_rotation);

        const auto yaw_diff = origin_first_rotation_euler.x() - optimized_first_rotation_euler.x();

        auto rotation_diff = Utility::EigenBase::EulerToQuaternion(Vector3(yaw_diff, 0, 0));
        if (abs(abs(origin_first_rotation_euler.y()) - 90)    < 1.0 ||
            abs(abs(optimized_first_rotation_euler.y()) - 90) < 1.0)
        {
            ROS_ERROR_STREAM("euler singular point!");
            rotation_diff = origin_first_rotation * optimized_first_rotation.inverse();
        }
        //! update pose
        for(const auto& pose_ptr: optimized_states_measurement.pose_block_map)
        {
            auto pose = Optimization::Helper::GetPoseFromParameterBlock(pose_ptr.second);
            const Quaternion new_rotation = rotation_diff * pose.first;
            const Vector3    new_position = rotation_diff * (pose.second - optimized_first_position) + origin_first_position;
            optimized_states_measurement.imu_state_map.at(pose_ptr.first).rotation = new_rotation;
            optimized_states_measurement.imu_state_map.at(pose_ptr.first).position = new_position;
        }

        //! update speed bias
        for(const auto& speed_bias_ptr: optimized_states_measurement.speed_bias_block_map)
        {
            auto speed_bias = Optimization::Helper::GetSpeedBiasFromParameterBlock(speed_bias_ptr.second);
            const Vector3 new_velocity = rotation_diff * speed_bias.speed;
            optimized_states_measurement.imu_state_map.at(speed_bias_ptr.first).velocity =  new_velocity;
            optimized_states_measurement.imu_state_map.at(speed_bias_ptr.first).linearized_ba = speed_bias.ba;
            optimized_states_measurement.imu_state_map.at(speed_bias_ptr.first).linearized_bg = speed_bias.bg;
        }

        //! repropagate imu raw measurements, if need

        for(auto& pre: optimized_states_measurement.imu_pre_integration_measurements_map)
        {
            const Vector3& old_ba = pre.second.linearized_ba;
            const Vector3& new_ba = optimized_states_measurement.imu_state_map.at(pre.first.first).linearized_ba;

            const double ba_diff = (old_ba - new_ba).norm();

            const Vector3& old_bg = pre.second.linearized_bg;
            const Vector3& new_bg = optimized_states_measurement.imu_state_map.at(pre.first.first).linearized_bg;

            const double bg_diff = (old_bg - new_bg).norm();

            if(ba_diff > parameters_.linearized_ba_thresh || bg_diff > parameters_.linearized_bg_thresh)
            {
                const auto& imu_raw_measurements = optimized_states_measurement.imu_raw_measurements_map.at(pre.first);
                pre.second = IMU::IMUProcessor::Repropagate(new_ba, new_bg,
                        pre.second.acceleration_0, pre.second.angular_velocity_0,
                        parameters_.noise, imu_raw_measurements);
            }
        }

        //! update feature
        for(const auto& feature_ptr: feature_block_map)
        {
            auto new_feature = Optimization::Helper::UpdateFeatureState(optimized_states_measurement.track_map,
                    optimized_states_measurement.frame_measurement_map, optimized_states_measurement.imu_state_map,
                    feature_ptr.first, feature_ptr.second, parameters_.camera_ptr, parameters_.q_i_c, parameters_.p_i_c);
//
//            bool success = true;
//            double error = 0.0;
//            const auto& measurements = optimized_states_measurement.track_map.at(feature_ptr.first).measurements;
//            for(const auto& measurement: measurements)
//            {
//                const auto& state_i = optimized_states_measurement.imu_state_map.at(measurement.state_id);
//                const Vector3 imu_point = state_i.rotation.inverse() * (new_feature.world_point - state_i.position);
//                const Vector3 camera_point = parameters_.q_i_c.inverse() * (imu_point - parameters_.p_i_c);
//                const auto& pt = optimized_states_measurement.frame_measurement_map.at(measurement.state_id).
//                        key_points[measurement.point_id].point;
//                error += (parameters_.camera_ptr->Project(camera_point) - Vector2{pt.x, pt.y}).norm();
//                if(camera_point.z() < 0.0)
//                {
//                    success = false;
//                    break;
//                }
//            }
//            error /= static_cast<double>(measurements.size());
//            if(error > 2.0)
//            {
//                success = false;
//            }

            if(new_feature.depth > 0.0)
            {
                optimized_states_measurement.feature_state_map.at(feature_ptr.first) = new_feature;
            }
            else
            {
                //! if current track has new feature, consider new feature is wrong match
                if(optimized_states_measurement.track_map.at(feature_ptr.first).active)
                {
                    optimized_states_measurement.track_map.at(feature_ptr.first).measurements.pop_back();
                    optimized_states_measurement.track_map.at(feature_ptr.first).active = false;
                }
                else
                {
                    //! total tarck is wrong
                    optimized_states_measurement.track_map.erase(feature_ptr.first);
                    optimized_states_measurement.feature_state_map.erase(feature_ptr.first);
                }
            }
        }

        return optimized_states_measurement;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    Optimization::MarginalizationInformation EstimationCore::
    Marginalize(const VIOStatesMeasurements& states_measurements,
                bool is_new_frame_key_frame) const
    {

        typedef Optimization::BaseParametersBlock::Type ParameterType;

        auto marginalization_information = states_measurements.marginalization_information;
        std::map<Vision::TrackKey, ParameterBlockPtr> feature_block_map;

        std::vector<ParameterBlockPtr> parameter_blocks;
        std::vector<ResidualBlockPtr>  residual_blocks;
        std::set<size_t> drop_set;

        IMU::StateKey drop_imu_key;

        ParameterBlockPtr drop_speed_bias_ptr;
        ParameterBlockPtr drop_pose_ptr;

        if(is_new_frame_key_frame)
        {
            const auto imu_1 =  states_measurements.imu_state_map.begin();
            const auto imu_2 =  std::next(imu_1);
            drop_imu_key = imu_1->first;

            for(const auto& feature_state: states_measurements.feature_state_map)
            {
                if(states_measurements.track_map.at(feature_state.first).measurements.front().state_id
                   == imu_1->first)
                {
                    auto feature_ptr = Optimization::ParameterBlockFactory::CreatInverseDepth(
                            feature_state.second.depth);
                    feature_block_map.insert(std::make_pair(feature_state.first, feature_ptr));
                }
            }

            residual_blocks = Optimization::Helper::CreatReprojectionResidualBlocks(
                    states_measurements.pose_block_map, feature_block_map,
                    states_measurements.track_map, states_measurements.frame_measurement_map,
                    parameters_.camera_ptr, parameters_.q_i_c, parameters_.p_i_c);
            if(states_measurements.imu_pre_integration_measurements_map.at(
                    std::make_pair(imu_1->first, imu_2->first)).sum_dt < parameters_.max_preintegration_time)
            {
                auto imu_residual_block = Optimization::Helper::CreatIMUResidualBlock(states_measurements.pose_block_map,
                        states_measurements.speed_bias_block_map, states_measurements.imu_pre_integration_measurements_map,
                        states_measurements.gravity_vector, std::make_pair(imu_1->first, imu_2->first));

                residual_blocks.push_back(imu_residual_block);
            }
        }
        else
        {
            drop_imu_key = std::next(states_measurements.imu_state_map.rbegin())->first;
        }

        drop_speed_bias_ptr = states_measurements.speed_bias_block_map.at(drop_imu_key);
        drop_pose_ptr = states_measurements.pose_block_map.at(drop_imu_key);

        if(states_measurements.marginalization_information.valid)
        {
            residual_blocks.push_back(Optimization::ResidualBlockFactory::CreatMarginalization(
                    states_measurements.marginalization_information));
        }

        if(!residual_blocks.empty())
        {
            std::set<ParameterBlockPtr> parameter_blocks_set;
            for(const auto& residual_ptr: residual_blocks)
            {
                for(const auto& parameter_ptr: residual_ptr->GetParameterBlock())
                {
                    if(parameter_blocks_set.find(parameter_ptr) == parameter_blocks_set.end())
                    {
                        parameter_blocks_set.insert(parameter_ptr);
                    }
                }
            }

            size_t i = 0;
            for(const auto& parameter_ptr: parameter_blocks_set)
            {
                parameter_blocks.push_back(parameter_ptr);
                if(parameter_ptr->GetType() == ParameterType::InverseDepth               ||
                   parameter_ptr == drop_pose_ptr  || parameter_ptr == drop_speed_bias_ptr)
                {
                    drop_set.insert(i);
                }
                ++i;
            }
            if(drop_set.empty())
            {
                return marginalization_information;
            }

            marginalization_information = Optimization::Marginalizer::Construct(
                    parameter_blocks, residual_blocks, drop_set, parameters_.optimizer_options.num_threads);
        }

        return marginalization_information;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    VIOStatesMeasurements EstimationCore::
    RemoveMeasurementsAndStates(const VIOStatesMeasurements& states_measurements,
                                bool is_new_frame_key_frame)const
    {
        if(is_new_frame_key_frame &&
           static_cast<int>(states_measurements.imu_state_map.size())<= parameters_.sliding_window_size)
        {
            return states_measurements;
        }

        VIOStatesMeasurements new_states_measurements(states_measurements);
        IMU::StateKey drop_imu_id;
        std::vector<IMU::StatePairKey> drop_pairs;

        if(is_new_frame_key_frame)
        {
            auto imu_1 = states_measurements.imu_state_map.begin();
            auto imu_2 = std::next(imu_1);
            drop_imu_id = imu_1->first;
            drop_pairs.emplace_back(imu_1->first, imu_2->first);
        }
        else
        {
            const auto imu_3 = states_measurements.imu_state_map.rbegin();
            const auto imu_2 = std::next(imu_3);
            const auto imu_1 = std::next(imu_2);
            drop_imu_id = imu_2->first;

            auto raw_13 = new_states_measurements.imu_raw_measurements_map.at(
                    std::make_pair(imu_1->first, imu_2->first));
            auto raw_23 = new_states_measurements.imu_raw_measurements_map.at(
                    std::make_pair(imu_2->first, imu_3->first));
            for(const auto& raw: raw_23)
            {
                raw_13.push_back(raw);
            }
            auto pre_12 = new_states_measurements.imu_pre_integration_measurements_map.at(
                    std::make_pair(imu_1->first, imu_2->first));
            IMU::IMUPreIntegrationMeasurement pre_13(pre_12.acceleration_0, pre_12.angular_velocity_0,
                    imu_1->second.linearized_ba, imu_1->second.linearized_bg);
            pre_13 = IMU::PreIntegrator::ComputeAll(pre_13, pre_12.acceleration_0, pre_12.angular_velocity_0,
                    parameters_.noise, raw_13);
            new_states_measurements.imu_raw_measurements_map.insert(std::make_pair(
                    std::make_pair(imu_1->first, imu_3->first), raw_13));
            new_states_measurements.imu_pre_integration_measurements_map.insert(std::make_pair(
                    std::make_pair(imu_1->first, imu_3->first), pre_13));

            drop_pairs.emplace_back(imu_1->first, imu_2->first);
            drop_pairs.emplace_back(imu_2->first, imu_3->first);
        }

        new_states_measurements.imu_state_map.erase(drop_imu_id);
        new_states_measurements.pose_block_map.erase(drop_imu_id);
        new_states_measurements.speed_bias_block_map.erase(drop_imu_id);
        new_states_measurements.frame_measurement_map.erase(drop_imu_id);

        for(const auto& drop_pair: drop_pairs)
        {
            new_states_measurements.imu_raw_measurements_map.erase(drop_pair);
            new_states_measurements.imu_pre_integration_measurements_map.erase(drop_pair);
        }

        std::vector<Vision::TrackKey> drop_track_ids;
        for(const auto& track: new_states_measurements.track_map)
        {
            //! remove measurement
            Vision::TrackMeasurements measurements;
            for(const auto& measurement : track.second.measurements)
            {
                if(measurement.state_id != drop_imu_id)
                {
                    measurements.push_back(measurement);
                }
            }
            new_states_measurements.track_map.at(track.first).measurements = measurements;

            //! need at least 4 measurements
            if( track.second.measurements.size() < 4 && (!track.second.active))
            {
                drop_track_ids.push_back(track.first);
            }
        }
        for(const auto& drop_track_id: drop_track_ids)
        {
            new_states_measurements.track_map.erase(drop_track_id);
            new_states_measurements.feature_state_map.erase(drop_track_id);
        }

        return new_states_measurements;
    }


}//end of SuperVIO::Estimation
