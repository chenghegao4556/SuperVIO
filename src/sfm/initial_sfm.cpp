//
// Created by chenghe on 3/23/20.
//
#include <sfm/initial_sfm.h>
#include <fstream>

namespace SuperVIO::SFM
{
    /////////////////////////////////////////////////////////////////////////////////////////////
    InitialSFM::SFMResult::
    SFMResult():
        success(false)
    {

    }

    /////////////////////////////////////////////////////////////////////////////////////////////
    InitialSFM::SFMResult InitialSFM::
    Construct(const Vision::TrackMap& track_map,
              const Vision::FrameMeasurementMap& frame_measurement_map,
              const CameraConstPtr& camera_ptr)
    {

        const auto& start_frame_key = frame_measurement_map.begin()->first;

        auto best = FindBestFrame(track_map, frame_measurement_map, camera_ptr);
        if(!best.first)
        {
            return SFMResult();
        }
        auto best_frame_key = best.second.first;

        //! initialize pose start frame and best frame
        auto relative_pose = RelativePose::Evaluate(frame_measurement_map.at(start_frame_key),
                                                    frame_measurement_map.at(best_frame_key),
                                                    best.second.second,
                                                    camera_ptr);
        if(!relative_pose.success)
        {
            return SFMResult();
        }

        SFMResult sfm_result;
        sfm_result.frame_pose_map.insert(std::make_pair(
                start_frame_key, Pose(true, Quaternion::Identity(), Vector3::Zero())));
        sfm_result.frame_pose_map.insert(std::make_pair(
                best_frame_key,  relative_pose));

        sfm_result.feature_state_map = TriangulateTracks(track_map, frame_measurement_map,
                 sfm_result.frame_pose_map, camera_ptr);

        //! solve pnp and triangulate two frame
        for(size_t i = 0; i < 1; ++i)
        {
            for(auto& frame_measurement: frame_measurement_map)
            {
                /**
                auto iter = sfm_result.frame_pose_map.find(frame_measurement.first);
                if(iter != sfm_result.frame_pose_map.end())
                {
                    continue;
                }
                 **/
                if(frame_measurement.first != frame_measurement_map.begin()->first ||
                   frame_measurement.first != best_frame_key)
                {
                    auto absolute_pose = AbsolutePose::Evaluate(frame_measurement.first,
                            frame_measurement.second, track_map, sfm_result.feature_state_map, camera_ptr);
                    if(absolute_pose.success)
                    {
                        sfm_result.frame_pose_map.insert(std::make_pair(
                                frame_measurement.first,  absolute_pose));

                        sfm_result.feature_state_map = TriangulateTracks(track_map, frame_measurement_map,
                                                                         sfm_result.frame_pose_map, camera_ptr);
                    }
                }
            }
        }
        if(sfm_result.frame_pose_map.size() != frame_measurement_map.size())
        {
            return SFMResult();
        }
        sfm_result.feature_state_map = TriangulateTracks(track_map, frame_measurement_map,
                 sfm_result.frame_pose_map, camera_ptr);
        sfm_result.success = true;

        return  OptimizeSFM(track_map, frame_measurement_map, sfm_result, camera_ptr);
    }

    /////////////////////////////////////////////////////////////////////////////////////////////
    InitialSFM::SFMResult InitialSFM::
    OptimizeSFM(const Vision::TrackMap& track_map,
                const Vision::FrameMeasurementMap& frame_measurement_map,
                const SFMResult& old_sfm_result,
                const CameraConstPtr& camera_ptr)
    {

        typedef Optimization::BaseParametersBlock::Ptr ParameterBlockPtr;

        std::map<Vision::StateKey, ParameterBlockPtr> pose_block_map;
        std::map<Vision::TrackKey, ParameterBlockPtr> feature_block_map;
        for(const auto& feature_state: old_sfm_result.feature_state_map)
        {
            if(track_map.at(feature_state.first).measurements.size() >= 3)
            {
                auto feature_ptr = Optimization::ParameterBlockFactory::CreatInverseDepth(
                        feature_state.second.depth);

                feature_block_map.insert(std::make_pair(feature_state.first, feature_ptr));
            }
        }

        for(const auto& frame_pose: old_sfm_result.frame_pose_map)
        {
            auto pose_ptr = Optimization::ParameterBlockFactory::CreatPose(
                    frame_pose.second.rotation, frame_pose.second.position);

            pose_block_map.insert(std::make_pair(frame_pose.first, pose_ptr));
        }

        auto residual_blocks = Optimization::Helper::CreatReprojectionResidualBlocks(
                pose_block_map, feature_block_map, track_map, frame_measurement_map,
                camera_ptr, Quaternion::Identity(), Vector3::Zero());

        std::vector<ParameterBlockPtr> parameter_blocks;
        for(const auto& feature_ptr: feature_block_map)
        {
            parameter_blocks.push_back(feature_ptr.second);
        }
        for(const auto& pose_ptr: pose_block_map)
        {
            parameter_blocks.push_back(pose_ptr.second);
        }
        pose_block_map.begin()->second->SetFixed();
        feature_block_map.begin()->second->SetFixed();
        Optimization::Optimizer::Options options;
        Optimization::Optimizer::Construct(options, parameter_blocks, residual_blocks);

        SFMResult optimized_sfm_result;
        optimized_sfm_result.success = true;
        for(const auto& pose_ptr: pose_block_map)
        {
            auto pose = Optimization::Helper::GetPoseFromParameterBlock(pose_ptr.second);
            optimized_sfm_result.frame_pose_map.insert(std::make_pair(pose_ptr.first, Pose(
                    true, pose.first, pose.second)));
        }

        for(const auto& feature_ptr: feature_block_map)
        {
            const auto depth = Optimization::Helper::GetDepthFromParameterBlock(feature_ptr.second);
            const auto& measurement_0 = track_map.at(feature_ptr.first).measurements.front();
            const auto& pt0 = frame_measurement_map.at(measurement_0.state_id).key_points[measurement_0.point_id].point;
            const auto& rotation0 = optimized_sfm_result.frame_pose_map.at(measurement_0.state_id).rotation;
            const auto& position0 = optimized_sfm_result.frame_pose_map.at(measurement_0.state_id).position;
            const Vector3 camera_point = camera_ptr->BackProject(Vector2{pt0.x, pt0.y}) * depth;
            const Vector3 world_point  = rotation0 * camera_point + position0;
            optimized_sfm_result.feature_state_map.insert(std::make_pair(feature_ptr.first,
                    Vision::FeatureState(depth, world_point)));
        }

        return optimized_sfm_result;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////
    std::pair<bool, std::pair<Vision::StateKey, Matches>> InitialSFM::
    FindBestFrame(const Vision::TrackMap& track_map,
                  const Vision::FrameMeasurementMap& frame_measurement_map,
                  const CameraConstPtr &camera_ptr)
    {
        //! find start frame index and end frame index
        const auto& start_frame_key = frame_measurement_map.begin()->first;
        const auto& end_frame_key   = frame_measurement_map.rbegin()->first;

        auto matches_map = ConstructPairwiseMatches(track_map, frame_measurement_map);
        auto parallax_map = ComputePairwiseParallax(track_map, frame_measurement_map, matches_map);
        auto max_parallax = FindMaxParallax(parallax_map);
        auto max_num_matches = FindMaxNumMatches(matches_map);
        auto scores = ComputeScore(parallax_map, matches_map, max_parallax, max_num_matches);

        //! check validation
        auto best_frame_index = end_frame_key;
        bool found = false;
        for(const auto& score: scores)
        {
            double hf_ratio = RelativePose::ComputeHFRatio(frame_measurement_map.at(start_frame_key),
                                                           frame_measurement_map.at(score.first),
                                                           matches_map.at(score.first));
            if(hf_ratio < 1.0)
            {
                best_frame_index = score.first;
                found = true;
                break;
            }
        }

        return std::make_pair(found, std::make_pair(best_frame_index, matches_map.at(best_frame_index)));
    }

    /////////////////////////////////////////////////////////////////////////////////////////////
    std::map<Vision::StateKey, Matches> InitialSFM::
    ConstructPairwiseMatches(const Vision::TrackMap& track_map,
                             const Vision::FrameMeasurementMap& frame_measurement_map)
    {
        //! construct pair wise matches
        const auto& start_frame_key = frame_measurement_map.begin()->first;
        std::map<Vision::StateKey, Matches> matches_map;
        for(const auto& track: track_map)
        {
            if(track.second.measurements.size() < 3)
            {
                continue;
            }
            if(track.second.measurements.begin()->state_id != start_frame_key ||
               track.second.measurements.size() < 2)
            {
                continue;
            }
            const auto& point_i_id = track.second.measurements.begin()->point_id;
            for(size_t j = 1; j < track.second.measurements.size(); ++j)
            {
                const auto& point_j_id = track.second.measurements[j].point_id;
                const auto& frame_state_key_j = track.second.measurements[j].state_id;

                auto iter = matches_map.find(frame_state_key_j);
                if(iter == matches_map.end())
                {
                    Matches matches;
                    matches.emplace_back(point_i_id, point_j_id, track.first);
                    matches_map.insert(std::make_pair(frame_state_key_j, matches));
                }
                else
                {
                    iter->second.emplace_back(point_i_id, point_j_id, track.first);
                }
            }
        }

        return matches_map;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////
    std::map<Vision::StateKey, double> InitialSFM::
    ComputePairwiseParallax(const Vision::TrackMap& track_map,
                            const Vision::FrameMeasurementMap& frame_measurement_map,
                            const std::map<Vision::StateKey, Matches>& matches_map)
    {
        const auto& start_frame_key = frame_measurement_map.begin()->first;
        std::map<Vision::StateKey, double> parallax_map;
        for(const auto& matches: matches_map)
        {
            std::vector<cv::DMatch> cv_matches;
            for(const auto& match: matches.second)
            {
                cv_matches.emplace_back(match.point_j_id, match.point_i_id, 0);
            }

            double parallax = Vision::FeatureMatcher::
            ComputeParallax(frame_measurement_map.at(start_frame_key),
                            frame_measurement_map.at(matches.first),
                            cv_matches);

            parallax_map.insert(std::make_pair(matches.first,parallax));
        }

        return parallax_map;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////
    double InitialSFM::
    FindMaxParallax(const std::map<Vision::StateKey, double>& parallax_map)
    {
        double max_parallax = 0;
        for(const auto& parallax: parallax_map)
        {
            if(parallax.second > max_parallax)
            {
                max_parallax = parallax.second;
            }
        }

        return max_parallax;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////
    size_t  InitialSFM::
    FindMaxNumMatches(const std::map<Vision::StateKey, Matches>& matches_map)
    {
        size_t max_num_matches = 0;
        for(const auto& matches: matches_map)
        {
            if(matches.second.size() > max_num_matches)
            {
                max_num_matches = matches.second.size();
            }
        }

        return max_num_matches;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////
    std::vector<std::pair<Vision::StateKey, double>> InitialSFM::
    ComputeScore(const std::map<Vision::StateKey, double>& parallax_map,
                 const std::map<Vision::StateKey, Matches>& matches_map,
                 double max_parallax,
                 size_t max_num_matches)
    {
        std::vector<std::pair<Vision::StateKey, double>> scores;
        for(auto& parallax: parallax_map)
        {
            size_t num_matches = matches_map.at(parallax.first).size();
            double score = parallax.second / max_parallax +
                           static_cast<double>(num_matches) / static_cast<double>(max_num_matches);
            scores.emplace_back(parallax.first, score);
        }

        //! sort score
        std::sort(scores.begin(), scores.end(),
                  [&](std::pair<Vision::StateKey, double> p1, std::pair<size_t, double> p2)
                  {
                      return p1.second > p2.second;
                  });

        return scores;

    }

    /////////////////////////////////////////////////////////////////////////////////////////////
    Vision::FeatureStateMap InitialSFM::
    TriangulateTracks(const Vision::TrackMap& track_map,
                      const Vision::FrameMeasurementMap& frame_measurement_map,
                      const std::map<Vision::StateKey, Pose>& frame_pose_map,
                      const CameraConstPtr& camera_ptr)
    {
        Vision::FeatureStateMap new_feature_state_map;
        Vision::Triangulator::Parameters tri_param(false, true, 0, 0, 2.0);
        for(auto& track: track_map)
        {
            if(track.second.measurements.size() <2)
            {
                continue;
            }

            Vector3s positions;
            Quaternions rotations;
            std::vector<cv::Point2f> image_points;
            for(const auto& measurement: track.second.measurements)
            {
                auto iter = frame_pose_map.find(measurement.state_id);
                if(iter != frame_pose_map.end())
                {
                    rotations.push_back(iter->second.rotation);
                    positions.push_back(iter->second.position);
                    image_points.push_back(frame_measurement_map.at(measurement.state_id).
                           key_points[measurement.point_id].point);
                }
            }
            if(rotations.size() < 2)
            {
                continue;
            }
            auto result = Vision::Triangulator::TriangulatePoints(positions,
                                                                  rotations,
                                                                  image_points,
                                                                  Quaternion::Identity(),
                                                                  Vector3::Zero(),
                                                                  camera_ptr,
                                                                  tri_param);
            if(result.status == Vision::Triangulator::Status::Success)
            {
                new_feature_state_map.insert(std::make_pair(track.first, Vision::FeatureState(result.depth,
                        result.world_point)));
            }
        }

        return new_feature_state_map;
    }

}//end of SuperVIO