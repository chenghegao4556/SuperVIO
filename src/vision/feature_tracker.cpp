//
// Created by chenghe on 3/10/20.
//
#include <vision/feature_tracker.h>

namespace SuperVIO::Vision
{

    /////////////////////////////////////////////////////////////////////////////////////////////
    size_t FeatureTracker::last_track_id_ = 0;

    /////////////////////////////////////////////////////////////////////////////////////////////
    FeatureTracker::Parameters::
    Parameters(float _match_thresh,
               float _ratio_thresh,
               double _parallax_thresh,
               bool _fundamental_filter):
               match_thresh(_match_thresh),
               ratio_thresh(_ratio_thresh),
               parallax_thresh(_parallax_thresh),
               fundamental_filter(_fundamental_filter)
    {

    }

    /////////////////////////////////////////////////////////////////////////////////////////////
    FeatureTracker::TrackingResult FeatureTracker::
    Tracking(const TrackMap& old_tracks_map,
             const FrameMeasurement& last_frame,
             const FrameMeasurement& new_frame,
             const double& last_state_key,
             const double& new_state_key,
             const Parameters& parameters)
    {
        TrackingResult tracking_result;
        tracking_result.track_map = old_tracks_map;
        std::map<size_t, size_t> point_track_map;

        for(const auto& track: tracking_result.track_map)
        {
            if(track.second.active)
            {
                auto point_id = track.second.measurements.back().point_id;
                auto iter = point_track_map.find(point_id);
                ROS_ASSERT(iter == point_track_map.end());
                point_track_map[point_id] = track.first;
            }
        }

        auto result = FeatureMatcher::Match(last_frame,
                                            new_frame,
                                            parameters.match_thresh,
                                            parameters.ratio_thresh,
                                            parameters.fundamental_filter);

        tracking_result.parallax = result.first;
        tracking_result.num_matches = result.second.size();

        const auto& matches = result.second;
        std::set<size_t> tracked_feature_ids;
        std::set<size_t> tracked_points_id;

        //! add new measurement to each track
        for(const auto& match: matches)
        {
            auto track_id_iter = point_track_map.find(match.trainIdx);
            if(track_id_iter == point_track_map.end())
            {
                Track track;
                track.measurements.emplace_back(last_state_key, match.trainIdx);
                track.measurements.emplace_back(new_state_key, match.queryIdx);
                tracking_result.track_map.insert(std::make_pair(last_track_id_, track));

                last_track_id_++;
                continue;
            }
            auto track_iter = tracking_result.track_map.find(track_id_iter->second);
            ROS_ASSERT(track_iter != tracking_result.track_map.end());
            track_iter->second.measurements.emplace_back(new_state_key, match.queryIdx);

            auto tracked_feature_iter = tracked_feature_ids.find(track_id_iter->second);
            ROS_ASSERT(tracked_feature_iter == tracked_feature_ids.end());
            tracked_feature_ids.insert(track_id_iter->second);

            auto tracked_point_iter = tracked_points_id.find(match.queryIdx);
            ROS_ASSERT(tracked_point_iter == tracked_points_id.end());
            tracked_points_id.insert(match.queryIdx);
        }

        //! dis-active all lost tracks
        for(auto& track: tracking_result.track_map)
        {
            if(track.second.active)
            {
                auto iter = tracked_feature_ids.find(track.first);
                if(iter == tracked_feature_ids.end())
                {
                    track.second.active = false;
                }
            }
        }

        //!creat new track for new detected features
        for(size_t i = 0; i < new_frame.key_points.size(); ++i)
        {
            auto iter = tracked_points_id.find(i);
            if(iter == tracked_points_id.end())
            {
                Track track;
                track.measurements.emplace_back(new_state_key, i);

                tracking_result.track_map.insert(std::make_pair(last_track_id_, track));

                last_track_id_++;
            }
        }

        //! delete track has only one measurement
        for(auto& track: tracking_result.track_map)
        {
            if(!track.second.active && track.second.measurements.size() == 1)
            {
                tracking_result.track_map.erase(track.first);
            }
        }

        return tracking_result;

    }

    /////////////////////////////////////////////////////////////////////////////////////////////
    TrackMap FeatureTracker::
    CreatEmptyTrack(const FrameMeasurement& frame,
                    const StateKey& state_key)
    {
        TrackMap tracks;
        for(size_t i = 0; i < frame.key_points.size(); ++i)
        {
            Track track;
            track.measurements.emplace_back(state_key, i);
            tracks.insert(std::make_pair(last_track_id_, track));

            last_track_id_++;
        }

        return tracks;
    }


}//end of SuperVIO
