//
// Created by chenghe on 3/23/20.
//

#ifndef SUPER_VIO_INITIAL_SFM_H
#define SUPER_VIO_INITIAL_SFM_H

#include <vision/vision_measurements.h>
#include <vision/triangulator.h>
#include <sfm/relative_pose.h>
#include <sfm/absolute_pose.h>
#include <sfm/match.h>
#include <optimization/helper.h>
#include <optimization/optimizer.h>
namespace SuperVIO::SFM
{
    class InitialSFM {
    public:
        typedef Vision::Camera::ConstPtr CameraConstPtr;

        class SFMResult
        {
        public:
            explicit SFMResult();
            bool success;
            std::map<Vision::StateKey, Pose> frame_pose_map;
            Vision::FeatureStateMap feature_state_map;
        };


        /**
         * @brief try to get structure from motion
         * @param[in/out] tracks
         * @param[in/out] frames
         * @param[in] camera_ptr
         */
        static SFMResult
        Construct(const Vision::TrackMap& track_map,
                  const Vision::FrameMeasurementMap& frame_measurement_map,
                  const CameraConstPtr& camera_ptr);

    protected:

        /**
         * @brief traditional bundle adjustment
         */
        static SFMResult
        OptimizeSFM(const Vision::TrackMap& track_map,
                    const Vision::FrameMeasurementMap& frame_measurement_map,
                    const SFMResult& old_sfm_result,
                    const CameraConstPtr& camera_ptr);

        /**
         * @brief find a frame in frame database with largest score
         */
        static std::pair<bool, std::pair<Vision::StateKey, Matches>>
        FindBestFrame(const Vision::TrackMap& track_map,
                      const Vision::FrameMeasurementMap& frame_measurement_map,
                      const CameraConstPtr &camera_ptr);

        /**
         * @brief convert tracks to matches((frame 0, frame 1), (frame 0, frame 2) .....)
         */
        static std::map<Vision::StateKey, Matches>
        ConstructPairwiseMatches(const Vision::TrackMap& track_map,
                                 const Vision::FrameMeasurementMap& frame_measurement_map);

        /**
         * @brief compute motion parallax of all frame pairs
         */
        static std::map<Vision::StateKey, double>
        ComputePairwiseParallax(const Vision::TrackMap& track_map,
                                const Vision::FrameMeasurementMap& frame_measurement_map,
                                const std::map<Vision::StateKey, Matches>& matches_map);

        /**
         * @brief find max element in parallax_map
         */
        static double FindMaxParallax(const std::map<Vision::StateKey, double>& parallax_map);

        /**
         * @brief find max element in matches_map
         */
        static size_t
        FindMaxNumMatches(const std::map<Vision::StateKey, Matches>& matches_map);

        /**
         * @brief compute pairwise scores: parallax / max_parallax + num_matches/ max_num_matches
         */
        static std::vector<std::pair<Vision::StateKey, double>>
        ComputeScore(const std::map<Vision::StateKey, double>& parallax_map,
                     const std::map<Vision::StateKey, Matches>& matches_map,
                     double max_parallax,
                     size_t max_num_matches);

        /**
         * @brief triangulate one track
         */
        static  Vision::FeatureStateMap
        TriangulateTracks(const Vision::TrackMap& track_map,
                          const Vision::FrameMeasurementMap& frame_measurement_map,
                          const std::map<Vision::StateKey, Pose>& frame_pose_map,
                          const CameraConstPtr& camera_ptr);
    };//end of InitialSFM
}//end of SuperVIO

#endif //SUPERVIO_INITIAL_SFM_H
