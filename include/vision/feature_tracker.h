//
// Created by chenghe on 3/10/20.
//

#ifndef SUPER_VIO_FEATURE_TRACKER_H
#define SUPER_VIO_FEATURE_TRACKER_H

#include <vision/feature_matcher.h>
#include <utility/eigen_type.h>
#include <vision/vision_measurements.h>
namespace SuperVIO::Vision
{

    class FeatureTracker
    {
    public:
        class Parameters
        {
        public:
            explicit Parameters(float _match_thresh = 0.6,
                                float _ratio_thresh = 0.6,
                                double _parallax_thresh = 10,
                                bool _fundamental_filter = true);

            float match_thresh;
            float ratio_thresh;
            double parallax_thresh;
            bool fundamental_filter;
        };//end of Parameters

        class TrackingResult
        {
        public:
            TrackMap track_map;
            size_t num_matches;
            double parallax;
        };


        //! tracking between last frame and current frame
        static TrackingResult
        Tracking(const TrackMap& old_tracks_map,
                 const FrameMeasurement& last_frame,
                 const FrameMeasurement& new_frame,
                 const double& last_state_key,
                 const double& new_state_key,
                 const Parameters& parameters);

        //! creat empty tracks for a image
        static TrackMap
        CreatEmptyTrack(const FrameMeasurement& frame,
                        const StateKey& state_key);

    private:
        static size_t last_track_id_;
    };//end of FeatureTracker

}//end of SuperVIO

#endif //SUPER_VIO_FEATURE_TRACKER_H
