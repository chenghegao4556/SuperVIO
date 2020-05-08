//
// Created by chenghe on 3/23/20.
//

#ifndef SUPER_VIO_ABSOLUTE_POSE_H
#define SUPER_VIO_ABSOLUTE_POSE_H

#include <sfm/match.h>
#include <vision/camera.h>
#include <vision/vision_measurements.h>

namespace SuperVIO::SFM
{
    class AbsolutePose
    {
    public:
        typedef Vision::Camera::ConstPtr CameraConstPtr;

        /**
         * @brief solve pnp problem
         */
        static Pose
        Evaluate(const Vision::StateKey& state_key,
                 const Vision::FrameMeasurement& frame_measurement,
                 const Vision::TrackMap& track_map,
                 const Vision::FeatureStateMap& feature_map,
                 const CameraConstPtr& camera_ptr);
    };//end of AbsolutePose
}//end of SuperVIO

#endif //SUPER_VIO_ABSOLUTE_POSE_H
