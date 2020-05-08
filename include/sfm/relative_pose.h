//
// Created by chenghe on 3/23/20.
//

#ifndef SUPER_VIO_RELATIVE_POSE_H
#define SUPER_VIO_RELATIVE_POSE_H

#include <vision/vision_measurements.h>
#include <vision/camera.h>
#include <sfm/match.h>
namespace SuperVIO::SFM
{

    class RelativePose
    {
    public:
        typedef Vision::Camera::ConstPtr CameraConstPtr;

        //! compute pairwise score
        static double ComputeHFRatio(const Vision::FrameMeasurement& frame_i,
                                     const Vision::FrameMeasurement& frame_j,
                                     const Matches& matches);

        //! compute pose from essential maxtrix
        static Pose
        Evaluate(const Vision::FrameMeasurement& frame_i,
                 const Vision::FrameMeasurement& frame_j,
                 const Matches& matches,
                 const CameraConstPtr& camera_ptr);

    };//end of RelativePose
}//end of SuperVIO

#endif //SUPER_VIO_RELATIVE_POSE_H
