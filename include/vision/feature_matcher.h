//
// Created by chenghe on 3/10/20.
//

#ifndef SUPER_VIO_FEATURE_MATCHER_H
#define SUPER_VIO_FEATURE_MATCHER_H

#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/ccalib.hpp>
#include <ros/ros.h>
#include <vision/vision_measurements.h>
namespace SuperVIO::Vision
{

    class FeatureMatcher
    {
    public:
        typedef std::vector<cv::DMatch> Matches;

        //!knn match between train and query features
        static std::pair<double, Matches>
        Match(const FrameMeasurement& train_frame,
              const FrameMeasurement& query_frame,
              float match_thresh,
              float ratio_thresh,
              bool  f_check);

        //! remove outlier
        static Matches
        FCheck(const FrameMeasurement& train_frame,
               const FrameMeasurement& query_frame,
               const Matches& matches);

        static double
        ComputeParallax(const FrameMeasurement& train_frame,
                        const FrameMeasurement& query_frame,
                        const Matches& matches);


    };//end of FeatureMatcher

}//end of SuperVIO
#endif //SUPER_VIO_FEATURE_MATCHER_H
