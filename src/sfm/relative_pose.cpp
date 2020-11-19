//
// Created by chenghe on 3/23/20.
//
#include <utility/eigen_base.h>
#include <sfm/relative_pose.h>
#include <opencv2/core/eigen.hpp>
#include <iostream>

namespace SuperVIO::SFM
{
    /////////////////////////////////////////////////////////////////////////////////////////////
    double RelativePose::
    ComputeHFRatio(const Vision::FrameMeasurement& frame_i,
                   const Vision::FrameMeasurement& frame_j,
                   const Matches& matches)
    {
        if(matches.empty())
        {
            return -1;
        }
        std::vector<cv::Point2f> points_i;
        std::vector<cv::Point2f> points_j;
        for(const auto& match: matches)
        {
            points_i.push_back(frame_i.key_points[match.point_i_id].point);
            points_j.push_back(frame_j.key_points[match.point_j_id].point);
        }
        //! 1. compute inliers of fundamental matrix
        std::vector<uchar> f_inliers;
        cv::findFundamentalMat(points_i, points_j, f_inliers, cv::FM_RANSAC);
        double num_f_inliers = 0;
        std::for_each(f_inliers.begin(), f_inliers.end(), [&](uchar in)
        {
            num_f_inliers += static_cast<double>(in);
        });
        //! 2. compute inliers of homography matrix
        std::vector<uchar> h_inliers;
        cv::findHomography(points_i, points_j, h_inliers, cv::RANSAC);
        double num_h_inliers = 0;
        std::for_each(h_inliers.begin(), h_inliers.end(), [&](uchar in)
        {
            num_h_inliers += static_cast<double>(in);
        });
        //! 3. compute h / f ratio
        double hf_ratio = num_h_inliers / num_f_inliers;

        return hf_ratio;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////
    Pose RelativePose::
    Evaluate(const Vision::FrameMeasurement& frame_i,
             const Vision::FrameMeasurement& frame_j,
             const Matches& matches,
             const CameraConstPtr& camera_ptr)
    {
        if(matches.size() <= 10)
        {
            return Pose(false);
        }

        std::vector<cv::Point2f> points_i;
        std::vector<cv::Point2f> points_j;
        for(const auto& match: matches)
        {
            points_i.push_back(frame_i.key_points[match.point_i_id].point);
            points_j.push_back(frame_j.key_points[match.point_j_id].point);
        }

        cv::Mat inliers;
        auto essential = cv::findEssentialMat(points_i, points_j, camera_ptr->GetIntrinsicMatrixCV(),
                cv::RANSAC, 0.99, 2, inliers);


        cv::Mat r, t;
        cv::recoverPose(essential, points_i, points_j, camera_ptr->GetIntrinsicMatrixCV(), r, t, inliers);
        double inlier_count = cv::sum(inliers)[0];
        bool success = false;
        if(inlier_count >= 20)
        {
            success = true;
        }

        Matrix3 rotation_matrix;
        Vector3 translation_vector;
        cv::cv2eigen(r, rotation_matrix);
        cv::cv2eigen(t, translation_vector);
        Quaternion quaternion(rotation_matrix);

        return Pose(success, quaternion.inverse(), -(quaternion.inverse() * translation_vector));
    }
}//end of SuperVIO