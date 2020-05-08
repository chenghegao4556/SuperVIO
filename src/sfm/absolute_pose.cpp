//
// Created by chenghe on 3/23/20.
//
#include <sfm/absolute_pose.h>
#include <opencv2/core/eigen.hpp>
namespace SuperVIO::SFM
{
    /////////////////////////////////////////////////////////////////////////////////////////////
    Pose AbsolutePose::
    Evaluate(const Vision::StateKey& state_key,
             const Vision::FrameMeasurement& frame_measurement,
             const Vision::TrackMap& track_map,
             const Vision::FeatureStateMap& feature_map,
             const CameraConstPtr& camera_ptr)
    {
        std::vector<cv::Point2f> point_2d;
        std::vector<cv::Point3d> point_3d;
        for(const auto& track: track_map)
        {
            auto iter = feature_map.find(track.first);
            if(iter != feature_map.end())
            {
                for(const auto& measurement: track.second.measurements)
                {
                    if(measurement.state_id == state_key)
                    {
                        point_2d.push_back(frame_measurement.key_points[measurement.point_id].point);
                        point_3d.emplace_back(iter->second.world_point(0),
                                              iter->second.world_point(1),
                                              iter->second.world_point(2));

                        break;
                    }
                }
            }

        }

        if(point_2d.size() <= 10)
        {
            return Pose(false);
        }

        std::vector<uchar> inliers;
        cv::Mat cv_R_c_w, cv_r_c_w, cv_p_c_w;
        cv::solvePnPRansac(point_3d, point_2d, camera_ptr->GetIntrinsicMatrixCV(), cv::Mat(),
                           cv_r_c_w, cv_p_c_w, false, 100, 2, 0.99, inliers, cv::SOLVEPNP_AP3P);

        cv::Rodrigues ( cv_r_c_w, cv_R_c_w );
        Matrix3 eigen_r_c_w;
        Vector3 eigen_p_c_w;
        cv2eigen(cv_R_c_w, eigen_r_c_w);
        cv2eigen(cv_p_c_w, eigen_p_c_w);

        Quaternion q_w_c(eigen_r_c_w.inverse());
        Vector3    t_w_c = -eigen_r_c_w.inverse() * eigen_p_c_w;

        return Pose(true, q_w_c, t_w_c);
    }
}//end of SuperVIO
