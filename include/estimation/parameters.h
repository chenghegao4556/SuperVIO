//
// Created by chenghe on 4/29/20.
//

#ifndef SUPER_VIO_PARAMETERS_H
#define SUPER_VIO_PARAMETERS_H
#include <imu/imu_noise.h>
#include <vision/feature_extractor.h>
#include <vision/feature_tracker.h>
#include <vision/triangulator.h>
#include <optimization/optimizer.h>
namespace SuperVIO::Estimation
{
    class Parameters
    {
    public:
        Parameters();

        bool Load(const ros::NodeHandle& nh_private);

        std::string super_point_weight_path;
        Vision::FeatureExtractor::Parameters feature_extractor_parameters;
        Vision::FeatureTracker::Parameters   feature_tracker_parameters;
        Vision::Triangulator::Parameters triangulator_parameters;
        Optimization::Optimizer::Options optimizer_options;
        int min_num_match;
        int sliding_window_size;
        double parallax_thresh;
        double linearized_ba_thresh;
        double linearized_bg_thresh;
        double max_preintegration_time;

        double accelerator_noise;
        double gyroscope_noise;
        double accelerator_random_walk_noise;
        double gyroscope_random_walk_noise;

        double gravity_norm;

        Vision::Camera::ConstPtr camera_ptr;
        Vector3 p_i_c;
        Quaternion q_i_c;

        Matrix18 noise;
    };//end of Parameters
}//end of SuperVIO::Estimation

#endif //SUPER_VIO_PARAMETERS_H
