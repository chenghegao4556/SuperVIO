//
// Created by chenghe on 4/29/20.
//
#include <estimation/parameters.h>
namespace SuperVIO::Estimation
{
    Parameters::
    Parameters():
            min_num_match(40),
            sliding_window_size(10),
            parallax_thresh(10.0),
            linearized_ba_thresh(1.0),
            linearized_bg_thresh(1.0),
            max_preintegration_time(10.0),
            accelerator_noise(0.01),
            gyroscope_noise(0.01),
            accelerator_random_walk_noise(0.01),
            gyroscope_random_walk_noise(0.01),
            gravity_norm(9.81),
            camera_ptr(nullptr),
            p_i_c(Vector3::Zero()),
            q_i_c(Quaternion::Identity()),
            noise(Matrix18::Zero())
    {

    }

    bool Parameters::
    Load(const ros::NodeHandle& nh_private)
    {
        const std::string feature_extractor_namespace = "feature_extractor";
        int inference_height, inference_width;
        nh_private.getParam(feature_extractor_namespace + "/confidence_thresh", feature_extractor_parameters.confidence_threshold);
        nh_private.getParam(feature_extractor_namespace + "/distance_thresh", feature_extractor_parameters.distance_threshold);
        nh_private.getParam(feature_extractor_namespace + "/horizontal_boarder_width", feature_extractor_parameters.horizontal_boarder_width);
        nh_private.getParam(feature_extractor_namespace + "/vertical_boarder_width", feature_extractor_parameters.vertical_boarder_width);
        nh_private.getParam(feature_extractor_namespace + "/inference_width", inference_width);
        nh_private.getParam(feature_extractor_namespace + "/inference_height", inference_height);
        nh_private.getParam(feature_extractor_namespace + "/weight_path", super_point_weight_path);
        ROS_ASSERT(inference_height % 8 == 0);
        ROS_ASSERT(inference_width % 8 == 0);
        feature_extractor_parameters.inference_resolution = cv::Size(inference_width, inference_height);

        const std::string feature_tracker_namespace = "feature_tracker";
        nh_private.getParam(feature_tracker_namespace + "/match_thresh", feature_tracker_parameters.match_thresh);
        nh_private.getParam(feature_tracker_namespace + "/parallax_thresh", feature_tracker_parameters.parallax_thresh);
        nh_private.getParam(feature_tracker_namespace + "/ratio_thresh", feature_tracker_parameters.ratio_thresh);
        nh_private.getParam(feature_tracker_namespace + "/fundamental_filter", feature_tracker_parameters.fundamental_filter);

        const std::string triangulator_namespace = "triangulator";
        nh_private.getParam(triangulator_namespace + "/view_angle_check", triangulator_parameters.view_angle_check);
        nh_private.getParam(triangulator_namespace + "/baseline_check", triangulator_parameters.baseline_check);
        nh_private.getParam(triangulator_namespace + "/baseline_thresh", triangulator_parameters.baseline_thresh);
        nh_private.getParam(triangulator_namespace + "/error_thresh", triangulator_parameters.error_thresh);
        nh_private.getParam(triangulator_namespace + "/view_angle_thresh", triangulator_parameters.view_angle_thresh);

        const std::string optimizer_namespace = "optimizer";
        nh_private.getParam(optimizer_namespace + "/linear_solver_type", optimizer_options.linear_solver_type);
        nh_private.getParam(optimizer_namespace + "/trust_region_strategy", optimizer_options.trust_region_strategy);
        nh_private.getParam(optimizer_namespace + "/max_iteration", optimizer_options.max_iteration);
        nh_private.getParam(optimizer_namespace + "/num_threads", optimizer_options.num_threads);
        nh_private.getParam(optimizer_namespace + "/verbose", optimizer_options.verbose);
        nh_private.getParam(optimizer_namespace + "/max_solver_time", optimizer_options.max_solver_time);

        const std::string estimation_namespace = "estimation";
        nh_private.getParam(estimation_namespace + "/min_num_match", min_num_match);
        nh_private.getParam(estimation_namespace + "/sliding_window_size", sliding_window_size);
        nh_private.getParam(estimation_namespace + "/parallax_thresh", parallax_thresh);
        nh_private.getParam(estimation_namespace + "/linearized_ba_thresh", linearized_ba_thresh);
        nh_private.getParam(estimation_namespace + "/linearized_bg_thresh", linearized_bg_thresh);
        nh_private.getParam(estimation_namespace + "/max_preintegration_time", max_preintegration_time);


        const std::string imu_namespace = "imu";
        nh_private.getParam(imu_namespace + "/accelerator_noise", accelerator_noise);
        nh_private.getParam(imu_namespace + "/gyroscope_noise", gyroscope_noise);
        nh_private.getParam(imu_namespace + "/accelerator_random_walk_noise", accelerator_random_walk_noise);
        nh_private.getParam(imu_namespace + "/gyroscope_random_walk_noise", gyroscope_random_walk_noise);
        noise = IMU::IMUNoise::CreatNoiseMatrix(accelerator_noise, gyroscope_noise,
                accelerator_random_walk_noise, gyroscope_random_walk_noise);

        const std::string camera_namespace = "camera";
        std::vector<double> input_intrinsic, input_distortion, output_intrinsic;
        std::string camera_type;
        int input_camera_height, input_camera_width, output_camera_height, output_camera_width;
        bool use_gpu_for_undistortion;
        nh_private.getParam(camera_namespace + "/camera_type", camera_type);
        nh_private.getParam(camera_namespace + "/input_intrinsic", input_intrinsic);
        nh_private.getParam(camera_namespace + "/input_distortion", input_distortion);
        nh_private.getParam(camera_namespace + "/output_intrinsic", output_intrinsic);
        nh_private.getParam(camera_namespace + "/input_image_height", input_camera_height);
        nh_private.getParam(camera_namespace + "/input_image_width", input_camera_width);
        nh_private.getParam(camera_namespace + "/output_image_height", output_camera_height);
        nh_private.getParam(camera_namespace + "/output_image_width", output_camera_width);
        nh_private.getParam(camera_namespace + "/use_gpu_for_undistortion", use_gpu_for_undistortion);
        camera_ptr = Vision::Camera::Creat(camera_type, input_intrinsic, input_distortion, output_intrinsic,
                                           cv::Size(input_camera_width, input_camera_height),
                                           cv::Size(output_camera_width, output_camera_height),
                                           use_gpu_for_undistortion);

        const std::string extrinsic_namespace = "extrinsic";
        std::vector<double> extrinsic_rotation;
        std::vector<double> extrinsic_position;
        nh_private.getParam(extrinsic_namespace + "/extrinsic_rotation", extrinsic_rotation);
        nh_private.getParam(extrinsic_namespace + "/extrinsic_position", extrinsic_position);
        nh_private.getParam(extrinsic_namespace + "/gravity_norm", gravity_norm);
        ROS_ASSERT(extrinsic_position.size() == 3);
        ROS_ASSERT(extrinsic_rotation.size() == 4 || extrinsic_rotation.size() == 9);

        p_i_c = Vector3{extrinsic_position[0], extrinsic_position[1], extrinsic_position[2]};
        if(extrinsic_rotation.size() == 9)
        {
            Matrix3 rotation;
            rotation<<extrinsic_rotation[0], extrinsic_rotation[1], extrinsic_rotation[2],
                      extrinsic_rotation[3], extrinsic_rotation[4], extrinsic_rotation[5],
                      extrinsic_rotation[6], extrinsic_rotation[7], extrinsic_rotation[8];
            q_i_c = Quaternion(rotation).normalized();
        }
        else
        {
            q_i_c = Quaternion{extrinsic_rotation[0], extrinsic_rotation[1], extrinsic_rotation[2], extrinsic_rotation[3]};
            q_i_c.normalize();
        }

        return true;

    }


}//end of SuperVIO::Estimation

