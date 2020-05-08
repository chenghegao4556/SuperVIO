//
// Created by chenghe on 3/12/20.
//

#include <gtest/gtest.h>

#include <vision/feature_extractor.h>
#include <vision/feature_tracker.h>
#include <sfm/initial_sfm.h>
#include <visualization/visualizer.h>
class Timer
{
public:

    typedef std::chrono::steady_clock::time_point TimePoint;

    void Tic()
    {
        t1 = std::chrono::steady_clock::now();
    }

    void Toc()
    {
        t2 = std::chrono::steady_clock::now();
    }

    void Duration(const std::string &s)
    {
        std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>> ( t2-t1 );
        LOG(INFO) <<s<<" costs time: "<<time_used.count() <<" seconds.";
    }

private:
    TimePoint t1, t2;
};
int main()
{
    using namespace SuperVIO;
    ros::Time::init();
    std::string weight_path("/home/chenghe/catkin_ws/src/SuperVIO/data/superpoint.pt");
    std::string dict_path("/home/chenghe/catkin_ws/src/SuperVIO/data/");
    std::vector<std::string> image_pathes{"0.png", "1.png", "2.png", "3.png", "4.png", "5.png", "6.png",
                                          "7.png", "8.png", "9.png", "10.png"};

    Vision::FeatureExtractor::Parameters parameters(10, 10, cv::Size(640, 480), 8, 0.015f);
    auto feature_extractor_ptr = Vision::FeatureExtractor::Creat(parameters, weight_path);
    Vision::FrameMeasurementMap frame_measurement_map;
    Vision::TrackMap track_map;
    Vision::FeatureTracker::Parameters fp;
    cv::Mat image;
    for(size_t i = 0; i < image_pathes.size(); ++i)
    {
        image = cv::imread(dict_path + image_pathes[i]);
        Timer timer;
        timer.Tic();
        auto frame_measurement = feature_extractor_ptr->Compute(image);
        auto time_stamp = (double)i;
        std::cout<<"t: "<<time_stamp<<std::endl;
        if(i == 0)
        {
            track_map = Vision::FeatureTracker::CreatEmptyTrack(frame_measurement, time_stamp);
        }
        else
        {
            auto last_frame_measurement = frame_measurement_map.rbegin();
            std::cout<<"l: "<<last_frame_measurement->first<<std::endl;
            auto track_result = Vision::FeatureTracker::Tracking(track_map, last_frame_measurement->second,
                    frame_measurement, last_frame_measurement->first, time_stamp, fp);
            std::cout<<"parallax  "<<track_result.parallax<<" num matches "<<track_result.num_matches<<std::endl;
            track_map = track_result.track_map;
        }

        frame_measurement_map.insert(std::make_pair(time_stamp, frame_measurement));
        timer.Toc();
        timer.Duration("tracking cost");
    }
    Visualization::Visualizer visualizer(Quaternion::Identity(), Vector3::Zero());
    image = visualizer.VisualizeTrackedImage(track_map, frame_measurement_map, image);
    cv::imwrite("tracking.jpg", image);

    std::vector<double> k{7.188560000000e+02,  7.188560000000e+02, 6.071928000000e+02, 1.852157000000e+02};
    std::vector<double> d{0.0, 0.0, 0.0, 0.0};
    auto camera_ptr = Vision::Camera::Creat(
            "simplepinhole",k, d, k, cv::Size(1000, 1000), cv::Size(1000, 1000), false);
    Timer timer;
    timer.Tic();
    auto result = SFM::InitialSFM::Construct(track_map, frame_measurement_map, camera_ptr);
    timer.Toc();
    for(const auto& pose: result.frame_pose_map)
    {
        const auto& position = pose.second.position;
        std::cout<<"frame: "<<pose.first<<std::endl;
        std::cout<<"position: "<<position.x()<<" "<<position.y()<<" "<<position.z()<<std::endl;
    }
    timer.Duration("initial sfm");

    double error = 0.0;
    double size  = 0.0;
    for(const auto& feature: result.feature_state_map)
    {
        const auto& world_point = feature.second.world_point;
        for(const auto& measurement: track_map.at(feature.first).measurements)
        {
            const auto& pt = frame_measurement_map.at(measurement.state_id).key_points[measurement.point_id].point;
            const auto& rotation = result.frame_pose_map.at(measurement.state_id).rotation;
            const auto& position = result.frame_pose_map.at(measurement.state_id).position;
            const Vector3 camera_point = rotation.inverse() * (world_point - position);
            error += (camera_ptr->Project(camera_point) - Vector2{pt.x, pt.y}).norm();
            size += 1.0;
        }
    }
    std::cout<<"mean projection error: "<<error / size<<std::endl;

    return 0;
}
