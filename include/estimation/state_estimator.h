//
// Created by chenghe on 4/12/20.
//

#ifndef SRC_STATE_ESTIMATOR_H
#define SRC_STATE_ESTIMATOR_H

#include <iostream>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Imu.h>
#include <estimation/estimation_core.h>
#include <visualization/visualizer.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber.h>

namespace SuperVIO::Estimation
{
    class StateEstimator
    {
    public:
        typedef std::unique_lock<std::shared_mutex> WriteLock;
        typedef std::shared_lock<std::shared_mutex> ReadLock;
        typedef std::map<double, Vector3, std::less<>,
                Eigen::aligned_allocator<std::pair<const double, Vector3>>> Vector3_Map;
        typedef std::map<double, Quaternion, std::less<>,
                Eigen::aligned_allocator<std::pair<const double, Quaternion>>> Quaternion_Map;
        StateEstimator(const ros::NodeHandle& nh,
                       const ros::NodeHandle& nh_private);
        ~StateEstimator();

        void ImageCallBack(const sensor_msgs::ImageConstPtr& image_msg);
        void IMUCallBack(const sensor_msgs::ImuConstPtr& imu_msg);

    protected:

        bool ImageBufferEmpty();
        bool IMUArrived(double time);
        void MainThread();

        IMU::IMURawMeasurements
        GetIMURawMeasurements(double last_image_time,
                              double new_image_time);

        std::pair<double, cv::Mat>
        GetFrontImage();


    private:

        ros::NodeHandle nh_;
        ros::NodeHandle nh_private_;
        std::string image_topic_;
        std::string imu_topic_;
        image_transport::ImageTransport image_transport_;
        image_transport::Subscriber image_subscriber_;
        ros::Subscriber imu_subscriber_;

        Parameters parameters_;
        std::shared_mutex mutex_;
        std::thread thread_;
        std::queue<std::pair<double, cv::Mat>> images_;
        std::queue<std::pair<double, IMU::IMURawMeasurement>>   imus_;
        Visualization::Visualizer::Ptr visualizer_ptr_;
    };//end of StateEstimator
}//end of SuperVIO::Estimation

#endif //SRC_STATE_ESTIMATOR_H
