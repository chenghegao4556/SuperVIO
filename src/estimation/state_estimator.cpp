//
// Created by chenghe on 4/12/20.
//
#include <estimation/state_estimator.h>
#include <sensor_msgs/PointCloud.h>
#include <iostream>
namespace SuperVIO::Estimation
{
    class Timer
    {
    public:

        typedef std::chrono::steady_clock::time_point TimePoint;
        Timer():
            t1(std::chrono::steady_clock::now()),
            t2(std::chrono::steady_clock::now())
        {

        }

        void Tic()
        {
            t1 = std::chrono::steady_clock::now();
        }

        void Toc()
        {
            t2 = std::chrono::steady_clock::now();
        }

        double Duration()
        {
            std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>> ( t2-t1 );
            return time_used.count();
        }

    private:
        TimePoint t1, t2;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    StateEstimator::
    StateEstimator(const ros::NodeHandle& nh,
                   const ros::NodeHandle& nh_private):
                    nh_(nh),
                    nh_private_(nh_private),
                    image_transport_(nh_)
    {
        const std::string sensor_topic_namespace = "sensor_topic";
        nh_private.getParam(sensor_topic_namespace + "/image_topic", image_topic_);
        nh_private.getParam(sensor_topic_namespace + "/imu_topic", imu_topic_);

        image_subscriber_ = image_transport_.subscribe(image_topic_, 200,
                &StateEstimator::ImageCallBack, this);
        imu_subscriber_ = nh_.subscribe(imu_topic_, 2000, &StateEstimator::IMUCallBack, this,
                ros::TransportHints().tcpNoDelay());

        parameters_.Load(nh_private_);
        visualizer_ptr_ = Visualization::Visualizer::Creat(parameters_.q_i_c, parameters_.p_i_c);
        feature_publisher_ = nh_.advertise<sensor_msgs::PointCloud>("/feature_tracker/feature", 1000);
        image_publisher_ = nh_.advertise<sensor_msgs::Image>("/dense_mapping/image", 1000);
        thread_ = std::thread([this] { MainThread(); });
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    StateEstimator::
    ~StateEstimator()
    {
        thread_.join();
        while (!imus_.empty())
        {
            imus_.pop();
        }
        while (!images_.empty())
        {
            images_.pop();
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void StateEstimator::
    ImageCallBack(const sensor_msgs::ImageConstPtr& image_msg)
    {
        WriteLock write_lock(mutex_);
        std::string encoding = image_msg->encoding;
        if (encoding == "8UC1")
        {
            encoding = "mono8";
        }
        auto distort_image = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::MONO8)->image;
        auto undistort_image = parameters_.camera_ptr->UndistortImage(distort_image);
        images_.push(std::make_pair(image_msg->header.stamp.toSec(), undistort_image));
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void StateEstimator::
    IMUCallBack(const sensor_msgs::ImuConstPtr& imu_msg)
    {
        WriteLock write_lock(mutex_);
        Vector3 acceleration(imu_msg->linear_acceleration.x,
                             imu_msg->linear_acceleration.y,
                             imu_msg->linear_acceleration.z);
        Vector3 angular_velocity(imu_msg->angular_velocity.x,
                                 imu_msg->angular_velocity.y,
                                 imu_msg->angular_velocity.z);
        imus_.push(std::make_pair(imu_msg->header.stamp.toSec(),
                                 IMU::IMURawMeasurement(0.0, acceleration, angular_velocity)));
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    bool StateEstimator::
    IMUArrived(double time)
    {
        ReadLock read_lock(mutex_);
        bool arrived = false;
        if(!imus_.empty())
        {
            if(imus_.back().first >= time)
            {
                arrived = true;
            }
        }
        return arrived;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    IMU::IMURawMeasurements StateEstimator::
    GetIMURawMeasurements(double last_image_time,
                          double new_image_time)
    {
        WriteLock write_lock(mutex_);
        std::vector<std::pair<double, IMU::IMURawMeasurement>> imus;
        if(new_image_time <= imus_.back().first)
        {
            while(imus_.front().first <= last_image_time)
            {
                imus_.pop();
            }
            while(imus_.front().first < new_image_time)
            {
                imus.push_back(imus_.front());
                imus_.pop();
            }
            imus.push_back(imus_.front());
        }

        IMU::IMURawMeasurements raw_measurements;
        for(size_t i = 0; i < imus.size(); ++i)
        {
            if(i == 0)
            {
                imus[i].second.dt = imus[i].first - last_image_time;
            }
            else if (i == (imus.size() - 1))
            {
                imus[i].second.dt = new_image_time - imus[i - 1].first;
            }
            else
            {
                imus[i].second.dt = imus[i].first - imus[i - 1].first;
            }

            raw_measurements.push_back(imus[i].second);
        }

        return raw_measurements;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    std::pair<double, cv::Mat> StateEstimator::
    GetFrontImage()
    {
        WriteLock write_lock(mutex_);
        auto front = images_.front();
        images_.pop();
        return front;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    bool StateEstimator::
    ImageBufferEmpty()
    {
        ReadLock read_lock(mutex_);
        return images_.empty();
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void StateEstimator::
    MainThread()
    {
        bool   first_image = true;
        double last_image_time = 0.0;
        EstimationCore estimation_core(parameters_);
        VIOStatesMeasurements states_measurements;
        size_t num_lost = 0;
        IMU::IMURawMeasurements temp_raw_measurements;
        while(ros::ok())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            if(!ImageBufferEmpty())
            {
                auto image_msg = this->GetFrontImage();
                auto current_image_time = image_msg.first;
                auto undistort_image = image_msg.second;

                while (!IMUArrived(current_image_time))
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                }

                auto imu_raw_measurements = this->GetIMURawMeasurements(last_image_time, current_image_time);

                Timer timer;
                if(first_image)
                {
                    Vector3 acceleration_0 = imu_raw_measurements.rbegin()->acceleration;
                    Vector3 angular_velocity_0 = imu_raw_measurements.rbegin()->angular_velocity;
                    states_measurements = estimation_core.InitializeVIOStatesMeasurements(current_image_time,
                            undistort_image,  acceleration_0, angular_velocity_0);

                    first_image = false;
                }
                else
                {
                    timer.Tic();
                    if(num_lost > 0)
                    {
                        auto temp_raws = temp_raw_measurements;
                        for(const auto& raw: imu_raw_measurements)
                        {
                            temp_raws.push_back(raw);
                        }

                        states_measurements = estimation_core.Estimate(states_measurements, current_image_time,
                                undistort_image, temp_raws);
                        if(!states_measurements.lost)
                        {
                            ROS_INFO_STREAM("RECOVER FROM LOST!!!");
                            temp_raw_measurements.clear();
                            num_lost = 0;
                        }
                    }
                    else
                    {
                        states_measurements = estimation_core.Estimate(states_measurements, current_image_time,
                                                                       undistort_image, imu_raw_measurements);
                    }
                    if(states_measurements.lost)
                    {
                        num_lost ++;
                        for(const auto& raw: imu_raw_measurements)
                        {
                            temp_raw_measurements.push_back(raw);
                        }
                        continue;
                    }
                    timer.Toc();
                }
                if(states_measurements.initialized)
                {
                    visualizer_ptr_->SetData(states_measurements.imu_state_map,
                                             states_measurements.feature_state_map, states_measurements.track_map,
                                             states_measurements.frame_measurement_map, undistort_image);
                    PublishData(states_measurements, undistort_image, current_image_time);
                }
                else
                {
                    visualizer_ptr_->SetTrackedImage(states_measurements.track_map,
                            states_measurements.frame_measurement_map, undistort_image);
                }
                visualizer_ptr_->SetFPS(timer.Duration());

                last_image_time = current_image_time;
            }
        }
        states_measurements.feature_state_map.clear();
        states_measurements.frame_measurement_map.clear();
        states_measurements.imu_state_map.clear();
        states_measurements.imu_raw_measurements_map.clear();
        states_measurements.imu_pre_integration_measurements_map.clear();
        states_measurements.track_map.clear();
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void StateEstimator::
    PublishData(const VIOStatesMeasurements& states_measurements, const cv::Mat& image,
                const double current_state_key) const
    {
        sensor_msgs::PointCloud feature_msg;
        auto frame_iter = states_measurements.frame_measurement_map.find(current_state_key);
        ROS_ASSERT(frame_iter != states_measurements.frame_measurement_map.end());
        auto state_iter = states_measurements.imu_state_map.find(current_state_key);
        ROS_ASSERT(state_iter != states_measurements.imu_state_map.end());
        const Quaternion q_w_c = state_iter->second.rotation * parameters_.q_i_c;
        const Vector3    p_w_c = state_iter->second.rotation * parameters_.p_i_c + state_iter->second.position;
        const Quaternion q_c_w = q_w_c.inverse();
        const Vector3    p_c_w = - (q_w_c.inverse() * p_w_c);
        for(const auto& feature: states_measurements.feature_state_map)
        {
            const Vector3 camera_point = q_c_w * feature.second.world_point + p_c_w;
            Vector2 point = parameters_.camera_ptr->Project(camera_point);
            if(point.x() >= 0 && point.x() <= image.size().width &&
               point.y() >= 0 && point.y() <= image.size().height &&
               camera_point.z() > 0.0)
            {
                    const double depth = camera_point.z();
                    geometry_msgs::Point32 p;
                    p.x = point.x();
                    p.y = point.y();
                    p.z = depth;
                    feature_msg.points.push_back(p);
            }
        }


        std_msgs::Header header;
        header.stamp = ros::Time(current_state_key);

        sensor_msgs::ImagePtr image_msg = cv_bridge::CvImage(header, "mono8", image).toImageMsg();
        feature_msg.header = header;

        feature_publisher_.publish(feature_msg);
        image_publisher_.publish(image_msg);
    }
}//end of SuperVIO::Estimation
