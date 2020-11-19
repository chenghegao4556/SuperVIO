//
// Created by chenghe on 8/9/20.
//
#include <dense_mapping/densifier.h>
#include <dense_mapping/depth_interpolator.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <shared_mutex>
#include <image_transport/image_transport.h>
#include <nav_msgs/Odometry.h>
#include <thread>
#include <utility>
class Frame
{
public:
    Frame(double _time_stamp,
          const SuperVIO::Quaternion& _q_w_c,
          SuperVIO::Vector3  _p_w_c,
          cv::Mat  _image,
          cv::Mat  _depth):
          time_stamp(_time_stamp),
          q_w_c(_q_w_c),
          p_w_c(std::move(_p_w_c)),
          image(std::move(_image)),
          depth(std::move(_depth))
    {

    }

    Frame():
        time_stamp(0.0),
        q_w_c(SuperVIO::Quaternion::Identity()),
        p_w_c(SuperVIO::Vector3::Zero())
    {

    }

    double time_stamp;
    SuperVIO::Quaternion q_w_c;
    SuperVIO::Vector3    p_w_c;

    cv::Mat image;
    cv::Mat depth;
};

class ARDepthNode
{
public:
    typedef std::unique_lock<std::shared_mutex> WriteLock;
    typedef std::shared_lock<std::shared_mutex> ReadLock;

    explicit ARDepthNode(const ros::NodeHandle& nh):
            index(0),
            current_image_time_(3.0),
            current_feature_time_(2.0),
            current_pose_time_(1.0),
            nh_(nh),
            image_transport_(nh_)
    {
        Eigen::setNbThreads(4);
        image_subscriber_   = image_transport_.subscribe("/dense_mapping/image", 200,
                                                         &ARDepthNode::ImageCallBack, this);
        feature_subscriber_ = nh_.subscribe("/feature_tracker/feature", 2000,
                                            &ARDepthNode::FeatureCallback, this);

        camera_pose_subscriber_ = nh_.subscribe("/camera_pose", 2000,
                                                &ARDepthNode::CameraPoseCallback, this);

        thread_ = std::thread([this] { MainThread(); });
    }

    ~ARDepthNode()
    {
        thread_.join();
    }

    void CameraPoseCallback(const nav_msgs::OdometryConstPtr& odometry_msg)
    {
        WriteLock write_lock(mutex_);
        q_w_c_ = SuperVIO::Quaternion(odometry_msg->pose.pose.orientation.w,
                                      odometry_msg->pose.pose.orientation.x,
                                      odometry_msg->pose.pose.orientation.y,
                                      odometry_msg->pose.pose.orientation.z);

        p_w_c_ = SuperVIO::Vector3(odometry_msg->pose.pose.position.x,
                                   odometry_msg->pose.pose.position.y,
                                   odometry_msg->pose.pose.position.z);

        current_pose_time_ = odometry_msg->header.stamp.toSec();

    }

    void FeatureCallback(const sensor_msgs::PointCloudConstPtr &feature_msg)
    {
        WriteLock write_lock(mutex_);
        points_.clear();
        depths_.clear();
        for (const auto& point : feature_msg->points)
        {
            double x = point.x;
            double y = point.y;
            double depth = point.z;
            if(depth > 100)
            {
                continue;
            }
            points_.emplace_back(x, y);
            depths_.emplace_back(depth);
        }
        current_feature_time_ = feature_msg->header.stamp.toSec();
    }

    void ImageCallBack(const sensor_msgs::ImageConstPtr& image_msg)
    {
        WriteLock write_lock(mutex_);
        std::string encoding = image_msg->encoding;
        if (encoding == "8UC1")
        {
            encoding = "mono8";
        }
        current_image_ = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::MONO8)->image;
        current_image_time_ = image_msg->header.stamp.toSec();
    }

    bool IsAvailable(double last_image_time)
    {
        ReadLock read_lock(mutex_);
        return (current_feature_time_ == current_image_time_) &&
               (current_pose_time_    == current_image_time_) &&
               (current_feature_time_ != last_image_time);
    }

    void GetData(Frame& current_frame,
                 cv::Mat& last_depth,
                 cv::Mat& sparse_depth,
                 std::vector<cv::Mat>& reference_image,
                 double& last_time)
    {
        ReadLock read_lock(mutex_);
        current_frame.time_stamp = current_image_time_;
        current_frame.q_w_c = q_w_c_;
        current_frame.p_w_c = p_w_c_;
        current_frame.image = current_image_;
        double distance = 0;
        if(!frames_.empty())
        {
            last_depth = frames_.rbegin()->second.depth;
            distance   = (frames_.rbegin()->second.p_w_c - p_w_c_).norm() / 2;
        }

        sparse_depth = cv::Mat::zeros(current_image_.size(), CV_64FC1);
        for(size_t i = 0; i < depths_.size(); ++i)
        {
            sparse_depth.at<double>(points_[i]) = depths_[i];
        }

        for (const auto& frame: frames_)
        {
            if((frame.second.p_w_c - p_w_c_).norm() > distance)
            {
                reference_image.push_back(frame.second.image);
                break;
            }
        }

        last_time = current_image_time_;
    }

    void AddFrame(const Frame& frame)
    {
        WriteLock write_lock(mutex_);
        if(!frames_.empty() || frames_.size() > 10)
        {
            double distance = (frames_.rbegin()->second.p_w_c - frame.p_w_c).norm() / 2;
            if(distance < 0.1)
            {
                frames_.erase(frames_.rbegin()->first);
            }
            else
            {
                frames_.erase(frames_.begin()->first);
            }
        }
        frames_.insert({frame.time_stamp, frame});
        if(!frame.depth.empty())
        {
            double min_val, max_val;
            cv::Mat filtered_depthmap_visual =
                    cv::Mat::zeros(frame.depth.size(), frame.depth.depth());
            frame.depth.copyTo(filtered_depthmap_visual, frame.depth < 100);
//            cv::minMaxLoc(filtered_depthmap_visual, &min_val, &max_val);
            filtered_depthmap_visual =
                    255 * (filtered_depthmap_visual) / 20;
            filtered_depthmap_visual.convertTo(filtered_depthmap_visual, CV_8U);
            cv::applyColorMap(filtered_depthmap_visual, filtered_depthmap_visual, cv::COLORMAP_JET);
            std::stringstream file_name;
            file_name << "./"<<std::setw(8)<<std::setfill('0')<<index<<".jpg";
            cv::imwrite(file_name.str(), filtered_depthmap_visual);
            cv::namedWindow("dense mapping", cv::WINDOW_NORMAL);
            cv::imshow("dense mapping", filtered_depthmap_visual);
            cv::waitKey(1);
            index++;
        }
    }

    void MainThread()
    {
        double last_image_time = -1;
        while(ros::ok())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            if(IsAvailable(last_image_time))
            {
                Frame frame;
                cv::Mat current_image, sparse_depth, last_depth;
                std::vector<cv::Mat> reference_images;
                GetData(frame, last_depth, sparse_depth, reference_images, last_image_time);
                if(!frames_.empty())
                {
                    frame.depth = SuperVIO::DenseMapping::DepthInterpolator::Estimate(
                            sparse_depth, reference_images, frame.image, last_depth);
                }
                AddFrame(frame);
            }
        }
    }

private:
    int index;
    double current_image_time_;
    double current_feature_time_;
    double current_pose_time_;
    ros::NodeHandle nh_;
    image_transport::ImageTransport image_transport_;
    image_transport::Subscriber image_subscriber_;
    ros::Subscriber feature_subscriber_;
    ros::Subscriber camera_pose_subscriber_;
    std::shared_mutex mutex_;
    std::thread thread_;

    cv::Mat current_image_;
    SuperVIO::Quaternion q_w_c_;
    SuperVIO::Vector3    p_w_c_;
    std::vector<cv::Point2f> points_;
    std::vector<double> depths_;
    std::map<double, Frame> frames_;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "dense_mapper");
    ros::NodeHandle nh;
    ARDepthNode node(nh);
    ros::spin();
}

