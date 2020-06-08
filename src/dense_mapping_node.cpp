//
// Created by chenghe on 6/8/20.
//
#include <dense_mapping/densifier.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <shared_mutex>
#include <image_transport/image_transport.h>
#include <thread>

class DenseMappingNode
{
public:
    typedef std::unique_lock<std::shared_mutex> WriteLock;
    typedef std::shared_lock<std::shared_mutex> ReadLock;

    explicit DenseMappingNode(const ros::NodeHandle& nh):
                                                    current_image_time_(2.0),
                                                    current_feature_time_(1.0),
                                                    nh_(nh),
                                                    image_transport_(nh_)
    {
        image_subscriber_   = image_transport_.subscribe("/dense_mapping/image", 200,
                &DenseMappingNode::ImageCallBack, this);
        feature_subscriber_ = nh_.subscribe("/feature_tracker/feature", 2000,
                &DenseMappingNode::FeatureCallback, this);
        thread_ = std::thread([this] { MainThread(); });
    }

    ~DenseMappingNode()
    {
        thread_.join();
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
        image_ = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::MONO8)->image;
        current_image_time_ = image_msg->header.stamp.toSec();
    }

    bool IsAvailable(double last_image_time)
    {
        ReadLock read_lock(mutex_);
        return (current_feature_time_ == current_image_time_) && (current_feature_time_ != last_image_time);
    }

    void GetData(cv::Mat& image, std::vector<cv::Point2f>& points, std::vector<double>& depths, double& last_time)
    {
        ReadLock read_lock(mutex_);
        image = image_;
        points = points_;
        depths = depths_;
        last_time = current_image_time_;
    }

    void MainThread()
    {
        double last_image_time = -1;
        while(ros::ok())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            if(IsAvailable(last_image_time))
            {
                /**
                 * @todo dense mapping
                 */
                 cv::Mat image;
                std::vector<cv::Point2f> points;
                std::vector<double> depths;
                GetData(image, points, depths, last_image_time);
                auto depth_map = SuperVIO::DenseMapping::Densifier::Evaluate(image, points, depths);
            }
        }
    }

private:
    double current_image_time_;
    double current_feature_time_;
    ros::NodeHandle nh_;
    image_transport::ImageTransport image_transport_;
    image_transport::Subscriber image_subscriber_;
    ros::Subscriber feature_subscriber_;

    std::shared_mutex mutex_;
    std::vector<cv::Point2f> points_;
    std::vector<double> depths_;
    cv::Mat image_;
    std::thread thread_;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "dense_mapper");
    ros::NodeHandle nh;
    DenseMappingNode node(nh);
    ros::spin();
}
