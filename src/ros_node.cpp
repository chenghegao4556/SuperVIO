//
// Created by chenghe on 5/3/20.
//
#include <estimation/state_estimator.h>
int main(int argc, char **argv)
{
    ros::init(argc, argv, "super_vio_estimator");
    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");
    SuperVIO::Estimation::StateEstimator state_estimator(nh, nh_private);
    ros::spin();
}