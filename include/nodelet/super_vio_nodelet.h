//
// Created by chenghe on 5/3/20.
//

#ifndef SUPER_VIO_NODELET_H
#define SUPER_VIO_NODELET_H
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <estimation/state_estimator.h>
namespace SuperVIO
{
    class SuperVIONodelet: public nodelet::Nodelet
    {
    public:
        SuperVIONodelet();
        void onInit() override;

    private:

        std::unique_ptr<Estimation::StateEstimator> state_estimator_ptr_;
    };//end of SynchronizationThreadsNodelet
}
#endif //SUPER_VIO_NODELET_H
