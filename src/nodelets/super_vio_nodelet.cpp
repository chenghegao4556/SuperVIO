//
// Created by chenghe on 5/3/20.
//
#include <nodelet/super_vio_nodelet.h>
PLUGINLIB_EXPORT_CLASS(SuperVIO::SuperVIONodelet, nodelet::Nodelet)
namespace SuperVIO
{
    ////////////////////////////////////////////////////////////////////////////////////////
    SuperVIONodelet::
    SuperVIONodelet():
            nodelet::Nodelet(),
            state_estimator_ptr_(nullptr)
    {

    }


////////////////////////////////////////////////////////////////////////////////////////
    void SuperVIONodelet::
    onInit()
    {
        state_estimator_ptr_.reset(new Estimation::StateEstimator(getNodeHandle(), getPrivateNodeHandle()));
    }

}//end of SuperVIO
