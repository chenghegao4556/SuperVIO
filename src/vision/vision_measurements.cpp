//
// Created by chenghe on 4/9/20.
//
#include <vision/vision_measurements.h>

#include <utility>

namespace SuperVIO::Vision
{
    /////////////////////////////////////////////////////////////////////
    KeyPoint::
    KeyPoint(const cv::Point2f& _point, float _sigma_x,
             float _sigma_y, float _confidence):
            point(_point),
            sigma_x(_sigma_x),
            sigma_y(_sigma_y),
            confidence(_confidence)
    {

    }

    /////////////////////////////////////////////////////////////////////
    FrameMeasurement::
    FrameMeasurement(KeyPoints  _key_points,
                     cv::Mat  _descriptors):
                     key_frame(false),
                     key_points(std::move(_key_points)),
                     descriptors(std::move(_descriptors))
    {

    }

    /////////////////////////////////////////////////////////////////////
    TrackMeasurement::
    TrackMeasurement(StateKey _state_id,
                     size_t   _point_id):
                     state_id(_state_id),
                     point_id(_point_id)
    {

    }

    /////////////////////////////////////////////////////////////////////
    Track::
    Track():
        active(true)
    {

    }

    /////////////////////////////////////////////////////////////////////
    FeatureState::
    FeatureState(double _depth, Vector3 _world_point):
        depth(_depth),
        world_point(std::move(_world_point))
    {

    }

}//end of SuperVIO::Vision
