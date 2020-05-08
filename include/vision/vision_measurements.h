//
// Created by chenghe on 4/9/20.
//

#ifndef SUPER_VIO_VISION_MEASUREMENTS_H
#define SUPER_VIO_VISION_MEASUREMENTS_H

#include <utility/eigen_type.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
namespace SuperVIO::Vision
{
    typedef double StateKey;

    class KeyPoint;
    typedef std::vector<KeyPoint> KeyPoints;

    class FrameMeasurement;
    typedef std::map<StateKey, FrameMeasurement> FrameMeasurementMap;

    class TrackMeasurement;
    typedef std::vector<TrackMeasurement> TrackMeasurements;

    typedef cv::Mat Image;
    typedef std::map<StateKey, Image> ImageMap;

    class Track;
    typedef size_t TrackKey;
    typedef std::map<TrackKey, Track> TrackMap;

    class FeatureState;
    typedef std::map<TrackKey, FeatureState> FeatureStateMap;

    class KeyPoint
    {
    public:
        KeyPoint(const cv::Point2f& _point,
                 float _sigma_x,
                 float _sigma_y,
                 float _confidence);

        KeyPoint() = delete;
        cv::Point2f point;
        float sigma_x;
        float sigma_y;
        float confidence;
    };//end of KeyPoint

    class FrameMeasurement
    {
    public:
        FrameMeasurement(KeyPoints  _key_points,
                         cv::Mat    _descriptors);

        bool key_frame;
        KeyPoints key_points;
        cv::Mat descriptors;
    };//end of FrameMeasurement

    class TrackMeasurement
    {
    public:
        TrackMeasurement(StateKey _state_id,
                         size_t   _point_id);

        StateKey     state_id;
        size_t       point_id;
    };//end of TrackedMeasurement

    class Track
    {
    public:
        Track();

        bool    active;
        TrackMeasurements measurements;
    };//end of Track

    class FeatureState
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        FeatureState(double _depth, Vector3 _world_point);

        double depth;
        Vector3 world_point;
    };
}//end of SuperVIO::Vision

#endif //SUPER_VIO_VISION_MEASUREMENTS_H
