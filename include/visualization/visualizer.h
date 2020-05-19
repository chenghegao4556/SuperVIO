//
// Created by chenghe on 5/1/20.
//

#ifndef SUPER_VIO_VISUALIZER_H
#define SUPER_VIO_VISUALIZER_H
#include <thread>
#include <shared_mutex>
#include <pangolin/pangolin.h>
#include <imu/imu_states_measurements.h>
#include <vision/vision_measurements.h>

namespace SuperVIO::Visualization
{
    class Visualizer
    {
    public:
        typedef std::unique_lock<std::shared_mutex> WriteLock;
        typedef std::shared_lock<std::shared_mutex> ReadLock;
        typedef std::shared_ptr<Visualizer> Ptr;

        static Ptr Creat(const Quaternion& q_i_c, Vector3 p_i_c);

        Visualizer(const Quaternion& q_i_c, Vector3 p_i_c);

        ~Visualizer();
        void SetData(const IMU::IMUStateMap& imu_state_map,
                     const Vision::FeatureStateMap& feature_state_map,
                     const Vision::TrackMap& track_map,
                     const Vision::FrameMeasurementMap& frame_measurement_map,
                     const Vision::Image& image);

        void SetTrackedImage(const Vision::TrackMap& track_map,
                             const Vision::FrameMeasurementMap& frame_measurement_map,
                             const Vision::Image& image);

        void ResetData();

        void SetFPS(const double& fps);

        [[nodiscard]] Vision::Image
        VisualizeTrackedImage(const Vision::TrackMap& track_map,
                              const Vision::FrameMeasurementMap& frame_measurement_map,
                              const Vision::Image& image) const;

    protected:
        void RenderLoop();
        void InitColorMap();
        void GetData(IMU::IMUStateMap& imu_state_map,
                     Vector3s& point_cloud,
                     Vision::Image& tracked_image);

    private:
        IMU::IMUStateMap imu_state_map_;
        Vision::FeatureStateMap  feature_state_map_;
        Vision::Image    tracked_image_;
        cv::Mat          color_map_;

        const Quaternion q_i_c_;
        const Vector3    p_i_c_;

        std::thread  thread_;
        std::shared_mutex mutex_;
        double fps_;
        double size_;
    };//end of Visualizer
}//end of SuperVIO::Visualization
#endif //SUPER_VIO_VISUALIZER_H
