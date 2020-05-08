//
// Created by chenghe on 4/12/20.
//

#ifndef SUPER_VIO_ESTIMATION_CORE_H
#define SUPER_VIO_ESTIMATION_CORE_H

#include <estimation/parameters.h>
#include <estimation/vio_states_measurements.h>

#include <vision/feature_extractor.h>
#include <vision/feature_tracker.h>
#include <vision/triangulator.h>

#include <imu/imu_noise.h>
#include <imu/imu_processor.h>
#include <imu/imu_checker.h>

#include <optimization/optimizer.h>
#include <optimization/marginalizer.h>
#include <optimization/helper.h>

#include <estimation/visual_imu_alignment.h>
namespace SuperVIO::Estimation
{
    class EstimationCore
    {
    public:

        typedef Optimization::BaseParametersBlock::Ptr ParameterBlockPtr;
        typedef Optimization::BaseResidualBlock::Ptr ResidualBlockPtr;



        class TrackSummery
        {
        public:

            TrackSummery(bool _valid_tracking,
                         bool _is_key_frame,
                         Vision::TrackMap  _track_map,
                         Vision::FrameMeasurement  _frame_measurement);

            bool valid_tracking;
            bool is_key_frame;
            Vision::TrackMap track_map;
            Vision::FrameMeasurement frame_measurement;
        };

        explicit EstimationCore(Parameters  parameters = Parameters());

        [[nodiscard]] VIOStatesMeasurements
        Estimate(const VIOStatesMeasurements& last_states_measurement,
                 const IMU::StateKey& state_key,
                 const Vision::Image& image,
                 const IMU::IMURawMeasurements& raw_measurements) const;

        [[nodiscard]] VIOStatesMeasurements
        InitializeVIOStatesMeasurements(const IMU::StateKey& state_key,
                                        const Vision::Image& image,
                                        const Vector3& acceleration_0,
                                        const Vector3& angular_velocity_0) const;

    private:

        [[nodiscard]] VIOStatesMeasurements
        InitializeVIOStatesMeasurements(const IMU::StateKey& state_key,
                                        const IMU::IMUState& imu_state,
                                        const Vision::FrameMeasurement& frame_measurement,
                                        const Vector3& acceleration_0,
                                        const Vector3& angular_velocity_0,
                                        const Vector3& gravity_vector) const;


        [[nodiscard]] TrackSummery
        TrackImage(const VIOStatesMeasurements& last_states_measurement,
                   const IMU::StateKey& state_key,
                   const Vision::Image& image) const;

        [[nodiscard]] VIOStatesMeasurements
        ProcessIMURawMeasurements(const VIOStatesMeasurements& last_states_measurement,
                                  const IMU::StateKey& state_key,
                                  const IMU::IMURawMeasurements& raw_measurements) const;

        [[nodiscard]] std::pair<bool, VIOStatesMeasurements>
        InitialAlignment(const VIOStatesMeasurements& uninitialized_states_measurements) const;

        [[nodiscard]] Vision::FeatureStateMap
        Triangulate(const VIOStatesMeasurements& last_states_measurements) const;

        [[nodiscard]] VIOStatesMeasurements
        Optimize(const VIOStatesMeasurements& unoptimized_states_measurements) const;

        [[nodiscard]] VIOStatesMeasurements
        RemoveMeasurementsAndStates(const VIOStatesMeasurements& states_measurements,
                                    bool is_new_frame_key_frame)const;

        [[nodiscard]] Optimization::MarginalizationInformation
        Marginalize(const VIOStatesMeasurements& states_measurements,
                    bool is_new_frame_key_frame) const;
    private:

        //! super point feature extractor
        Vision::FeatureExtractor::Ptr feature_extractor_;
        Parameters parameters_;
    };//end of EstimationCore
}//end of SuperVIO::Estimation

#endif //SUPER_VIO_ESTIMATION_CORE_H
