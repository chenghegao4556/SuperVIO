//
// Created by chenghe on 4/14/20.
//

#ifndef SUPER_VIO_VISUAL_IMU_ALIGNMENT_H
#define SUPER_VIO_VISUAL_IMU_ALIGNMENT_H

#include <estimation/vio_states_measurements.h>
#include <imu/imu_states_measurements.h>
#include <sfm/initial_sfm.h>
namespace SuperVIO::Estimation
{
    class VisualIMUAligner
    {
    public:
        typedef std::map<IMU::StateKey, Vector3, std::less<>,
                Eigen::aligned_allocator<std::pair<IMU::StateKey, Vector3>>> VelocityMap;
        class Result
        {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
            Result();

            bool success;
            double scale;
            Vector3 gravity_vector;
            VelocityMap velocity_map;
        };

        static Result
        SolveGravityVectorAndVelocities(const VIOStatesMeasurements& states_measurements,
                                        const std::map<Vision::StateKey, SFM::Pose>& visual_pose_map,
                                        const Vector3& p_i_c,
                                        const double& gravity_norm);

        static Vector3
        SolveGyroscopeBias(const VIOStatesMeasurements& last_states_measurements,
                           const std::map<Vision::StateKey, SFM::Pose>& visual_pose_map);

    protected:
        static MatrixX TangentBasis(Vector3 &gravity_vector);

        static Result
        RefineGravityVectorAndVelocities(const VIOStatesMeasurements& last_states_measurements,
                                         const std::map<Vision::StateKey, SFM::Pose>& visual_pose_map,
                                         const Vector3& p_i_c,
                                         const Vector3& gravity_vector,
                                         const double& gravity_norm);


    };//end of VisualIMUAligner
}//end of SuperVIO::Estimation
#endif //SUPER_VIO_VISUAL_IMU_ALIGNMENT_H
