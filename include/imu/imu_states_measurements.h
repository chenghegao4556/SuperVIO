//
// Created by chenghe on 4/9/20.
//

#ifndef SUPER_VIO_IMU_STATES_MEASUREMENTS_H
#define SUPER_VIO_IMU_STATES_MEASUREMENTS_H
#include <utility/eigen_type.h>
#include <utility/eigen_base.h>

namespace SuperVIO::IMU
{
    typedef double StateKey;
    typedef std::pair<StateKey, StateKey> StatePairKey;

    class IMUState;
    typedef std::map<StateKey, IMUState> IMUStateMap;

    class IMURawMeasurement;
    typedef std::vector<IMURawMeasurement> IMURawMeasurements;
    typedef std::map<StatePairKey, IMURawMeasurements> IMURawMeasurementsMap;

    class IMUPreIntegrationMeasurement;
    typedef std::map<StatePairKey, IMUPreIntegrationMeasurement> IMUPreIntegrationMeasurementMap;

    typedef std::pair<IMUState, IMUPreIntegrationMeasurement> IMUStateAndMeasurements;

    class IMUState
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        explicit IMUState(const Quaternion& _rotation = Quaternion::Identity(),
                          Vector3 _position = Vector3::Zero(),
                          Vector3 _velocity = Vector3::Zero(),
                          Vector3 _linearized_ba = Vector3::Zero(),
                          Vector3 _linearized_bg = Vector3::Zero());

        Quaternion rotation;
        Vector3 position;
        Vector3 velocity;

        Vector3 linearized_ba;
        Vector3 linearized_bg;
    };//end of IMUState

    class IMURawMeasurement
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        IMURawMeasurement(double _dt,
                          Vector3 _acceleration,
                          Vector3 _angular_velocity);

        double dt;
        Vector3 acceleration;
        Vector3 angular_velocity;

    };//end of IMURawMeasurement


    class IMUPreIntegrationMeasurement
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        IMUPreIntegrationMeasurement(Vector3 _acceleration_0,
                                     Vector3 _angular_velocity_0,
                                     Vector3 _linearized_ba,
                                     Vector3 _linearized_bg,
                                     double _sum_dt = 0,
                                     Matrix15 _jacobian = Matrix15::Identity(),
                                     Matrix15 _covariance = Matrix15::Zero(),
                                     const Quaternion& _delta_q = Quaternion::Identity(),
                                     Vector3 _delta_p = Vector3::Zero(),
                                     Vector3 _delta_v = Vector3::Zero());

        Vector3 acceleration_0;
        Vector3 angular_velocity_0;

        Vector3 linearized_ba;
        Vector3 linearized_bg;

        double sum_dt;

        Matrix15 jacobian;
        Matrix15 covariance;

        Quaternion delta_q;
        Vector3 delta_p;
        Vector3 delta_v;
    };//end of IMUPreIntegrationMeasurement
}//end of SuperVIO::IMU

#endif //SUPER_VIO_IMU_STATES_MEASUREMENTS_H
