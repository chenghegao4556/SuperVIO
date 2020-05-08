//
// Created by chenghe on 4/9/20.
//
#include <imu/imu_states_measurements.h>

#include <utility>
namespace SuperVIO::IMU
{

    ///////////////////////////////////////////////////////////////////////
    IMUState::
    IMUState(const Quaternion& _rotation,
             Vector3 _position,
             Vector3 _velocity,
             Vector3 _linearized_ba,
             Vector3 _linearized_bg):
            rotation(_rotation),
            position(std::move(_position)),
            velocity(std::move(_velocity)),
            linearized_ba(std::move(_linearized_ba)),
            linearized_bg(std::move(_linearized_bg))

    {

    }

    ///////////////////////////////////////////////////////////////////////
    IMURawMeasurement::
    IMURawMeasurement(double _dt,
                      Vector3 _acceleration,
                      Vector3 _angular_velocity):
                      dt(_dt),
                      acceleration(std::move(_acceleration)),
                      angular_velocity(std::move(_angular_velocity))
    {

    }

    ///////////////////////////////////////////////////////////////////////
    IMUPreIntegrationMeasurement::
    IMUPreIntegrationMeasurement(Vector3 _acceleration_0,
                                 Vector3 _angular_velocity_0,
                                 Vector3 _linearized_ba,
                                 Vector3 _linearized_bg,
                                 double _sum_dt,
                                 Matrix15 _jacobian,
                                 Matrix15 _covariance,
                                 const Quaternion& _delta_q,
                                 Vector3 _delta_p,
                                 Vector3 _delta_v):
            acceleration_0(std::move(_acceleration_0)),
            angular_velocity_0(std::move(_angular_velocity_0)),
            linearized_ba(std::move(_linearized_ba)),
            linearized_bg(std::move(_linearized_bg)),
            sum_dt(_sum_dt),
            jacobian(std::move(_jacobian)),
            covariance(std::move(_covariance)),
            delta_q(_delta_q),
            delta_p(std::move(_delta_p)),
            delta_v(std::move(_delta_v))
    {

    }

}//end of SuperVIO::IMU