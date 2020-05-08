//
// Created by chenghe on 4/4/20.
//

#include <optimization/parameter_blocks/pose_local_parameterization.h>
#include <utility/eigen_base.h>

namespace SuperVIO
{
namespace Optimization
{

    //////////////////////////////////////////////////////////////////////////////
    bool PoseLocalParameterization::
    Plus(const double* x,
         const double* delta_x,
         double* x_plus_delta_x) const
    {
        ConstMapVector3 p(x);
        ConstMapQuaternion q(x + 3);

        ConstMapVector3 dp(delta_x);

        const Quaternion dq = Utility::EigenBase::DeltaQ(ConstMapVector3(delta_x + 3));

        MapVector3 new_p(x_plus_delta_x);
        MapQuaternion new_q(x_plus_delta_x + 3);

        new_p = p + dp;
        new_q = (q * dq).normalized();

        return true;
    }

    //////////////////////////////////////////////////////////////////////////////
    bool PoseLocalParameterization::
    ComputeJacobian(const double* x, double* jacobian) const
    {
        MapMatrix76 j(jacobian);
        j.topRows<6>().setIdentity();
        j.bottomRows<1>().setZero();

        return true;
    }

    //////////////////////////////////////////////////////////////////////////////
    int PoseLocalParameterization::
    GlobalSize() const
    {
        return 7;
    }

    //////////////////////////////////////////////////////////////////////////////
    int PoseLocalParameterization::
    LocalSize() const
    {
        return 6;
    }

}//end of Optimization
}//end of SuperVIO