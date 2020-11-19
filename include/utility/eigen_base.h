//
// Created by chenghe on 3/6/20.
//

#ifndef SUPER_VIO_QUATERNION_BASE_H
#define SUPER_VIO_QUATERNION_BASE_H

#include <ros/ros.h>
#include <utility/eigen_type.h>
namespace SuperVIO::Utility
{
    class EigenBase
    {
    public:
        template <typename Derived>
        static Eigen::Quaternion<typename Derived::Scalar> DeltaQ(const Eigen::MatrixBase<Derived> &theta)
        {
            typedef typename Derived::Scalar Scalar_t;

            Eigen::Quaternion<Scalar_t> dq;
            Eigen::Matrix<Scalar_t, 3, 1> half_theta = theta;
            half_theta /= static_cast<Scalar_t>(2.0);
            dq.w() = static_cast<Scalar_t>(1.0);
            dq.x() = half_theta.x();
            dq.y() = half_theta.y();
            dq.z() = half_theta.z();
            dq.normalize();
            return dq;
        }

        template <typename Derived>
        static Eigen::Matrix<typename Derived::Scalar, 3, 3> SkewSymmetric(const Eigen::MatrixBase<Derived> &q)
        {
            Eigen::Matrix<typename Derived::Scalar, 3, 3> ans;
            ans <<  typename Derived::Scalar(0), -q(2),   q(1),
                    q(2),  typename Derived::Scalar(0),  -q(0),
                   -q(1), q(0),    typename Derived::Scalar(0);
            return ans;
        }

        template <typename Derived>
        static Eigen::Transform<typename Derived::Scalar, 3, Eigen::Isometry>
        ToPose3(const Eigen::Quaternion<typename Derived::Scalar>& q,
                const Eigen::MatrixBase<Derived> &t)
        {
            Eigen::Transform<typename Derived::Scalar, 3, Eigen::Isometry> pose3 =
                    Eigen::Transform<typename Derived::Scalar, 3, Eigen::Isometry>::Identity();
            Eigen::Matrix<typename Derived::Scalar, 3, 3> r = q.toRotationMatrix();
            pose3.rotate(r);
            pose3.pretranslate(t);

            return pose3;
        }

        template <typename Derived>
        static Eigen::Quaternion<typename Derived::Scalar> Positify(const Eigen::QuaternionBase<Derived> &q)
        {
            Eigen::Quaternion<typename Derived::Scalar> new_q = q;
            if(q.template w() < (typename Derived::Scalar)(0.0))
            {
                new_q = Eigen::Quaternion<typename Derived::Scalar>(-q.w(), -q.x(), -q.y(), -q.z());
            }
            return new_q;
        }

        template <typename Derived>
        static Eigen::Matrix<typename Derived::Scalar, 4, 4> Qleft(const Eigen::QuaternionBase<Derived> &q)
        {
            Eigen::Matrix<typename Derived::Scalar, 4, 4> ans;
            ans(0, 0) = q.w();
            ans.template block<1, 3>(0, 1) = -q.vec().transpose();
            ans.template block<3, 1>(1, 0) =  q.vec();
            ans.template block<3, 3>(1, 1) =  q.w() *
                    Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() + SkewSymmetric(q.vec());
            return ans;
        }

        template <typename Derived>
        static Eigen::Matrix<typename Derived::Scalar, 4, 4> Qright(const Eigen::QuaternionBase<Derived> &q)
        {
            Eigen::Matrix<typename Derived::Scalar, 4, 4> ans;
            ans(0, 0) = q.w();
            ans.template block<1, 3>(0, 1) = -q.vec().transpose();
            ans.template block<3, 1>(1, 0) =  q.vec();
            ans.template block<3, 3>(1, 1) =  q.w() *
                    Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() - SkewSymmetric(q.vec());
            return ans;
        }

        static Vector3 Rotation2Euler(const Matrix3 &rotation_matrix)
        {
            Vector3 n = rotation_matrix.col(0);
            Vector3 o = rotation_matrix.col(1);
            Vector3 a = rotation_matrix.col(2);

            Vector3 ypr;

            double y = atan2(n(1), n(0));
            double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
            double r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));

            ypr(0) = y;
            ypr(1) = p;
            ypr(2) = r;

            return ypr / M_PI * 180.0;
        }

        static Vector3 Quaternion2Euler(const Quaternion& q)
        {
            return Rotation2Euler(q.toRotationMatrix());
        }

        template <typename Derived>
        static Eigen::Matrix<typename Derived::Scalar, 3, 3> EulerToRotationMatrix(const Eigen::MatrixBase<Derived> &ypr)
        {
            typedef typename Derived::Scalar Scalar_t;

            Scalar_t y = ypr(0) / 180.0 * M_PI;
            Scalar_t p = ypr(1) / 180.0 * M_PI;
            Scalar_t r = ypr(2) / 180.0 * M_PI;

            Eigen::Matrix<Scalar_t, 3, 3> Rz;
            Rz << cos(y), -sin(y), 0,
                    sin(y), cos(y), 0,
                    0, 0, 1;

            Eigen::Matrix<Scalar_t, 3, 3> Ry;
            Ry << cos(p), 0., sin(p),
                    0., 1., 0.,
                    -sin(p), 0., cos(p);

            Eigen::Matrix<Scalar_t, 3, 3> Rx;
            Rx << 1., 0., 0.,
                    0., cos(r), -sin(r),
                    0., sin(r), cos(r);

            return Rz * Ry * Rx;
        }

        template <typename Derived>
        static Eigen::Quaternion<typename Derived::Scalar> EulerToQuaternion(const Eigen::MatrixBase<Derived> &euler)
        {
            typedef typename Derived::Scalar Scalar_t;

            return Eigen::Quaternion<Scalar_t>(EulerToRotationMatrix(euler));
        }

        static Quaternion
        GravityVector2Quaternion(const Vector3 &gravity_vector)
        {
            Matrix3 R0;
            Vector3 ng1 = gravity_vector.normalized();
            Vector3 ng2{0, 0, 1.0};
            R0 = Quaternion::FromTwoVectors(ng1, ng2).toRotationMatrix();
            auto yaw = Rotation2Euler(R0).x();
            R0 = EulerToRotationMatrix(Vector3{-yaw, 0, 0}) * R0;

            return Quaternion(R0);
        }



        static Quaternion
        Formalize(const Quaternion& quat)
        {
            Quaternion q = quat;
            q.normalize();

            if (q.w() < double(0.0))
            {
                q.w() *= -double(1.0);
                q.vec() *= -double(1.0);
            }

            assert(!q.coeffs().hasNaN());
            return q;
        }

        static Vector3
        LogMap(const Quaternion& quat)
        {
            assert(!quat.coeffs().hasNaN());
            Quaternion q = Formalize(quat);
            double norm = q.vec().norm();
            double epsilon = double(1e-7);

            if (norm < epsilon)
            {
                return Vector3::Zero();
            }
            else
            {
                // we have to use acos instead of std::acos
                // if we want to ulilize ceres's AD ability
                Vector3 v = acos(q.w()) * q.vec() / norm;
                assert(!v.hasNaN());
                return v;
            }
        }

        static Quaternion
        ExpMap(const Vector3& v)
        {
            assert(!v.hasNaN());
            Quaternion quat;

            if (v[0] == double(0.0) && v[1] == double(0.0) && v[2] == double(0.0))
            {
                quat = Quaternion(double(1), double(0), double(0), double(0));
            }
            else
            {
                double norm = v.norm();
                // for AD
                double a = sin(norm) / norm;
                quat.w() = cos(norm);
                quat.x() = a * v[0];
                quat.y() = a * v[1];
                quat.z() = a * v[2];
            }

            assert(!quat.coeffs().hasNaN());
            return quat;
        }
    };//end of QuaternionBase
}//end of SuperVIO

#endif //SUPER_VIO_QUATERNION_BASE_H
