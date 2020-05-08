//
// Created by chenghe on 4/14/20.
//

#include <estimation/visual_imu_alignment.h>

namespace SuperVIO::Estimation
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    VisualIMUAligner::Result::
    Result():
        success(false),
        scale(-1),
        gravity_vector(Vector3::Zero())
    {

    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    VisualIMUAligner::Result VisualIMUAligner::
    SolveGravityVectorAndVelocities(const VIOStatesMeasurements& states_measurements,
                                    const std::map<Vision::StateKey, SFM::Pose>& visual_pose_map,
                                    const Vector3& p_i_c,
                                    const double& gravity_norm)
    {
        auto num_states = visual_pose_map.size() * 3 + 3 + 1;

        MatrixX A{num_states, num_states};
        A.setZero();
        VectorX b{num_states};
        b.setZero();
        /**
         * delta_pij = R_bi_w(P_wj - P_wi - V_wi * dt + 0.5 * g_w * dt^2)
         * delta_vij = R_bi_w(V_wj - V_wi + g_w * dt)
         *
         * convert world frame to camera 0 frame and world velocity to body velocity
         * delta_pij = R_bi_c0[s*(P_co_bj - P_co_bi) - R_c0_bi * V_bi * dt + 0.5 * g_c0 * dt^2)
         * delta_vij = R_bi_co(R_c0_bj * V_bi - R_c0_bi * V_bj + g_c0 * dt)
         * extrinsic parameters: R_b_c, P_b_c (camera ==> body(IMU))
         * R_c0_bk = R_c0_ck * R_b_c^-1
         * s * P_c0_bk = s * P_c0_ck - R_c0_bk * P_b_c
         * ==>
         * delta_pij = s * R_bi_c0 *(P_co_ci - P_co_cj) - R_bi_c0 * R_c0_bj * P_b_c + P_b_c - V_bi * dt + 0.5 * R_bi_c0 * g_c0 * dt^2
         * delta_vij = R_bi_co(R_c0_bj * V_bi - R_c0_bi * V_bj + g_c0 * dt)
         *
         * x = [v_bi, v_bj, g_c0, s]
         * b = [ delta_pij + R_bi_c0 * R_c0_bj * P_b_c - P_b_c,
         *                  delta_vij                         ]
         * H = [-dt,                  0, 0.5 * R_bi_c0 * dt^2, R_bi_c0 *(P_co_ci - P_co_cj),
         *       -I, -R_bi_co * R_c0_bi,         R_bi_co * dt,                            0]
         */
        auto visual_pose_iter_i = visual_pose_map.begin();
        size_t i = 0;
        for (;std::next(visual_pose_iter_i) != visual_pose_map.end(); ++visual_pose_iter_i, ++i)
        {
            auto visual_pose_iter_j = std::next(visual_pose_iter_i);

            MatrixX temp_A(6, 10);
            temp_A.setZero();
            VectorX temp_b(6);
            temp_b.setZero();
            auto pre_integration_iter = states_measurements.imu_pre_integration_measurements_map.find(
                    std::make_pair(visual_pose_iter_i->first, visual_pose_iter_j->first));
            ROS_ASSERT(pre_integration_iter != states_measurements.imu_pre_integration_measurements_map.end());


            auto dt = pre_integration_iter->second.sum_dt;
            const auto& r_i = visual_pose_iter_i->second.rotation.toRotationMatrix();
            const auto& r_j = visual_pose_iter_j->second.rotation.toRotationMatrix();
            const auto& p_i = visual_pose_iter_i->second.position;
            const auto& p_j = visual_pose_iter_j->second.position;

            temp_A.block<3, 3>(0, 0) = -dt * Matrix3::Identity();
            temp_A.block<3, 3>(0, 6) =  r_i.transpose() * dt * dt / 2.0 * Matrix3::Identity();
            temp_A.block<3, 1>(0, 9) =  r_i.transpose() * (p_j - p_i) / 100.0;
            temp_b.block<3, 1>(0, 0) =  pre_integration_iter->second.delta_p + r_i.transpose() * r_j * p_i_c- p_i_c;
            temp_A.block<3, 3>(3, 0) = -Matrix3::Identity();
            temp_A.block<3, 3>(3, 3) =  r_i.transpose() * r_j;
            temp_A.block<3, 3>(3, 6) =  r_i.transpose() * dt * Matrix3::Identity();
            temp_b.block<3, 1>(3, 0) =  pre_integration_iter->second.delta_v;

            Matrix6 cov_inv;
            cov_inv.setIdentity();

            MatrixX r_A = temp_A.transpose() * cov_inv * temp_A;
            VectorX r_b = temp_A.transpose() * cov_inv * temp_b;

            A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
            b.segment<6>(i * 3) += r_b.head<6>();

            A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
            b.tail<4>() += r_b.tail<4>();

            A.block<6, 4>(i * 3, num_states - 4) += r_A.topRightCorner<6, 4>();
            A.block<4, 6>(num_states - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
        }
        A = A * 1000.0;
        b = b * 1000.0;

        VectorX solution = A.ldlt().solve(b);
        auto scale = solution(num_states - 1) / 100.0;
        ROS_INFO_STREAM(" first estimated scale: "<<scale);

        Vector3 gravity_vector = solution.segment<3>(num_states - 4);

        ROS_INFO_STREAM(" first estimated gravity vector: " << gravity_vector.norm() << " " << gravity_vector.transpose());
        if(scale < 0)
        {
            return Result();
        }


        return RefineGravityVectorAndVelocities(states_measurements, visual_pose_map, p_i_c, gravity_vector, gravity_norm);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    VisualIMUAligner::Result VisualIMUAligner::
    RefineGravityVectorAndVelocities(const VIOStatesMeasurements& states_measurements,
                                     const std::map<Vision::StateKey, SFM::Pose>& visual_pose_map,
                                     const Vector3& p_i_c,
                                     const Vector3& gravity_vector,
                                     const double& gravity_norm)
    {
        Vector3 optimized_gravity_vector = gravity_vector.normalized() * gravity_norm;
        Vector3 lx, ly;
        auto num_states = visual_pose_map.size() * 3 + 2 + 1;

        MatrixX A(num_states, num_states);
        A.setZero();
        VectorX b(num_states);
        b.setZero();

        VectorX solution;
        for(int k = 0; k < 4; k++)
        {
            MatrixX lxly(3, 2);
            lxly = TangentBasis(optimized_gravity_vector);
            size_t i = 0;
            /**
             * lxly = [b1, b2]
             * g_c = |g_c| * g_c.normalize() + [w1, w2]^T * lxly
             * dg = [w1, w2]
             * x = [v_bi, v_bj, dg, s]
             * b = [ delta_pij + R_bi_c0 * R_c0_bj * P_b_c - P_b_c - 0.5 * R_bi_c0 * dt^2 * |g_c| * g_c.normalize(),
             *                  delta_vij - R_bi_co * dt * |g_c| * g_c.normalize()                                 ]
             * H = [-dt,                  0, 0.5 * R_bi_c0 * dt^2 * lxly, R_bi_c0 *(P_co_ci - P_co_cj),
             *       -I, -R_bi_co * R_c0_bi,         R_bi_co * dt * lxly,                            0]
             */
            auto visual_pose_iter_i = visual_pose_map.begin();
            for (;std::next(visual_pose_iter_i) != visual_pose_map.end(); ++visual_pose_iter_i, ++i)
            {
                auto visual_pose_iter_j = std::next(visual_pose_iter_i);

                MatrixX temp_A(6, 9);
                temp_A.setZero();
                VectorX temp_b(6);
                temp_b.setZero();
                auto pre_integration_iter = states_measurements.imu_pre_integration_measurements_map.find(
                        std::make_pair(visual_pose_iter_i->first, visual_pose_iter_j->first));
                ROS_ASSERT(pre_integration_iter != states_measurements.imu_pre_integration_measurements_map.end());

                double dt = pre_integration_iter->second.sum_dt;
                const auto& r_i = visual_pose_iter_i->second.rotation.toRotationMatrix();
                const auto& r_j = visual_pose_iter_j->second.rotation.toRotationMatrix();
                const auto& p_i = visual_pose_iter_i->second.position;
                const auto& p_j = visual_pose_iter_j->second.position;


                temp_A.block<3, 3>(0, 0) = -dt * Matrix3::Identity();
                temp_A.block<3, 2>(0, 6) = r_i.transpose() * dt * dt / 2.0 * Matrix3::Identity() * lxly;
                temp_A.block<3, 1>(0, 8) = r_i.transpose() * (p_j - p_i) / 100.0;
                temp_b.block<3, 1>(0, 0) = pre_integration_iter->second.delta_p + r_i.transpose() * r_j * p_i_c - p_i_c -
                        r_i.transpose() * dt * dt / 2.0 * optimized_gravity_vector;

                temp_A.block<3, 3>(3, 0) = -Matrix3::Identity();
                temp_A.block<3, 3>(3, 3) =  r_i.transpose() * r_j;
                temp_A.block<3, 2>(3, 6) =  r_i.transpose() * dt * Matrix3::Identity() * lxly;
                temp_b.block<3, 1>(3, 0) =  pre_integration_iter->second.delta_v -
                        r_i.transpose() * dt * Matrix3::Identity() * optimized_gravity_vector;


                Matrix6 cov_inv;
                cov_inv.setIdentity();

                MatrixX r_A = temp_A.transpose() * cov_inv * temp_A;
                VectorX r_b = temp_A.transpose() * cov_inv * temp_b;

                A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
                b.segment<6>(i * 3) += r_b.head<6>();

                A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
                b.tail<3>() += r_b.tail<3>();

                A.block<6, 3>(i * 3, num_states - 3) += r_A.topRightCorner<6, 3>();
                A.block<3, 6>(num_states - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
            }
            A = A * 1000.0;
            b = b * 1000.0;
            solution = A.ldlt().solve(b);
            VectorX dg = solution.segment<2>(num_states - 3);
            optimized_gravity_vector = (optimized_gravity_vector + lxly * dg).normalized() * gravity_norm;
        }
        double scale = (solution.tail<1>())(0) / 100.0;
        ROS_INFO_STREAM(" refined scale: "<<scale);
        ROS_INFO_STREAM(" refined gravity vector" << optimized_gravity_vector.norm() << " " << optimized_gravity_vector.transpose());
        if(scale < 0.0 )
        {
            return Result();
        }
        Result result;
        result.gravity_vector = optimized_gravity_vector;
        result.success = true;
        result.scale = scale;
        size_t i = 0;
        for(const auto& pose: visual_pose_map)
        {
            auto velocity = solution.segment<3>(i * 3);
            result.velocity_map.insert(std::make_pair(pose.first, velocity));
            ++i;
        }
        return result;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    Vector3  VisualIMUAligner::
    SolveGyroscopeBias(const VIOStatesMeasurements& states_measurements,
                       const std::map<Vision::StateKey, SFM::Pose>& visual_pose_map)
    {
        typedef IMU::PreIntegrator::StateJacobianOrder StateJacobianOrder;
        Matrix3 A = Matrix3::Zero();
        Vector3 b = Vector3::Zero();
        auto visual_pose_iter_i = visual_pose_map.begin();
        for(;std::next(visual_pose_iter_i) != visual_pose_map.end(); ++visual_pose_iter_i)
        {
            /**
             * idea: minimize q_ij^-1(compute by PnP) * delta_q,
             *       subject to q_ij^-1(compute by PnP) * delta_q = I
             * first order taylor expansion of delta_q:
             * delta_q = q_ij(real rotation between i and j) * [1, 1/2 * d(delta_q)/d(gyro_bias) * gyro_bias]^T
             * 1/2 * d(delta_q)/d(gyro_bias) * gyro_bias = [delta_q^-1 * q_ij].vec
             * ==> H * gyro_bias = b
             *     H = 1/2 * d(delta_q)/d(gyro_bias)
             *     b = [delta_q^-1 * q_ij].vec
             */
            auto visual_pose_iter_j = std::next(visual_pose_iter_i);
            auto visual_q_ij = visual_pose_iter_i->second.rotation.inverse() *
                    visual_pose_iter_j->second.rotation;

            auto pre_integration_iter = states_measurements.imu_pre_integration_measurements_map.find(
                    std::make_pair(visual_pose_iter_i->first, visual_pose_iter_j->first));
            ROS_ASSERT(pre_integration_iter != states_measurements.imu_pre_integration_measurements_map.end());

            Matrix3 temp_A = pre_integration_iter->second.jacobian.block<3, 3>(StateJacobianOrder::O_R,
                    StateJacobianOrder::O_BG);
            Vector3 temp_b = 2.0 * (pre_integration_iter->second.delta_q.inverse() * visual_q_ij).vec();

            A += temp_A.transpose() * temp_A;
            b += temp_A.transpose() * temp_b;
        }
        Vector3 delta_bg = A.ldlt().solve(b);
        ROS_INFO_STREAM("gyroscope bias initial calibration " << delta_bg.transpose());

        return delta_bg;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    MatrixX VisualIMUAligner::
    TangentBasis(Vector3 &gravity_vector)
    {
        Vector3 b, c;
        Vector3 a = gravity_vector.normalized();
        Vector3 tmp(0, 0, 1);
        if(a == tmp)
            tmp << 1, 0, 0;
        b = (tmp - a * (a.transpose() * tmp)).normalized();
        c = a.cross(b);
        MatrixX bc(3, 2);
        bc.block<3, 1>(0, 0) = b;
        bc.block<3, 1>(0, 1) = c;
        return bc;
    }
}//end of SuperVIO::Estimation