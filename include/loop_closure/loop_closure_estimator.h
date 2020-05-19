//
// Created by chenghe on 5/12/20.
//

#ifndef SUPER_VIO_LOOP_CLOSURE_ESTIMATOR_H
#define SUPER_VIO_LOOP_CLOSURE_ESTIMATOR_H
#include <optimization/residual_blocks/residual_block_factory.h>
#include <optimization/optimizer.h>
#include <optimization/helper.h>
namespace SuperVIO::LoopClosure
{
    class Pose
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Pose(const Quaternion& _rotation, Vector3 _position);
        Quaternion rotation;
        Vector3    position;
    };//end of Pose

    class PoseGraph
    {
    public:

        std::map<IMU::StateKey, Pose> pose_vertices;
        std::map<IMU::StatePairKey, Pose> loop_edges;
        std::map<IMU::StatePairKey, Pose> sequence_edges;
    };//end of PoseGraph

    class LoopClosureEstimator
    {
    public:
        typedef Optimization::ResidualBlockFactory::Ptr  ResidualBlockPtr;
        typedef Optimization::ParameterBlockFactory::Ptr ParameterBlockPtr;

        class Parameters
        {
        public:
            double sequence_position_var;
            double sequence_rotation_var;
            double loop_position_var;
            double loop_rotation_var;
        };//end of Parameters

        static void Evaluate();
    protected:
        static bool FindLoop();
        static void EstimateRelativePose();

        static PoseGraph Optimize(const PoseGraph& pose_graph, const Parameters& parameters);

        static std::vector<ResidualBlockPtr>
        CreatResidual(const std::map<IMU::StateKey, ParameterBlockPtr>& vertices,
                      const std::map<IMU::StatePairKey, Pose>& edges,
                      const double& rotation_var,
                      const double& position_var);
    };
}//end of SuperVIO::LoopClosure
#endif //SUPER_VIO_LOOP_CLOSURE_ESTIMATOR_H
