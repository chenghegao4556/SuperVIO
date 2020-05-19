//
// Created by chenghe on 5/12/20.
//
#include <loop_closure/loop_closure_estimator.h>

namespace SuperVIO::LoopClosure
{
    /////////////////////////////////////////////////////////////////////////////////////////
    typedef Optimization::ParameterBlockFactory::Ptr ParameterBlockPtr;
    typedef Optimization::ResidualBlockFactory::Ptr  ResidualBlockPtr;

    /////////////////////////////////////////////////////////////////////////////////////////
    Pose::
    Pose(const Quaternion& _rotation, Vector3 _position):
        rotation(_rotation),
        position(std::move(_position))
    {

    }

    /////////////////////////////////////////////////////////////////////////////////////////
    void LoopClosureEstimator::
    Evaluate()
    {
        //! find loop candidates

        //! estimate relative pose

        //! optimize
    }

    /////////////////////////////////////////////////////////////////////////////////////////
    bool LoopClosureEstimator::
    FindLoop()
    {
        //! find loop and check
        //! collect possible frames for query
        //! bag of word query
    }

    /////////////////////////////////////////////////////////////////////////////////////////
    void LoopClosureEstimator::
    EstimateRelativePose()
    {
        //! feature matching and fundamental filter

        //! forward PnP(T_query_reference)

        //! backward PnP(T_reference_query)

        //! check || T_query_reference * T_reference_query || < threshold
    }

    /////////////////////////////////////////////////////////////////////////////////////////
    PoseGraph  LoopClosureEstimator::
    Optimize(const PoseGraph& pose_graph, const Parameters& parameters)
    {
        //! pose graph
        std::map<IMU::StateKey, ParameterBlockPtr> parameter_block_map;

        std::vector<ParameterBlockPtr> parameter_blocks;
        for(const auto& pose: pose_graph.pose_vertices)
        {
            auto pose_ptr = Optimization::ParameterBlockFactory::CreatPose(
                    pose.second.rotation, pose.second.position);
            parameter_block_map.insert(std::make_pair(pose.first, pose_ptr));
            parameter_blocks.push_back(pose_ptr);
        }

        parameter_block_map.begin()->second->SetFixed();

        auto residual_blocks = CreatResidual(parameter_block_map, pose_graph.sequence_edges,
                parameters.sequence_position_var, parameters.sequence_rotation_var);
        auto loop_residual_blocks = CreatResidual(parameter_block_map, pose_graph.loop_edges,
                parameters.loop_position_var, parameters.loop_rotation_var);

        for(const auto& loop: loop_residual_blocks)
        {
            residual_blocks.push_back(loop);
        }

        Optimization::Optimizer::Options options;
        options.linear_solver_type = "sparse_normal_cholesky";
        options.max_iteration = 5;
        Optimization::Optimizer::Construct(options, parameter_blocks, residual_blocks);

        auto new_pose_graph = pose_graph;
        for(const auto& pose_ptr: parameter_block_map)
        {
            auto pose = Optimization::Helper::GetPoseFromParameterBlock(pose_ptr.second);
            new_pose_graph.pose_vertices.at(pose_ptr.first).rotation = pose.first;
            new_pose_graph.pose_vertices.at(pose_ptr.first).position = pose.second;
        }

        return new_pose_graph;

    }

    /////////////////////////////////////////////////////////////////////////////////////////
    std::vector<ResidualBlockPtr> LoopClosureEstimator::
    CreatResidual(const std::map<IMU::StateKey, ParameterBlockPtr>& vertices,
                  const std::map<IMU::StatePairKey, Pose>& edges,
                  const double& rotation_var,
                  const double& position_var)
    {
        std::vector<ResidualBlockPtr>  residual_blocks;

        for(const auto& relative_pose: edges)
        {
            const auto& time_i = relative_pose.first.first;
            const auto& time_j = relative_pose.first.second;
            const auto& pose_i = vertices.at(time_i);
            const auto& pose_j = vertices.at(time_j);

            std::vector<ParameterBlockPtr> pose_ptrs{pose_i, pose_j};
            auto residual_ptr = Optimization::ResidualBlockFactory::CreatRelativePose(
                    relative_pose.second.rotation, relative_pose.second.position,
                    position_var, rotation_var, pose_ptrs);
            residual_blocks.push_back(residual_ptr);
        }

        return residual_blocks;
    }
}//end of SuperVIO::LoopClosure
