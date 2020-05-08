//
// Created by chenghe on 3/30/20.
//
#include <thread>
#include <optimization/marginalizer.h>
namespace SuperVIO::Optimization
{
    //////////////////////////////////////////////////////////////////////////////////////////////////
    Marginalizer::OrderedParameters::
    OrderedParameters():
        drop_landmark_size(0),
        drop_pose_size(0),
        drop_jacobian_size(0),
        keep_jacobian_size(0),
        total_jacobian_size(0)
    {

    }

    //////////////////////////////////////////////////////////////////////////////////////////////////
    MarginalizationInformation  Marginalizer::
    Construct(const std::vector<ParameterBlockPtr>& parameter_blocks,
              const std::vector<ResidualBlockPtr>&  residual_blocks,
              const std::set<size_t>& drop_set,
              const size_t& num_threads)
    {
        auto ordered_parameters = SetOrder(parameter_blocks, drop_set);
        auto full_information = ConstructHessian(ordered_parameters, residual_blocks, num_threads);
        MatrixX marginalized_information_matrix = full_information.first;
        VectorX marginalized_information_vector = full_information.second;
        if(ordered_parameters.drop_landmark_size != 0)
        {
            auto result = Utility::SchurComplement::Compute(full_information.first,
                                                            full_information.second,
                                                            ordered_parameters.drop_landmark_size,
                                                           (ordered_parameters.total_jacobian_size -
                                                            ordered_parameters.drop_landmark_size));
            marginalized_information_matrix = result.information_matrix;
            marginalized_information_vector = result.information_vector;
        }
        if(ordered_parameters.drop_pose_size != 0)
        {
            auto result = Utility::SchurComplement::Compute(marginalized_information_matrix,
                                                            marginalized_information_vector,
                                                            ordered_parameters.drop_pose_size,
                                                            ordered_parameters.keep_jacobian_size);
            marginalized_information_matrix = result.information_matrix;
            marginalized_information_vector = result.information_vector;
        }

        auto result = Utility::HessianMatrixDecompose::Compute(marginalized_information_matrix,
                                                               marginalized_information_vector);

        return CreatMarginalizationInformation(result.linearized_jacobians,
                                               result.linearized_residuals,
                                               ordered_parameters);
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////
    Marginalizer::OrderedParameters Marginalizer::
    SetOrder(const std::vector<ParameterBlockPtr>& parameter_blocks,
             const std::set<size_t>& drop_set)
    {
        OrderedParameters ordered_parameters;
        for(size_t i = 0; i < parameter_blocks.size(); ++i)
        {
            if(drop_set.find(i) != drop_set.end())
            {
                if(parameter_blocks[i]->GetType() == BaseParametersBlock::Type::InverseDepth)
                {
                    ordered_parameters.drop_landmarks.push_back(parameter_blocks[i]);
                }
                else
                {
                    ordered_parameters.drop_poses.push_back(parameter_blocks[i]);
                }
            }
            else
            {
                ordered_parameters.keep_poses.push_back(parameter_blocks[i]);
            }
        }
        ordered_parameters.drop_jacobian_size = 0;

        for(const auto& drop_landmark: ordered_parameters.drop_landmarks)
        {
            drop_landmark->SetJacobianId(ordered_parameters.drop_jacobian_size);
            ordered_parameters.drop_jacobian_size += drop_landmark->GetLocalSize();
        }

        ordered_parameters.drop_landmark_size = ordered_parameters.drop_jacobian_size;

        for(const auto& drop_pose: ordered_parameters.drop_poses)
        {
            drop_pose->SetJacobianId(ordered_parameters.drop_jacobian_size);
            ordered_parameters.drop_jacobian_size += drop_pose->GetLocalSize();
        }

        ordered_parameters.drop_pose_size = ordered_parameters.drop_jacobian_size -
                ordered_parameters.drop_landmark_size;
        ordered_parameters.total_jacobian_size = ordered_parameters.drop_jacobian_size;

        for(const auto& keep_pose: ordered_parameters.keep_poses)
        {
            keep_pose->SetJacobianId(ordered_parameters.total_jacobian_size);
            ordered_parameters.total_jacobian_size += keep_pose->GetLocalSize();
        }

        ordered_parameters.keep_jacobian_size = ordered_parameters.total_jacobian_size -
                ordered_parameters.drop_jacobian_size;

        return ordered_parameters;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////
    void Marginalizer::ConstructHessianThread::
    Evaluate()
    {
        information_matrix = MatrixX(jacobian_size, jacobian_size);
        information_vector = VectorX(jacobian_size);

        information_matrix.setZero();
        information_vector.setZero();

        for(const auto& residual_block: residual_per_thread)
        {
            residual_block->Evaluate();
        }
        for (const auto& residual_block: residual_per_thread)
        {
            for(size_t i = 0; i < residual_block->GetParameterBlock().size(); ++i)
            {
                const auto& parameter_i = residual_block->GetParameterBlock()[i];
                const size_t jacobian_id_i = parameter_i->GetJacobianId();
                const size_t local_size_i = parameter_i->GetLocalSize();
                const auto& jacobian_i = residual_block->GetJacobians()[i].leftCols(local_size_i);

                for(size_t j = i; j < residual_block->GetParameterBlock().size(); ++j)
                {
                    const auto& parameter_j = residual_block->GetParameterBlock()[j];
                    const size_t jacobian_id_j = parameter_j->GetJacobianId();
                    const size_t local_size_j = parameter_j->GetLocalSize();
                    const auto& jacobian_j = residual_block->GetJacobians()[j].leftCols(local_size_j);

                    information_matrix.block(jacobian_id_i, jacobian_id_j, local_size_i, local_size_j)
                            += jacobian_i.transpose() * jacobian_j;

                    if (i != j)
                    {
                        information_matrix.block(jacobian_id_j, jacobian_id_i, local_size_j, local_size_i)
                                = information_matrix.block(jacobian_id_i, jacobian_id_j, local_size_i, local_size_j).transpose();
                    }
                }

                information_vector.segment(jacobian_id_i, local_size_i)
                        += jacobian_i.transpose() * residual_block->GetResiduals();
            }
        }
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////
    std::pair<MatrixX, VectorX> Marginalizer::
    ConstructHessian(const OrderedParameters& ordered_parameters,
                     const std::vector<ResidualBlockPtr>& residual_blocks,
                     const size_t& num_threads)
    {
        std::vector<std::thread> threads;
        std::vector<ConstructHessianThread> hessian_constructors;
        ROS_ASSERT(num_threads != 0);
        hessian_constructors.resize(num_threads);
        int i = 0;
        for (const auto& residual: residual_blocks)
        {
            hessian_constructors[i].residual_per_thread.push_back(residual);
            i++;
            i = i % num_threads;
        }
        for (auto& hessian_constructor: hessian_constructors)
        {
            hessian_constructor.jacobian_size = ordered_parameters.total_jacobian_size;
            threads.emplace_back(std::bind(&ConstructHessianThread::Evaluate, &hessian_constructor));
        }
        for(auto& thread: threads)
        {
            thread.join();
        }

        MatrixX information_matrix(ordered_parameters.total_jacobian_size,
                ordered_parameters.total_jacobian_size);
        VectorX information_vector(ordered_parameters.total_jacobian_size);

        information_matrix.setZero();
        information_vector.setZero();

        for(const auto& hessian_constructor: hessian_constructors)
        {
            information_matrix += hessian_constructor.information_matrix;
            information_vector += hessian_constructor.information_vector;
        }

        return std::make_pair(information_matrix, information_vector);

    }

    //////////////////////////////////////////////////////////////////////////////////////////////////
    MarginalizationInformation Marginalizer::
    CreatMarginalizationInformation(const MatrixX& linearized_jacobians,
                                      const VectorX& linearized_residuals,
                                      const OrderedParameters& ordered_parameters)
    {
        MarginalizationInformation marginalization_information;

        for(const auto& keep_pose: ordered_parameters.keep_poses)
        {
            size_t new_jacobian_index = keep_pose->GetJacobianId() -
                    ordered_parameters.drop_jacobian_size;
            keep_pose->SetJacobianId(new_jacobian_index);
            marginalization_information.local_block_sizes.push_back(keep_pose->GetLocalSize());
            marginalization_information.global_block_sizes.push_back(keep_pose->GetGlobalSize());
            marginalization_information.jacobian_indices.push_back(keep_pose->GetJacobianId());

            auto* data = new double[keep_pose->GetGlobalSize()];
            memcpy(data, keep_pose->GetData(), sizeof(double) * keep_pose->GetGlobalSize());
            marginalization_information.old_parameters_data.push_back(data);
            marginalization_information.parameter_blocks.push_back(keep_pose);
        }
        marginalization_information.linearized_jacobians = linearized_jacobians;
        marginalization_information.linearized_residuals = linearized_residuals;

        marginalization_information.valid = true;
        return marginalization_information;
    }
}//end of Optimization
