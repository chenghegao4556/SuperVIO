//
// Created by chenghe on 3/30/20.
//
#include <optimization/factors/marginalization_factor.h>
namespace SuperVIO::Optimization
{
    ///////////////////////////////////////////////////////////////////////////////////////////////////
    MarginalizationInformation::
    MarginalizationInformation():
        valid(false)
    {

    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////
    MarginalizationFactor::
    MarginalizationFactor(const MarginalizationInformation& marginalization_information):
                          marginalization_information_(marginalization_information),
                          residual_size_(std::accumulate(marginalization_information.local_block_sizes.begin(),
                                                         marginalization_information.local_block_sizes.end(),
                                                         static_cast<size_t>(0)))

    {
        for (const auto& global_size: marginalization_information_.global_block_sizes)
        {
            mutable_parameter_block_sizes()->push_back(static_cast<int>(global_size));
        }

        set_num_residuals(static_cast<int>(residual_size_));
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////
    bool MarginalizationFactor::
    Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        VectorX dx(residual_size_);
        for (size_t i = 0; i < marginalization_information_.global_block_sizes.size(); ++i)
        {
            size_t global_size    = marginalization_information_.global_block_sizes[i];
            size_t jacobian_index = marginalization_information_.jacobian_indices[i];
            const auto& old_data  = marginalization_information_.old_parameters_data[i];

            VectorX new_parameter = ConstMapVectorX(parameters[i], global_size);
            VectorX old_parameter = ConstMapVectorX(old_data, global_size);

            if (global_size != GlobalSize::Pose)
            {
                dx.segment(jacobian_index, global_size) = new_parameter - old_parameter;
            }
            else
            {
                Vector3 q_diff = Utility::EigenBase::Positify(
                        Quaternion(&old_data[3]).inverse() * Quaternion(&parameters[i][3])).vec();

                dx.segment<3>(jacobian_index + 0) = new_parameter.head<3>() - old_parameter.head<3>();
                dx.segment<3>(jacobian_index + 3) = 2.0 * q_diff;
            }
        }
        MapVectorX(residuals, residual_size_) = marginalization_information_.linearized_residuals +
                marginalization_information_.linearized_jacobians * dx;
        if (jacobians)
        {
            for (size_t i = 0; i < marginalization_information_.global_block_sizes.size(); ++i)
            {
                if (jacobians[i])
                {
                    size_t global_size = marginalization_information_.global_block_sizes[i];
                    size_t local_size  = marginalization_information_.local_block_sizes[i];
                    size_t jacobian_index = marginalization_information_.jacobian_indices[i];
                    MapDynamicMatrix jacobian(jacobians[i], residual_size_, global_size);
                    jacobian.setZero();
                    jacobian.leftCols(local_size) = marginalization_information_.linearized_jacobians.middleCols(
                            jacobian_index, local_size);
                }
            }
        }
        return true;
    }

}//end of SuperVIO