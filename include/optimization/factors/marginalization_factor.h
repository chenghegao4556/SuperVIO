//
// Created by chenghe on 3/28/20.
//

#ifndef SUPER_VIO_MARGINALIZATION_FACTOR_H
#define SUPER_VIO_MARGINALIZATION_FACTOR_H

#include <ceres/ceres.h>
#include <utility/eigen_type.h>
#include <utility/eigen_base.h>
#include <optimization/parameter_blocks/base_parameter_block.h>
namespace SuperVIO::Optimization
{
    class MarginalizationInformation
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef BaseParametersBlock::Ptr ParameterBlockPtr;

        MarginalizationInformation();

        bool valid;
        std::vector<size_t> local_block_sizes;
        std::vector<size_t> global_block_sizes;
        std::vector<size_t> jacobian_indices;
        std::vector<double const *> old_parameters_data;
        MatrixX linearized_jacobians;
        VectorX linearized_residuals;
        std::vector<ParameterBlockPtr> parameter_blocks;
    };//end of MarginalizationInformation

    class MarginalizationFactor : public ceres::CostFunction
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        enum GlobalSize
        {
            Pose = 7,
            SpeedBias = 9
        };

        MarginalizationFactor(const MarginalizationInformation& marginalization_information);


        bool Evaluate(double const *const *parameters,
                      double *residuals,
                      double **jacobians) const override;

    private:

        const MarginalizationInformation marginalization_information_;
        const size_t residual_size_;
    };//end of MarginalizationFactor
}//end of SuperVIO


#endif //SUPER_VIO_MARGINALIZATION_FACTOR_H
