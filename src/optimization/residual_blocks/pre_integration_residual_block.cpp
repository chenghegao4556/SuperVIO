//
// Created by chenghe on 4/2/20.
//

#include <optimization/residual_blocks/pre_integration_residual_block.h>
namespace SuperVIO::Optimization
{
    typedef PreIntegrationResidualBlock::Ptr Ptr;
    typedef PreIntegrationResidualBlock::Type Type;

    ///////////////////////////////////////////////////////////////////////////////////
    Ptr PreIntegrationResidualBlock::
    Creat(const IMU::IMUPreIntegrationMeasurement& pre_integration_measurement,
          const Vector3& gravity_vector,
          const std::vector<ParameterBlockPtr>& parameter_blocks)
    {
        Ptr p(new PreIntegrationResidualBlock(pre_integration_measurement,
                                              gravity_vector,
                                              parameter_blocks));
        return p;
    }

    ///////////////////////////////////////////////////////////////////////////////////
    Type PreIntegrationResidualBlock::
    GetType() const
    {
        return Type::PreIntegration;
    }

    ///////////////////////////////////////////////////////////////////////////////////
    PreIntegrationResidualBlock::
    PreIntegrationResidualBlock(const IMU::IMUPreIntegrationMeasurement& pre_integration_measurement,
                                const Vector3& gravity_vector,
                                const std::vector<ParameterBlockPtr>& parameter_blocks):
        BaseResidualBlock(CostFunctionFactory::CreatPreIntegrationCostFunction(pre_integration_measurement,
                                                                               gravity_vector),
                          nullptr,
                          parameter_blocks)
    {
        ROS_ASSERT(parameter_blocks.size() == 4);
        ROS_ASSERT(parameter_blocks[0]->GetType() == BaseParametersBlock::Type::Pose);
        ROS_ASSERT(parameter_blocks[1]->GetType() == BaseParametersBlock::Type::SpeedBias);
        ROS_ASSERT(parameter_blocks[2]->GetType() == BaseParametersBlock::Type::Pose);
        ROS_ASSERT(parameter_blocks[3]->GetType() == BaseParametersBlock::Type::SpeedBias);
    }
}//end of SuperVIO
