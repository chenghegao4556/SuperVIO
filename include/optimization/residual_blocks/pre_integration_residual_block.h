//
// Created by chenghe on 4/2/20.
//

#ifndef SUPER_VIO_PRE_INTEGRATION_RESIDUAL_BLOCK_H
#define SUPER_VIO_PRE_INTEGRATION_RESIDUAL_BLOCK_H

#include <optimization/factors/cost_function_factory.h>
#include <optimization/residual_blocks/base_residual_block.h>
namespace SuperVIO::Optimization
{
    class PreIntegrationResidualBlock: public BaseResidualBlock
    {
    public:
        static Ptr Creat(const IMU::IMUPreIntegrationMeasurement& pre_integration_measurement,
                         const Vector3& gravity_vector,
                         const std::vector<ParameterBlockPtr>& parameter_blocks);

        [[nodiscard]] Type GetType() const override;

    protected:
        explicit PreIntegrationResidualBlock(const IMU::IMUPreIntegrationMeasurement& pre_integration_measurement,
                                             const Vector3& gravity_vector,
                                             const std::vector<ParameterBlockPtr>& parameter_blocks);
    };
}//end of SuperVIO

#endif //SRC_PRE_INTEGRATION_RESIDUAL_BLOCK_H
