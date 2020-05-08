//
// Created by chenghe on 3/31/20.
//

#ifndef SUPER_VIO_BASE_PARAMETER_BLOCK_H
#define SUPER_VIO_BASE_PARAMETER_BLOCK_H
#include <ros/ros.h>
#include <ceres/ceres.h>
#include <utility/eigen_type.h>
namespace SuperVIO::Optimization
{
    class BaseParametersBlock
    {
    public:
        enum class Type
        {
            Pose = 0,
            SpeedBias = 1,
            InverseDepth = 2,
        };//end of GroupId

        enum class GlobalSize
        {
            InverseDepth = 1,
            Pose = 7,
            SpeedBias = 9,
        };

        enum class LocalSize
        {
            InverseDepth = 1,
            Pose = 6,
            SpeedBias = 9,
        };

        typedef std::shared_ptr<BaseParametersBlock> Ptr;
        typedef std::shared_ptr<const BaseParametersBlock> ConstPtr;

        virtual ~BaseParametersBlock();
        [[nodiscard]] virtual Type GetType() const =0;
        [[nodiscard]] virtual size_t GetGlobalSize() const = 0;
        [[nodiscard]] virtual size_t GetLocalSize() const = 0;

        [[nodiscard]] size_t GetJacobianId() const;

        double* GetData();
        [[nodiscard]] virtual ceres::LocalParameterization* GetLocalParameterization() const = 0;
        [[nodiscard]] bool IsValid() const;
        [[nodiscard]] bool IsFixed() const;

        void SetJacobianId(size_t jacobian_id);
        void SetFixed();
        void SetVariable();


    protected:
        void SetValid(bool is_valid);
        explicit BaseParametersBlock(size_t _global_size);

        size_t jacobian_id_;
        bool valid_;
        bool fixed_;
        double* data_;
    };//end of ParametersBlock

}//end of SuperVIO

#endif //SUPER_VIO_BASE_PARAMETER_BLOCK_H
