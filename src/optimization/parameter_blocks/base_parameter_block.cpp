//
// Created by chenghe on 3/31/20.
//

#include <optimization/parameter_blocks/base_parameter_block.h>
namespace SuperVIO::Optimization
{
    ///////////////////////////////////////////////////////////////////////////////////
    BaseParametersBlock::
    BaseParametersBlock(size_t _global_size):
                            jacobian_id_(0),
                            valid_(false),
                            fixed_(false),
                            data_(new double[_global_size]())
    {

    }

    ///////////////////////////////////////////////////////////////////////////////////
    BaseParametersBlock::
    ~BaseParametersBlock()
    {
        delete [] data_;
    }

    ///////////////////////////////////////////////////////////////////////////////////
    size_t BaseParametersBlock::
    GetJacobianId() const
    {
        return jacobian_id_;
    }

    ///////////////////////////////////////////////////////////////////////////////////
    double* BaseParametersBlock::
    GetData()
    {
        return data_;
    }

    ///////////////////////////////////////////////////////////////////////////////////
    bool BaseParametersBlock::
    IsValid() const
    {
        return valid_;
    }

    ///////////////////////////////////////////////////////////////////////////////////
    bool BaseParametersBlock::
    IsFixed() const
    {
        return fixed_;
    }

    ///////////////////////////////////////////////////////////////////////////////////
    void BaseParametersBlock::
    SetFixed()
    {
        fixed_ = true;
    }

    ///////////////////////////////////////////////////////////////////////////////////
    void BaseParametersBlock::
    SetVariable()
    {
        fixed_ = false;
    }

    ///////////////////////////////////////////////////////////////////////////////////
    void BaseParametersBlock::
    SetValid(bool is_valid)
    {
        valid_ = is_valid;
    }

    ///////////////////////////////////////////////////////////////////////////////////
    void BaseParametersBlock::
    SetJacobianId(size_t jacobian_id)
    {
        jacobian_id_ = jacobian_id;
    }
}//end of SuperVIO