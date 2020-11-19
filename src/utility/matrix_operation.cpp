//
// Created by chenghe on 3/5/20.
//

#include <utility/matrix_operation.h>

namespace SuperVIO::Utility
{
    ////////////////////////////////////////////////////////////////////////////////////////
    void EigenAssert::
    CheckSquareMatrix(const MatrixX& input_matrix)
    {
        if(input_matrix.cols() != input_matrix.rows())
        {
            ROS_ERROR_STREAM("input matrix is not a Square matrix!");
            throw std::runtime_error("Matrix Size Error");
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////
    void EigenAssert::
    CheckSquareMatrix(const MatrixX& input_matrix, size_t size)
    {
        CheckSquareMatrix(input_matrix);
        if(input_matrix.rows() != (int)size)
        {
            ROS_ERROR_STREAM("Square matrix rows is not "<<size<<" * "<<size<<"!");
            throw std::runtime_error("Matrix Size Error");
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////
    void EigenAssert::
    CheckVectorSize(const VectorX& input_vector,
                    size_t size)
    {
        if(input_vector.size() != (int)size)
        {
            ROS_ERROR_STREAM("Vector matrix's size is not"<<size<<"!");
            throw std::runtime_error("Vector Size Error");
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////
    void EigenAssert::
    CheckMatrixSize(const MatrixX& input_matrix,
                    size_t rows, size_t cols)
    {
        if(input_matrix.rows() != (int)rows)
        {
            ROS_ERROR_STREAM("input matrix rows is not "<<rows);
            throw std::runtime_error("Matrix Size Error");
        }
        if(input_matrix.cols() != (int)cols)
        {
            ROS_ERROR_STREAM("input matrix cols is not "<<cols);
            throw std::runtime_error("Matrix Size Error");
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////
    HessianMatrixDecompose::
    Result::Result(MatrixX _linearized_jacobians,
                   VectorX _linearized_residuals):
        linearized_jacobians(std::move(_linearized_jacobians)),
        linearized_residuals(std::move(_linearized_residuals))
    {

    }

    ////////////////////////////////////////////////////////////////////////////////////////
    HessianMatrixDecompose::Result HessianMatrixDecompose::
    Compute(const MatrixX& information_matrix,
            const VectorX& information_vector)
    {
        EigenAssert::CheckSquareMatrix(information_matrix);
        EigenAssert::CheckVectorSize(information_vector, information_matrix.rows());

        const double eps = std::numeric_limits<double>::min();

        Eigen::SelfAdjointEigenSolver<MatrixX> solver(information_matrix);

        VectorX S = VectorX((solver.eigenvalues().array() > eps).
                select(solver.eigenvalues().array(), 0));
        VectorX S_inv = VectorX((solver.eigenvalues().array() > eps).
                select(solver.eigenvalues().array().inverse(), 0));

        VectorX sqrt_vector = S.cwiseSqrt();
        VectorX inverse_sqrt_vector = S_inv.cwiseSqrt();

        MatrixX linearized_jacobians = sqrt_vector.asDiagonal() *
                solver.eigenvectors().transpose();
        VectorX linearized_residuals = inverse_sqrt_vector.asDiagonal() *
                solver.eigenvectors().transpose() * information_vector;

        return Result(linearized_jacobians, linearized_residuals);
    }

    ////////////////////////////////////////////////////////////////////////////////////////
    MatrixX MatrixInverse::
    Compute(const MatrixX& input_matrix)
    {
        EigenAssert::CheckSquareMatrix(input_matrix);
        const double eps = std::numeric_limits<double>::min();

        Eigen::SelfAdjointEigenSolver<MatrixX> solver(input_matrix);
        MatrixX inverse_matrix = solver.eigenvectors() * VectorX((solver.eigenvalues().array() > eps).
                select(solver.eigenvalues().array().inverse(), 0)).asDiagonal() * solver.eigenvectors().transpose();

        return inverse_matrix;
    }

    ////////////////////////////////////////////////////////////////////////////////////////
    SchurComplement::Result::
    Result(MatrixX _information_matrix,
           VectorX _information_vector):
           information_matrix(std::move(_information_matrix)),
           information_vector(std::move(_information_vector))
    {

    }

    ////////////////////////////////////////////////////////////////////////////////////////
    SchurComplement::Result SchurComplement::
    Compute(const MatrixX& full_information_matrix,
            const VectorX& full_information_vector,
            size_t dropped_size,
            size_t keep_size)
    {

        EigenAssert::CheckSquareMatrix(full_information_matrix, dropped_size + keep_size);
        EigenAssert::CheckVectorSize(full_information_vector, dropped_size + keep_size);

        MatrixX Amm = 0.5 * (full_information_matrix.block(0, 0, dropped_size, dropped_size) +
                full_information_matrix.block(0, 0, dropped_size, dropped_size).transpose());
        MatrixX Amm_inv = MatrixInverse::Compute(Amm);

        VectorX bmm = full_information_vector.segment(0, dropped_size);

        MatrixX Amr = full_information_matrix.block(0, dropped_size, dropped_size, keep_size);
        MatrixX Arm = full_information_matrix.block(dropped_size, 0, keep_size, dropped_size);

        MatrixX Arr = full_information_matrix.block(dropped_size, dropped_size, keep_size, keep_size);
        VectorX brr = full_information_vector.segment(dropped_size, keep_size);

        MatrixX marginalized_information_matrix = Arr - Arm * Amm_inv * Amr;
        VectorX marginalized_information_vector = brr - Arm * Amm_inv * bmm;

        return Result(marginalized_information_matrix,
                      marginalized_information_vector);

    }
}//end of Utility