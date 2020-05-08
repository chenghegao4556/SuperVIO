//
// Created by chenghe on 3/5/20.
//

#ifndef SUPER_VIO_MATRIX_OPERATION_H
#define SUPER_VIO_MATRIX_OPERATION_H

#include <ros/ros.h>
#include <utility/eigen_type.h>

namespace SuperVIO::Utility
{
    class EigenAssert
    {
    public:
        /**
         * @brief check input matrix is square?
         * @param[in] input_matrix
         */
        static void CheckSquareMatrix(const MatrixX& input_matrix);

        /**
         * @brief check input matrix's size
         * @param[in] input_matrix
         * @param[in] rows
         * @param[in] cols
         */
        static void CheckMatrixSize(const MatrixX& input_matrix, size_t rows, size_t cols);

        /**
         * @brief check input vector's size
         * @param[in] input_vector
         * @param[in] size
         */
        static void CheckVectorSize(const VectorX& input_vector, size_t size);

        /**
         * @brief check square matrix's size
         * @param[in] input_matrix
         * @param[in] size
         */
        static void CheckSquareMatrix(const MatrixX& input_matrix, size_t size);
    };//end of EigenAssert

    class HessianMatrixDecompose
    {
    public:
        class Result
        {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            Result(MatrixX _linearized_jacobians,
                   VectorX _linearized_residuals);

            Result() = delete;

            MatrixX linearized_jacobians;
            VectorX linearized_residuals;
        };//end of Result
        /**
         * @brief decompose heissian matrix ==> jacobians matrix
         * @param[in] information_matrix
         * @param[in] information_vector
         */
        static Result Compute(const MatrixX& information_matrix,
                              const VectorX& information_vector);
    };//end of MatrixSqrt

    class MatrixInverse
    {
    public:
        /**
         * @brief compute inverse matrix of square matrix
         * @param[in] input_matrix
         */
        static MatrixX Compute(const MatrixX& input_matrix);
    };//end of MatrixInverse

    class SchurComplement
    {
    public:
        class Result
        {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            Result(MatrixX _information_matrix,
                   VectorX _information_vector);

            Result() = delete;

            MatrixX information_matrix;
            VectorX information_vector;
        };//end of Result

        /**
         * @brief compute schur complement
         * @param[in] full_information_matrix
         * @param[in] full_information_vector
         * @param[in] dropped_size
         * @param[in] keep_size
         */
        static Result Compute(const MatrixX& full_information_matrix,
                              const VectorX& full_information_vector,
                              size_t dropped_size,
                              size_t keep_size);
    };//end of SchurComplement
}//end of SuperVIO



#endif //SUPER_VIO_MATRIX_OPERATION_H
