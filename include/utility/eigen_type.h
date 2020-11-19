//
// Created by chenghe on 3/5/20.
//

#ifndef SRC_EIGEN_TYPE_H
#define SRC_EIGEN_TYPE_H

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/StdVector>
#include <map>
namespace SuperVIO
{
    typedef Eigen::Vector2d Vector2;
    typedef Eigen::Vector3d Vector3;
    typedef Eigen::Vector4d Vector4;
    typedef Eigen::Matrix<double, 6, 1> Vector6;
    typedef Eigen::Matrix<double, 9, 9> Matrix9;
    typedef Eigen::Matrix<double, 9, 1>  Vector9;
    typedef Eigen::Matrix<double, 15, 1> Vector15;
    typedef Eigen::Matrix2d Matrix2;
    typedef Eigen::Matrix3d Matrix3;
    typedef Eigen::Isometry3d Pose3;
    typedef Eigen::Quaterniond Quaternion;

    typedef Eigen::Matrix<double, 6, 6> Matrix6;
    typedef Eigen::Matrix<double, 15, 15> Matrix15;
    typedef Eigen::Matrix<double, 18, 18> Matrix18;

    typedef Eigen::Matrix<double, 2, 3> Matrix23;
    typedef Eigen::Matrix<double, 2, 7, Eigen::RowMajor> Matrix27;

    typedef Eigen::Matrix<double, 3, 6> Matrix36;

    typedef Eigen::Matrix<double, 15, 7, Eigen::RowMajor> Matrix15_7;
    typedef Eigen::Matrix<double, 15, 9, Eigen::RowMajor> Matrix15_9;
    typedef Eigen::Matrix<double, 7, 6, Eigen::RowMajor> Matrix76;

    typedef Eigen::VectorXd VectorX;
    typedef Eigen::MatrixXd MatrixX;

    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
            DynamicMatrix;

    typedef Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> MapMatrix6;

    typedef Eigen::Map<DynamicMatrix> MapDynamicMatrix;



    typedef std::map<double, Vector3, std::less<>,
            Eigen::aligned_allocator<std::pair<const double, Vector3>>> DoubleVector3Map;

    typedef std::map<double, Quaternion, std::less<>,
            Eigen::aligned_allocator<std::pair<const double, Quaternion>>> DoubleQuaternionMap;

    typedef Eigen::Map<VectorX> MapVectorX;
    typedef Eigen::Map<MatrixX> MapMatrixX;

    typedef Eigen::Map<const VectorX> ConstMapVectorX;
    typedef Eigen::Map<const MatrixX> ConstMapMatrixX;

    typedef Eigen::Map<Vector3> MapVector3;
    typedef Eigen::Map<Quaternion> MapQuaternion;
    typedef Eigen::Map<const Vector3> ConstMapVector3;
    typedef Eigen::Map<const Quaternion> ConstMapQuaternion;

    typedef Eigen::Map<Vector2> MapVector2;
    typedef Eigen::Map<Vector6> MapVector6;
    typedef Eigen::Map<Vector9> MapVector9;

    typedef Eigen::Map<Vector15> MapVector15;

    typedef Eigen::Map<Matrix27> MapMatrix27;

    typedef Eigen::Map<Matrix76> MapMatrix76;

    typedef Eigen::Map<Matrix15_7> MapMatrix15_7;
    typedef Eigen::Map<Matrix15_9> MapMatrix15_9;

    typedef std::vector<Vector3, Eigen::aligned_allocator<Vector3>> Vector3s;
    typedef std::vector<Vector2, Eigen::aligned_allocator<Vector2>> Vector2s;
    typedef std::vector<Quaternion, Eigen::aligned_allocator<Quaternion>> Quaternions;
    typedef std::vector<Pose3, Eigen::aligned_allocator<Pose3>> Pose3s;
}//end of SuperVio

#endif //SRC_EIGEN_TYPE_H
