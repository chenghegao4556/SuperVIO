//
// Created by chenghe on 5/12/20.
//

#ifndef SUPER_VIO_DENSIFIER_H
#define SUPER_VIO_DENSIFIER_H
#include <dense_mapping/mesh_regularizer.h>
namespace SuperVIO::DenseMapping
{
    class Densifier
    {
    public:
        static void Evaluate();
    protected:
        static void EstimateNormal();
        static void DelaunayTriangulate();
        static void RegulateMesh();
        static void interpolateMesh();
    };//end of
}//end of SuperVIO::DenseMapping

#endif //SUPER_VIO_DENSIFIER_H
