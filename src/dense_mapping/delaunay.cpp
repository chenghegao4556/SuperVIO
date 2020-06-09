//
// Created by chenghe on 6/9/20.
//
#include <dense_mapping/delaunay.h>
#include <iostream>

namespace SuperVIO::DenseMapping
{
    std::vector<std::vector<cv::Point2f>> Delaunay::
    Triangulate(const std::vector<cv::Point2f>& points)
    {
        struct triangulateio in;
        struct triangulateio out;
        int32_t k;

        // inputs
        in.numberofpoints = points.size();
        in.pointlist = (float*)malloc(in.numberofpoints*2*sizeof(float)); // NOLINT
        k = 0;
        for (const auto & point : points)
        {
            in.pointlist[k++] = point.x;
            in.pointlist[k++] = point.y;
        }
        in.numberofpointattributes = 0;
        in.pointattributelist      = NULL;
        in.pointmarkerlist         = NULL;
        in.numberofsegments        = 0;
        in.numberofholes           = 0;
        in.numberofregions         = 0;
        in.regionlist              = NULL;

        // outputs
        out.pointlist              = NULL;
        out.pointattributelist     = NULL;
        out.pointmarkerlist        = NULL;
        out.trianglelist           = NULL;
        out.triangleattributelist  = NULL;
        out.neighborlist           = NULL;
        out.segmentlist            = NULL;
        out.segmentmarkerlist      = NULL;
        out.edgelist               = NULL;
        out.edgemarkerlist         = NULL;

        // do triangulation (z=zero-based, n=neighbors, Q=quiet, B=no boundary markers)
        char parameters[] = "zQB";
        triangulate(parameters, &in, &out, NULL);
        std::vector<std::vector<cv::Point2f>> triangles;
        k = 0;
        for (size_t i = 0; i < out.numberoftriangles; i++)
        {
            std::vector<cv::Point2f> tri{points[out.trianglelist[k]],
                                         points[out.trianglelist[k+1]],
                                         points[out.trianglelist[k+2]]};
            k+=3;
            triangles.push_back(tri);
        }

        free(in.pointlist);
        free(out.pointlist);
        free(out.trianglelist);
        free(out.edgelist);
        free(out.neighborlist);

        return triangles;
    }
}