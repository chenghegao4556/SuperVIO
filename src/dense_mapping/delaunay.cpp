//
// Created by chenghe on 6/9/20.
//
#include <dense_mapping/delaunay.h>
#include <iostream>

namespace SuperVIO::DenseMapping
{
    /////////////////////////////////////////////////////////////////////////////////////////
    Triangle2D::
    Triangle2D(const cv::Point2f& _a, const cv::Point2f& _b, const cv::Point2f& _c,
               double _d_a, double _d_b, double _d_c):
            a(_a),
            b(_b),
            c(_c),
            depth_a(_d_a),
            depth_b(_d_b),
            depth_c(_d_c)
    {

    }

    /////////////////////////////////////////////////////////////////////////////////////////
    std::pair<std::vector<cv::Vec3i>, std::vector<cv::Vec2i>> Delaunay::
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
        char parameters[] = "zneQB";
        triangulate(parameters, &in, &out, NULL);
        std::vector<cv::Vec3i> triangles;
        triangles.resize(out.numberoftriangles);
        k = 0;
        for (int i = 0; i < out.numberoftriangles; i++)
        {
            triangles[i] = cv::Vec3i(out.trianglelist[k],
                                     out.trianglelist[k+1],
                                     out.trianglelist[k+2]);
            k+=3;
        }

        std::vector<cv::Vec2i> edges;
        edges.resize(out.numberofedges);
        k = 0;
        for (int i = 0; i < out.numberofedges; i++)
        {
            edges[i] = cv::Vec2i(out.edgelist[k], out.edgelist[k+1]);
            k+=2;
        }

        free(in.pointlist);
        free(out.pointlist);
        free(out.trianglelist);
        free(out.edgelist);
        free(out.neighborlist);

        return std::make_pair(triangles, edges);
    }
}