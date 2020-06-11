//
// Created by chenghe on 6/10/20.
//

#ifndef SRC_MESH_REGULARIZER_H
#define SRC_MESH_REGULARIZER_H

#include <vector>
#include <memory>
#include <ros/ros.h>
#include <ros/console.h>
#include <boost/graph/adjacency_list.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <dense_mapping/delaunay.h>
namespace SuperVIO::DenseMapping
{
    struct VertexData
    {
        int id;
        cv::Point2f pos; // Position.
        float x = 0.0f; // Main primal variable.
        float w1 = 0.0f; // Plane parameters.
        float w2 = 0.0f;

        float x_bar = 0.0f; // Extragradient varaibles.
        float w1_bar = 0.0f;
        float w2_bar = 0.0f;

        float x_prev = 0.0f; // Previous x value.
        float w1_prev = 0.0f;
        float w2_prev = 0.0f;

        float data_term = 0.0f; // Data term.
        float data_weight = 1.0f; // Weight on data term.
    };

    struct EdgeData
    {
        float alpha = 1.0f; // Edge weights.
        float beta = 1.0f;
        float q1 = 0.0f; // Dual variables.
        float q2 = 0.0f;
        float q3 = 0.0f;
        bool valid = true; // False if this edge should be removed.
    };
    using Graph =
    boost::adjacency_list<boost::hash_setS, // Edges will be stored in a hash map.
            boost::hash_setS, // Vertices will be stored in a hash map
            boost::undirectedS, // Undirected graph
            VertexData, // Data stored at each vertex
            EdgeData>; // Data stored at each edge

// These descriptors are essentially handles to the vertices and edges.
    using VertexHandle = boost::graph_traits<Graph>::vertex_descriptor;
    using EdgeHandle = boost::graph_traits<Graph>::edge_descriptor;

    class MeshRegularizer
    {
    public:
        struct Parameters
        {
            float data_factor = 0.05f; // lambda in the TV literature.
            float step_x = 0.001f; // Primal step size.
            float step_q = 125.0f; // Dual step size.
            float theta = 0.25f; // Extra gradient step size.

            float x_min = 0.0f; // Feasible set.
            float x_max = 10.0f;
        };
        static std::vector<Triangle2D>
        Evaluate(const std::vector<cv::Point2f>& points, const std::vector<double>& depths,
                 const std::vector<cv::Vec3i>& triangle_indices, const std::vector<cv::Vec2i>& edge_indices);

        static float SmoothnessCost(const Parameters& params, const Graph& graph);

        static float DataCost(const Parameters& params, const Graph& graph);

        static float Cost(const Parameters& params, const Graph& graph);

    protected:

        static Graph
        CreatGraph(const std::vector<cv::Point2f>& points, const std::vector<double>& depths,
                   const std::vector<cv::Vec2i>& edge_indices, const double scale);

        static std::vector<Triangle2D>
        GraphToMesh(const std::vector<cv::Point2f>& points, const std::vector<double>& depths,
                    const Graph& graph, const std::vector<cv::Vec3i>& triangle_indices, const double scale);

        static Graph
        Step(const Parameters& params, const Graph& graph);

        static Graph DualStep(const Parameters& params, const Graph& graph);

        static Graph PrimalStep(const Parameters& params, const Graph& graph);

        static Graph ExtraGradientStep(const Parameters& params, const Graph& graph);

        static float ProximalNLTGV2Conj(float step, float q);

        static float ProximalL1(float x_min, float x_max, float step_x, float data_weight,
                                float x, float data);
    };
}

#endif //SRC_MESH_REGULARIZER_H
