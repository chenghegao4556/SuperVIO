//
// Created by chenghe on 6/10/20.
//
#include <dense_mapping/mesh_regularizer.h>
namespace SuperVIO::DenseMapping
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    std::vector<Triangle2D> MeshRegularizer::
    Evaluate(const std::vector<cv::Point2f>& points, const std::vector<double>& depths,
             const std::vector<cv::Vec3i>& triangle_indices, const std::vector<cv::Vec2i>& edge_indices)
    {
        double sum = 0.0;
        for(const auto& depth: depths)
        {
            sum += (1.0 / depth);
        }
        double scale = sum/ static_cast<double>(depths.size());
        scale = 1;
        auto graph = CreatGraph(points, depths, edge_indices, scale);
        Parameters params;
        std::cout<<"before "<<SmoothnessCost(params, graph);
        graph = Step(params, graph);
        std::cout<<"after "<<SmoothnessCost(params, graph);
        auto mesh = GraphToMesh(points, depths, graph, triangle_indices, scale);

        return mesh;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    Graph MeshRegularizer::
    CreatGraph(const std::vector<cv::Point2f>& points, const std::vector<double>& depths,
               const std::vector<cv::Vec2i>& edge_indices, const double scale)
    {
        Graph graph;
        std::unordered_map<int, VertexHandle> vertex_map;
        for (size_t i = 0; i < points.size(); ++i)
        {
            VertexHandle vertex_key = boost::add_vertex(VertexData(), graph);
            vertex_map.insert(std::make_pair(i, vertex_key));
            auto& vertex = graph[vertex_key];
            vertex.id = i;
            vertex.pos = points[i];
            double inverse_depth = 1.0 / depths[i];
            vertex.data_term = static_cast<float>(inverse_depth / scale);

            vertex.x = vertex.data_term;
            vertex.x_bar = vertex.data_term;
            vertex.x_prev = vertex.data_term;
        }

        for (const auto & edge_index : edge_indices)
        {
            const auto& vertex_key_i = vertex_map[edge_index[0]];
            const auto& vertex_key_j = vertex_map[edge_index[1]];

            // Compute edge length.
            const auto& point_i = points[edge_index[0]];
            const auto& point_j = points[edge_index[1]];
            cv::Point2f diff(point_i - point_j);
            float edge_length = std::sqrt(diff.x*diff.x + diff.y*diff.y);

            if (!boost::edge(vertex_key_i, vertex_key_j, graph).second)
            {
                boost::add_edge(vertex_key_i, vertex_key_j, EdgeData(), graph);
            }
            const auto& edge_key = boost::edge(vertex_key_i, vertex_key_j, graph);
            auto& edge = graph[edge_key.first];
            edge.alpha = 1.0f / edge_length;
            edge.beta = 1.0f;
            edge.valid = true;
        }

        return graph;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    std::vector<Triangle2D> MeshRegularizer::
    GraphToMesh(const std::vector<cv::Point2f>& points, const std::vector<double>& depths,
                const Graph& graph, const std::vector<cv::Vec3i>& triangle_indices, const double scale)
    {
        auto new_depths = depths;
        Graph::vertex_iterator vertex_iter, end;
        boost::tie(vertex_iter, end) = boost::vertices(graph);
        for ( ; vertex_iter != end; ++vertex_iter)
        {
            const auto& vertex = graph[*vertex_iter];
            double inverse_depth = vertex.x * scale;
            new_depths[vertex.id] = 1.0 / inverse_depth;
        }

        std::vector<Triangle2D> mesh;
        for(const auto& tri: triangle_indices)
        {
            auto d_a = new_depths[tri[2]];
            auto d_b = new_depths[tri[1]];
            auto d_c = new_depths[tri[0]];
            mesh.emplace_back(points[tri[2]], points[tri[1]], points[tri[0]], d_a, d_b, d_c);
        }

        return mesh;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    Graph MeshRegularizer::
    Step(const Parameters& params, const Graph& graph)
    {
        Graph new_graph = graph;
        for(size_t i = 0; i < 10; ++i)
        {
            Graph::vertex_iterator vertex_iter, end;
            boost::tie(vertex_iter, end) = boost::vertices(new_graph);
            for ( ; vertex_iter != end; ++vertex_iter)
            {
                VertexData& vertex = (new_graph)[*vertex_iter];
                vertex.x_prev  = vertex.x;
                vertex.w1_prev = vertex.w1;
                vertex.w2_prev = vertex.w2;
            }
            new_graph = DualStep(params, new_graph);
            new_graph = PrimalStep(params, new_graph);
            new_graph = ExtraGradientStep(params, new_graph);
        }

        return new_graph;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    Graph MeshRegularizer::
    DualStep(const Parameters& params, const Graph& graph)
    {
        Graph new_graph = graph;
        Graph::edge_iterator eit, end;
        boost::tie(eit, end) = boost::edges(new_graph);
        for ( ; eit != end; ++eit)
        {
            EdgeData& edge = (new_graph)[*eit];
            VertexData& vtx_ii = (new_graph)[boost::source(*eit, new_graph)];
            VertexData& vtx_jj = (new_graph)[boost::target(*eit, new_graph)];

            // Update q1.
            float K1x = edge.alpha * (vtx_ii.x_bar - vtx_jj.x_bar);
            K1x -= edge.alpha * (vtx_ii.pos.x - vtx_jj.pos.x) * vtx_ii.w1_bar;
            K1x -= edge.alpha * (vtx_ii.pos.y - vtx_jj.pos.y) * vtx_ii.w2_bar;
            edge.q1 = ProximalNLTGV2Conj(params.step_q, edge.q1 + params.step_q * K1x);

            // Update q2.
            float K2x = edge.beta * (vtx_ii.w1_bar - vtx_jj.w1_bar);
            edge.q2 = ProximalNLTGV2Conj(params.step_q, edge.q2 + params.step_q * K2x);

            // Update q3.
            float K3x = edge.beta * (vtx_ii.w2_bar - vtx_jj.w2_bar);
            edge.q3 = ProximalNLTGV2Conj(params.step_q, edge.q3 + params.step_q * K3x);
        }

        return new_graph;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    Graph MeshRegularizer::
    PrimalStep(const Parameters& params, const Graph& graph)
    {
        Graph new_graph = graph;
        Graph::edge_iterator eit, end;
        boost::tie(eit, end) = boost::edges(new_graph);
        for ( ; eit != end; ++eit)
        {
            EdgeData& edge = (new_graph)[*eit];
            VertexData& vtx_ii = (new_graph)[boost::source(*eit, new_graph)];
            VertexData& vtx_jj = (new_graph)[boost::target(*eit, new_graph)];

            // Apply updates that require q1.
            vtx_ii.x -= edge.q1 * params.step_x * edge.alpha;
            vtx_jj.x += edge.q1 * params.step_x * edge.alpha;

            vtx_ii.w1 += edge.q1 * params.step_x * edge.alpha *
                         (vtx_ii.pos.x - vtx_jj.pos.x);

            vtx_ii.w2 += edge.q1 * params.step_x * edge.alpha *
                         (vtx_ii.pos.y - vtx_jj.pos.y);

            // Apply updates that require q2.
            vtx_ii.w1 -= edge.q2 * params.step_x * edge.beta;
            vtx_jj.w1 += edge.q2 * params.step_x * edge.beta;

            // Apply updates that require q3.
            vtx_ii.w2 -= edge.q3 * params.step_x * edge.beta;
            vtx_jj.w2 += edge.q3 * params.step_x * edge.beta;
        }

        // Apply proximal operator to each vertex.
        Graph::vertex_iterator vit, vend;
        boost::tie(vit, vend) = boost::vertices(new_graph);
        for ( ; vit != vend; ++vit)
        {
            VertexData& vtx = (new_graph)[*vit];
            vtx.x = ProximalL1(params.x_min, params.x_max, params.step_x,
                               params.data_factor * vtx.data_weight, vtx.x, vtx.data_term);
        }

        return new_graph;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    Graph MeshRegularizer::
    ExtraGradientStep(const Parameters& params, const Graph& graph)
    {
        Graph new_graph = graph;

        Graph::vertex_iterator vit, end;
        boost::tie(vit, end) = boost::vertices(new_graph);
        for ( ; vit != end; ++vit)
        {
            VertexData& vtx = (new_graph)[*vit];
            float new_x_bar = vtx.x + params.theta * (vtx.x - vtx.x_prev);

            // Project back onto the feasible set.
            new_x_bar = (new_x_bar < params.x_min) ? params.x_min : new_x_bar;
            new_x_bar = (new_x_bar > params.x_max) ? params.x_max : new_x_bar;
            vtx.x_bar = new_x_bar;

            vtx.w1_bar = vtx.w1 + params.theta * (vtx.w1 - vtx.w1_prev);
            vtx.w2_bar = vtx.w2 + params.theta * (vtx.w2 - vtx.w2_prev);
        }

        return new_graph;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    float MeshRegularizer::
    ProximalNLTGV2Conj(float step, float q)
    {
        float absq = (q > 0) ? q : -q;
        float new_q = q / (absq > 1 ? absq : 1);
        return new_q;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    float MeshRegularizer::
    ProximalL1(float x_min, float x_max, float step_x, float data_weight,
               float x, float data)
    {
        float diff = x - data;
        float thresh = step_x * data_weight;

        float new_x = 0.0f;
        if (diff > thresh)
        {
            new_x = x - thresh;
        }
        else if (diff < -thresh)
        {
            new_x = x + thresh;
        }
        else
        {
            new_x = data;
        }

        // Project back onto the feasible set.
        new_x = (new_x < x_min) ? x_min : new_x;
        new_x = (new_x > x_max) ? x_max : new_x;
        return new_x;
    }

    float MeshRegularizer::
    SmoothnessCost(const Parameters& params, const Graph& graph)
    {
        float cost = 0.0f;

        auto fast_abs = [](float a) { return (a >= 0) ? a : -a; };

        Graph::edge_iterator eit, end;
        boost::tie(eit, end) = boost::edges(graph);
        for ( ; eit != end; ++eit) {
            const EdgeData& edge = graph[*eit];
            const VertexData& vtx_ii = graph[boost::source(*eit, graph)];
            const VertexData& vtx_jj = graph[boost::target(*eit, graph)];

            cv::Point2f xy_diff = vtx_ii.pos - vtx_jj.pos;
            cost += edge.alpha * fast_abs(vtx_ii.x - vtx_jj.x -
                                          vtx_ii.w1 * xy_diff.x - vtx_ii.w2 * xy_diff.y);
            cost += edge.beta * fast_abs(vtx_ii.w1 - vtx_jj.w1) +
                    edge.beta * fast_abs(vtx_ii.w2 - vtx_jj.w2);
        }

        return cost;
    }

    float MeshRegularizer::
    DataCost(const Parameters& params, const Graph& graph)
    {
        float cost = 0.0f;
        Graph::vertex_iterator vit, end;
        boost::tie(vit, end) = boost::vertices(graph);
        for ( ; vit != end; ++vit) {
            const VertexData& vtx = graph[*vit];
            float diff = (vtx.x - vtx.data_term) * vtx.data_weight;
            diff = (diff > 0) ? diff : -diff;
            cost += diff;
        }

        return cost;
    }

    float MeshRegularizer::
    Cost(const Parameters& params, const Graph& graph)
    {
        return SmoothnessCost(params, graph) + DataCost(params, graph);
    }
}