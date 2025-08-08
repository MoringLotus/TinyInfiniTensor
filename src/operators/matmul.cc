#include "operators/matmul.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        // 检查输入数量
        if (inputs.size() != 2) {
            return std::nullopt;
        }

        const auto &A = inputs[0];
        const auto &B = inputs[1];
        const auto &dimsA = A->getDims();
        const auto &dimsB = B->getDims();

        // 检查维度数量至少为2
        if (dimsA.size() < 2 || dimsB.size() < 2) {
            return std::nullopt;
        }

        // 获取最后两个维度（矩阵维度）
        size_t rankA = dimsA.size();
        size_t rankB = dimsB.size();
        size_t M = dimsA[rankA - 2];
        size_t K_A = dimsA[rankA - 1];
        size_t K_B = dimsB[rankB - 2];
        size_t N = dimsB[rankB - 1];

        // 考虑转置
        if (transA) {
            std::swap(M, K_A);
        }
        if (transB) {
            std::swap(K_B, N);
        }

        // 检查K维度是否匹配
        if (K_A != K_B) {
            return std::nullopt;
        }

        // 计算输出形状
        vector<Shape> outputShapes;

        // 处理广播维度
        size_t broadcastRank = std::max(rankA, rankB);
        Shape broadcastDims(broadcastRank - 2);

        for (size_t i = 0; i < broadcastRank - 2; ++i) {
            size_t dimA = (i < rankA - 2) ? dimsA[i] : 1;
            size_t dimB = (i < rankB - 2) ? dimsB[i] : 1;
            
            if (dimA == dimB) {
                broadcastDims[i] = dimA;
            } else if (dimA == 1) {
                broadcastDims[i] = dimB;
            } else if (dimB == 1) {
                broadcastDims[i] = dimA;
            } else {
                // 不兼容的广播维度
                return std::nullopt;
            }
        }
        // 构建最终输出形状
        Shape outputShape = broadcastDims;
        outputShape.push_back(M);
        outputShape.push_back(N);

        outputShapes.push_back(outputShape);
        return outputShapes;
    }

} // namespace infini