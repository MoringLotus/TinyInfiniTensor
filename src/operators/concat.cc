#include "operators/concat.h"
#include "utils/operator_utils.h"

namespace infini {
ConcatObj::ConcatObj(GraphObj *graph, TensorVec inputs, Tensor output, int _dim)
    : OperatorObj(OpType::Concat, inputs, {output}) {
    int rank = inputs[0]->getRank();
    dim = get_real_axis(_dim, rank);
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> ConcatObj::inferShape(const TensorVec &inputs) {
    Shape dims = inputs[0]->getDims(); // 数组的 shape
    auto rank = inputs[0]->getRank();
    // =================================== 作业 ===================================
    // TODO：修改 dims，返回正确的 concat 后的 shape
    // REF: https://onnx.ai/onnx/operators/onnx__Concat.html#concat-13
    // =================================== 作业 ===================================
    if(inputs.size() == 0) {
        return std::nullopt;
    }
    for(auto input: inputs){
        if(input->getDims().size() != rank)
            return std::nullopt;
    }
    vector<int> res(rank, 0);
    for(auto input: inputs){
        for(size_t i = 0; i < rank; i++){
            if(i == size_t(dim)){
                res[i] += input->getDims()[i];
            }else if (i != size_t(dim)){
                res[i] = input->getDims()[i];
            }
        }
    }
    return {{res}};
}

std::string ConcatObj::toString() const {
    std::ostringstream os;
    os << "Concat[" << getGuid() << "]";
    os << "(";
    for (auto input : inputs)
        os << vecToString(input->getDims()) << ",";
    os << "dim=" << dim << ",";
    os << "input=";
    for (auto input : inputs)
        os << input->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

} // namespace infini
