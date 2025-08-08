#include "core/graph.h"
#include <algorithm>
#include <numeric>
#include <queue>
#include "operators/matmul.h"
#include "operators/transpose.h"
namespace infini
{

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
        for (const auto &tensor : tensors)
            oss << tensor << "\n";

        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
            vector<UidBaseType> preds, succs;
            for (auto &o : op->getPredecessors())
                preds.emplace_back(o->getGuid());
            for (auto &o : op->getSuccessors())
                succs.emplace_back(o->getGuid());
            oss << "OP " << op->getGuid();
            oss << ", pred " << vecToString(preds);
            oss << ", succ " << vecToString(succs);
            oss << ", " << op << "\n";
        }
        return oss.str();
    }

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        std::vector<Operator> sorted;
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        while (sorted.size() < ops.size())
        {
            // Any node is move to sorted in this loop.
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified)
            {
                return false;
            }
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }

    void GraphObj::optimize()
    {
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来实现指定的图优化规则
        // 图优化规则如下：
        // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）
        // 2. 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
        // =================================== 作业 ===================================

        // rule1: 删除无用的transpose算子
        for (size_t i = 0; i < ops.size(); ++i)
        {
            Operator op = ops[i];
            if (op->getOpType() == OpType::Transpose)
            {
                Tensor tensor = op->getOutput();
                if (!tensor)
                    continue;
                auto targets = tensor->getTargets();
                if (targets.empty())
                    continue;
                Operator op_next = targets[0];
                if (op_next->getOpType() == OpType::Transpose)
                {
                    TransposeObj *op1 = as<TransposeObj>(op).get();
                    TransposeObj *op2 = as<TransposeObj>(op_next).get();
                    auto op1_permute = op1->getPermute();
                    auto op2_permute = op2->getPermute();
                    if (op1_permute.size() != op2_permute.size())
                        continue;
                    bool flag = true;
                    for (int j = 0; j < (int)op1_permute.size(); j++)
                    {
                        if (op1_permute[op2_permute[j]] != j)
                        {
                            flag = false;
                            continue;
                        }
                    }
                    if (!flag) //flag为false说明 无法合并
                        continue;
                    // 获取第一个转置算子的输入张量（原始输入数据）
                    Tensor originalInput = op->getInputs()[0];  

                    // 获取第一个转置算子的输出张量（第一次转置结果）
                    Tensor firstTransposeOutput = op->getOutput();  

                    // 获取第二个转置算子的输出张量（最终转置结果） 
                    Tensor secondTransposeOutput = op_next->getOutput();

                    // 获取使用最终结果的消费者算子（如矩阵乘法）
                    Operator consumerOp = secondTransposeOutput->getTargets()[0];  

                    // 保留消费者算子的其他输入（如矩阵乘法的右矩阵）
                    Tensor consumerOtherInput = consumerOp->getInputs()[1];  

                    // 重定向消费者算子的输入：跳过两个转置，直接使用原始输入
                    consumerOp->replaceInput(consumerOp->getInputs()[0], originalInput);

                    // 更新原始输入的连接关系：
                    originalInput->removeTarget(op);          // 移除对第一个转置的引用
                    originalInput->addTarget(consumerOp);     // 添加对消费者算子的引用
                    originalInput->setSource(nullptr);        // 清除可能存在的生产者标记

                    // 清理冗余资源
                    removeOperator(op);                      // 删除第一个转置算子
                    removeOperator(op_next);                 // 删除第二个转置算子
                    removeTensor(firstTransposeOutput);       // 删除中间结果张量
                    removeTensor(secondTransposeOutput);     // 删除最终结果张量

                    // 更新算子间的拓扑依赖关系
                    consumerOp->removePredecessors(op_next); // 移除与第二个转置的依赖

                    // 如果原始输入有生产者，建立新的依赖关系
                    if (originalInput->getSource()) {
                        consumerOp->addPredecessors(originalInput->getSource());
                        originalInput->getSource()->addSuccessors(consumerOp);
                    }
                }
            }
        }
        
        // 遍历图中的所有算子，寻找可优化的矩阵乘法算子
        for (size_t opIndex = 0; opIndex < ops.size(); ++opIndex) {
            Operator currentOp = ops[opIndex];
            
            // 只处理矩阵乘法算子
            if (currentOp->getOpType() == OpType::MatMul) {
                // 获取矩阵乘法的输入张量列表（左矩阵和右矩阵）
                TensorVec matmulInputs = currentOp->getInputs();
                int inputIndex = 0;  // 用于标识当前是左输入(0)还是右输入(1)
                
                // 检查每个输入张量
                for (Tensor inputTensor : matmulInputs) {
                    inputIndex++;
                    
                    // 检查输入张量是否有生产者算子
                    if (inputTensor->getSource()) {
                        Operator producerOp = inputTensor->getSource();
                        
                        // 如果生产者是转置算子
                        if (producerOp->getOpType() == OpType::Transpose) {
                            TransposeObj *transposeOp = as<TransposeObj>(producerOp).get();
                            Shape transposePerm = transposeOp->getPermute();
                            bool isLastTwoDimsSwap = true;
                            
                            /* 验证转置操作是否只交换最后两个维度：
                            * 1. 前n-2个维度必须保持原顺序（即perm[j] == j）
                            * 2. 最后两个维度必须交换（即perm[-2] == rank-1 且 perm[-1] == rank-2）
                            */
                            for (int dim = 0; dim < (int)transposePerm.size() - 2; dim++) {
                                if (transposePerm[dim] != dim) {
                                    isLastTwoDimsSwap = false;
                                    break;
                                }
                            }
                            if (transposePerm[transposePerm.size() - 2] != (int)transposePerm.size() - 1 || 
                                transposePerm[transposePerm.size() - 1] != (int)transposePerm.size() - 2) {
                                isLastTwoDimsSwap = false;
                            }
                            
                            // 如果不满足条件则跳过优化
                            if (!isLastTwoDimsSwap) continue;
                            
                            // 获取矩阵乘法算子（用于修改转置属性）
                            MatmulObj *matmulOp = as<MatmulObj>(currentOp).get();
                            Tensor transposedTensor;
                            
                            // 根据输入位置设置对应的转置标志
                            if (inputIndex == 1) {  // 左输入
                                matmulOp->setTransA(true);  // 启用左矩阵转置
                                transposedTensor = matmulOp->getInputs(0);
                            } else {  // 右输入
                                matmulOp->setTransB(true);  // 启用右矩阵转置
                                transposedTensor = matmulOp->getInputs(1);
                            }
                            
                            // 获取转置算子的输入（原始未转置的张量）
                            Operator transposeOperator = transposedTensor->getSource();
                            Tensor originalTensor = transposeOperator->getInputs()[0];
                            
                            // 重定向矩阵乘法的输入：跳过转置算子，直接使用原始张量
                            matmulOp->replaceInput(transposedTensor, originalTensor);
                            
                            // 更新张量连接关系
                            originalTensor->removeTarget(transposeOperator);
                            originalTensor->addTarget(currentOp);
                            
                            // 清理资源：删除转置算子和中间张量
                            removeOperator(transposeOperator);
                            removeTensor(transposedTensor);
                            
                            // 更新拓扑关系：移除转置算子作为前驱
                            currentOp->removePredecessors(transposeOperator);
                            
                            // 如果原始张量有生产者，建立新的依赖关系
                            if (originalTensor->getSource()) {
                                currentOp->addPredecessors(originalTensor->getSource());
                                originalTensor->getSource()->addSuccessors(currentOp);
                            }
                        }
                    }
                }
            }
}
    }

    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc()
    {
        // topological sorting first
        IT_ASSERT(topo_sort() == true);

        // =================================== 作业 ===================================
        // TODO：利用 allocator 给计算图分配内存
        // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor 绑定内存
        // =================================== 作业 ===================================
        // allocator.info();
        // void* allocatorPtr = allocator.getPtr();
        // for(auto it = tensors.begin(); it != tensors.end(); it++){
        //     auto tensor = *it;
        //     size_t size = tensor->getBytes();
        //     size_t addr = allocator.alloc(size);
        //     char * tmpPtr = reinterpret_cast<char*>(allocatorPtr) + addr;
        //     Blob blob = make_ref<BlobObj>(runtime, (void *)tmpPtr);
        //     tensor->setDataBlob(blob);
        // }
        // topological sorting first
        IT_ASSERT(topo_sort() == true);

        // =================================== 作业 ===================================
        // TODO：利用 allocator 给计算图分配内存
        // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor 绑定内存
        // =================================== 作业 ===================================
        vector<size_t> offsets;
        for (auto tensor : tensors)
        {
            size_t size = tensor->getBytes();
            size_t offset = allocator.alloc(size);
            offsets.push_back(offset);
        }
        auto it = offsets.begin();
        void *basePtr = allocator.getPtr();
        for (auto tensor : tensors)
        {
            char *charPtr = reinterpret_cast<char *>(basePtr) + *it;
            void *ptr = charPtr;
            Blob blob = make_ref<BlobObj>(runtime, ptr);
            tensor->setDataBlob(blob);
            it++;
        }
        allocator.info();
    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime,
                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                      tensor->getRuntime()->toString() + " to " +
                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    // tensor's "source" and "target" must be in "ops".
    // tensor has no "source" and no "target" must not exist.
    // "inputs" or "outputs" of operators must be in "tensors"
    // "predecessors" and "successors" of an operator of "ops" must be in "ops".
    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
            }
        }
        std::set<UidBaseType> s;
        // check whether two tensors with the same FUID exist
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

} // namespace infini