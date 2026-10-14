# EmoAgent 提示词来源对照

论文没有公开三个 Agent 的完整 system prompt。本实现将“论文页面能直接看到的文字”和“为了稳定执行而补全的 JSON 约束”分开处理。

## 论文直接展示

| 位置 | 论文可见内容 | 代码位置 |
|---|---|---|
| 图 2，Emotion-Grounded Analysis | `Analyze the semantic of this image...`，分析范围包含色调、背景、物体、主体属性和表情 | `prompts.py::PAPER_SOURCE_ANALYSIS` |
| 图 18、图 19，`input_query` | `What are the semantics and emotion of the original image?`；论文正文说明 VLM 提取 `S_o` 并预测 `E_o` | 源图视觉分析提示；无标签推理集由 MLLM 输出 `E_o` |
| 图 18、图 19，`RAG_query` | 工作流插图展示了“如何修改源场景以表达目标情绪”的自然语言查询 | 仅作为论文可见提示样例，不直接作为本实现的嵌入查询模板 |
| 补充材料，EFK Database | 用情绪因素节点内容嵌入 `E_n` 与源图语义文本嵌入 `E_s` 计算 L2 距离；emotion 与 element type 是节点标签 | 中性源图语义查询 + 元数据过滤 + 全局 diverse top-k |
| 图 18、图 19，`Planning_evaluate` | 输出 `Reasonable` 后再逐步执行编辑 | `prompts.py::planning_evaluation_prompt` |
| 图 2，Critic CoT | 先检查整体目标情绪，再检查无效指令，最后检查执行错误 | `prompts.py::critic_prompt` |
| 图 18、图 19，`edit_reactify` | 输入编辑前图、编辑后图和当前指令，输出 `Correct` 或 `Wrong` | `prompts.py::step_validation_prompt` |

## 论文流程约束

- 源图和目标情绪进入 Planning Agent；
- 从 EFK 中取 diverse top-k 语义相关且目标情绪一致的候选池；
- 决策树顺序为目标情绪、编辑元素、编辑方法；
- 每套方案是有序指令序列；
- 每条指令执行 Pre-processing、Editing、Validation；
- Validation 失败时调整指令并重新执行当前动作；
- 初步结果由 Critic 独立评估；
- Critic 区分 Emotional Effectiveness 和 Execution Accuracy；
- 不合格结果进入独立 Editing/Critic 优化循环。

## 两组不能混同的概念

### 源情绪 `E_o`

论文原始方法由 VLM 从源图预测 `E_o`。当前推理数据集无源情绪标注，因此本实现同样由 Planning Agent 根据可见证据预测 `E_o`，并将 provenance 记录为 `planning_agent_vlm_prediction`。目标情绪不会提供给源图分析步骤。

用于检索的 `semantic_summary`/`retrieval_semantics` 必须保持中性，只包含可见主体、场景、构图、对象、动作、照明和颜色。它不得被已知源情绪或稍后的目标情绪改写成诸如 “excited spectators” 的情感解释。

### retrieval top-k 与方案数 `K`

retrieval top-k 是 EFK 候选池大小；`K` 是 Planning Agent 输出的独立编辑路径数量。实验中二者可以都取 5，但这不表示每个步骤各有五个选择。四步方案也不会因此产生或执行 `5^4` 个全组合。论文描述的是从候选池构建 `K` 个不同的有序指令集合，并分别生成 `K` 张图。

## 本实现的检索与规划补全

以下属于可复现性和稳健性补全，而不是论文公开的逐字算法：

1. 只将中性的源图语义文本编码为一个查询向量；
2. 用目标情绪和元素类别作为 EFK 元数据过滤条件；
3. 先扩大候选召回，再从同一距离空间中按相关性、场景兼容性和候选间冗余进行全局 diverse top-k 重排；
4. 对全部 `K` 个方案联合检查，比较完整的规范化指令签名，而非只比较 `strategy_summary`；
5. 对初始规划、确定性约束修复和 `Planning_evaluate` 修订后的方案均重新执行联合多样性检查；
6. 将仅改变步骤顺序、摘要措辞或随机种子的方案判为重复路径。

规范化签名关注实际操作、目标实体或概念、目标区域和关键属性，不把 `emotion_rationale`、`preserve` 的措辞差异误认为语义多样性。

## 本实现的 Editing/Validation 补全

论文插图中的编辑 prompt 和 `Correct/Wrong` 只展示了简化文本。本实现为统一的 FLUX 编辑器补充如下结构：

- Editing Agent 接收当前累计编辑图以及已完成步骤，判断本步是 `edit` 还是 `already_satisfied`；
- `edit` 输出受控区域编辑 prompt 和与当前指令对应的结构化 `verification_spec`；
- `already_satisfied` 跳过重复生成，但仍由验证器检查当前图中确实存在目标效果；
- `verification_spec` 明确目标变化、目标区域、放置/尺度/景深约束、必须保持项和禁止出现项；
- 验证器比较编辑前后图、当前指令和同一份 `verification_spec`，不能只依据泛化情绪描述通过；
- 保护约束按实体身份、数量、布局、不可变属性和本步授权的可变属性区分，避免“保护整个实体”与“修改其一个属性”互相矛盾。

## 本实现补全，不是论文逐字提示词

- 所有 JSON schema 和字段名；
- `max_step_retries`、`max_critic_rounds` 等有限终止条件；
- FLUX.2 prompt 中的 identity、pose、camera viewpoint 等保留约束；
- 无标签推理集采用论文中的 VLM 源情绪预测；
- 中性语义查询、emotion/type 元数据过滤及全局 diverse top-k 重排；
- 跨全部方案的联合语义去重和修订后二次检查；
- 当前累计图、`already_satisfied` 分支和结构化 `verification_spec`；
- ResNet 概率作为 Critic 的辅助信息；
- 不输出模型隐藏思维，只保存可核验的结论和 Action/Observation；
- 原文多工具库统一替换为任务书指定的 FLUX.2-klein-9B。

汇报时应表述为：“提示词依据论文图 2、图 18、图 19重建，论文未公开完整 system prompt。”
