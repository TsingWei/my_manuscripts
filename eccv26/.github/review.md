

以下从审稿人视角，按照新颖性、技术方法、实验、写作四个维度给出详细的改进建议。

---

## 一、新颖性与动机（Novelty & Motivation）

**1. Motion Window 与经典惩罚窗口的关系未讨论**

这是最关键的问题。论文提出的 Motion Window（基于前帧预测中心的 2D 高斯加权）与 SiamFC、SiamRPN++、ATOM 等经典追踪器中长期使用的 cosine/Gaussian penalty window 在思想上高度相似——都是利用目标运动的时序平滑先验对搜索区域进行空间加权。论文完全没有讨论这一关联，审稿人很可能会认为这是"将已有 trick 从后处理移到 token 选择阶段"，新颖性不足。

**建议：**
- 在 Related Work 或 Method 中明确讨论与经典 penalty window 的区别与联系，强调你的做法将空间先验从"后处理打分"前移到了"计算图裁剪"，从而实现了实质性的计算节省，而非仅仅改变排序。
- 如果可能，做一个对比实验：在预测头后端加传统 penalty window（不做 token sparsification），与你的方法对比 FLOPs 和精度，以此凸显将先验嵌入计算图的独特价值。

**2. 1×1 Conv Head 的技术贡献偏弱**

将 3×3 conv 替换为 1×1 conv 以适配稀疏 token 是一个自然的工程选择，缺乏足够的技术深度。审稿人可能会认为这不是一个独立的贡献点。

**建议：**
- 降低其作为独立贡献点的权重，或者增加更多分析，例如：1×1 head 为什么在稀疏场景下不仅更快，而且精度损失极小？是否因为稀疏 token 已经经过空间聚焦，上下文信息的需求降低了？这种分析能提升深度。
- 考虑讨论其他可能的 sparse head 设计（如 MLP-based、graph-based），说明 1×1 conv 是经过比较选出的方案。

---

## 二、技术方法（Technical Soundness）

**3. 训练与推理的不一致性**

论文最后一句提到"sparsification rate is gradually warmed-up during training following OSTrack, **without motion window added**"。这意味着训练时不使用 Motion Window，但推理时使用。这个 train-test mismatch 是一个严重的技术隐患，且论文几乎没有解释为什么这样做、以及这种不一致带来的影响。

**建议：**
- 增加一段专门讨论：为什么训练时不加 motion window？是因为训练数据是随机采样的帧对、缺乏连续帧的前帧预测？
- 如果是这个原因，应该讨论是否尝试过在训练中模拟 motion window（例如用 GT 加噪声来生成伪前帧预测），以及效果如何。
- 这个 gap 可能是导致你需要 dense auxiliary head 辅助收敛的根本原因，值得深入分析。

**4. 预测漂移（Drift）的鲁棒性不足**

Motion Window 强依赖前帧预测的准确性。一旦预测错误（如遮挡后漂移到相似物体），Motion Window 会进一步锁定错误区域，形成正反馈的"误差累积"。论文在 Per-Attribute Analysis 中也承认了 background clutter 场景下性能下降，但没有提出任何缓解策略。

**建议：**
- 增加一个简单的失败恢复机制的讨论或实验，例如：当预测置信度低于阈值时，扩大 \(\gamma\) 或退回到全局 attention score。即使不实现，也应在 limitation 中详细讨论。
- 在 conclusion 的 limitation 部分，当前的表述过于轻描淡写（"reduced receptive field"），应该明确指出 drift 累积风险。

**5. 第一帧处理未说明**

第一帧没有"前帧预测"，Motion Window 如何初始化？这是推理流程中的一个关键细节，论文完全缺失。

**建议：**
- 明确说明第一帧的处理方式（例如：第一帧使用全局 attention score 不加 motion window，或使用初始 GT 框作为 motion window 中心）。

---

## 三、实验（Experiments）

**6. 命名不一致：MaST vs. MST**

文中描述使用"MaST"，但 Table I、Table II 中均写作"MST-nano"和"MST-tiny"。这是一个明显的低级错误，会严重影响审稿人的印象。

**建议：** 全文统一为"MaST"。

**7. GOT-10k 实验协议问题**

GOT-10k 的标准评估协议要求**仅使用 GOT-10k 训练集**进行训练，但论文 Training 部分明确写了使用 COCO、LaSOT、GOT-10k、TrackingNet 四个数据集。如果 GOT-10k 的结果不是在其限定协议下得到的，这是一个严重的实验规范问题。

**建议：**
- 如果确实按照 GOT-10k 协议单独训练了模型，请在论文中明确说明。
- 如果没有，需要补充协议内实验，或者在表格中标注，否则审稿人会直接质疑实验公平性。

**8. 仅有 ViT-Tiny 骨干，泛化性不足**

所有实验仅基于 ViT-Tiny。审稿人会质疑：Motion Window 策略是否对更大模型（如 ViT-Small、ViT-Base）同样有效？

**建议：**
- 至少补充一组 ViT-Small 的实验，展示方法在不同模型容量下的通用性。即使只放在 ablation 中也可以。

**9. 缺少延迟分解分析（Latency Breakdown）**

论文的核心卖点是端到端的稀疏流水线，但没有给出 encoder 和 head 各自的延迟占比。审稿人无法判断 1×1 head 实际节省了多少时间。

**建议：**
- 增加一个 latency breakdown 表格，分别列出 patch embedding、encoder（dense 部分 + sparse 部分）、head 的耗时，在 RPi5 和 N100 上测量。这能有力支撑"fully sparse pipeline"的价值。

**10. 缺少内存占用比较**

边缘设备不仅受限于算力，内存同样关键。论文没有报告任何内存相关指标。

**建议：**
- 补充参数量（Params）和峰值内存占用的对比。

**11. UAV123 数据不一致**

Table II 中 MaST-tiny 在 UAV123 的 AUC 为 66.6，但正文写的是 63.8。请核实。

---

## 四、写作与表述（Writing Quality）

**12. 摘要和贡献的措辞需要收紧**

- 摘要最后一句"low-resource platform"应为"platforms"。
- 贡献第三条写的是"Our MST"，应为"Our MaST"。
- 标题说"Fully Sparse Pipeline"，但训练时使用了 dense auxiliary head，这在严格意义上并不"fully sparse"。建议在标题或正文中做出区分（如"fully sparse inference pipeline"）。

**13. 图表质量**

- Fig. 1 中多个 tracker 的点重叠严重，辨识度低。建议增大图幅或使用不同标记形状。
- Fig. 3 信息密度过高，建议拆分为两个子图：(a) 整体流水线；(b) Motion-Aware Sparsification Block 的详细结构。

**14. 缺少与更多近期方法的比较**

2024-2025 年的高效追踪领域有不少新方法（如 ODTrack、DropTrack、SeqTrack 的轻量版本等），论文的 related work 和实验对比都偏老。

**建议：** 补充更多 2024 年的 baseline 对比，尤其是同样声称在边缘设备上实时的方法。

---

## 五、总结与优先级排序

按重要性排序，最需要优先处理的问题：

1. **GOT-10k 协议合规性**（可能导致直接拒稿）
2. **MaST/MST 命名不一致**（低级错误，严重影响印象）
3. **与经典 penalty window 的关系讨论**（新颖性的核心质疑点）
4. **训练不加 Motion Window 的合理性解释**（技术 soundness）
5. **第一帧处理、drift 机制**（方法完整性）
6. **Latency breakdown 和内存分析**（实验说服力）
7. **更大骨干的泛化实验**（方法通用性）
8. **UAV123 数据不一致**（需要核实）
9. **写作细节打磨**

整体而言，论文的问题定义清晰、实验设计相对完整、在边缘设备上做了真实测速（这是一个加分项）。核心需要加强的是新颖性的论证深度和技术细节的严谨性。如果能解决上述问题，论文的竞争力会有明显提升。