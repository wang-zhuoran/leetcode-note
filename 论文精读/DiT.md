# [论文精读] Scalable Diffusion Models with Transformers (DiT)

> **Paper**: Scalable Diffusion Models with Transformers (ICCV 2023)
> **Authors**: William Peebles, Saining Xie
> **TL;DR**: DiT 将扩散模型中霸榜已久的 U-Net 替换为标准的 Vision Transformer (ViT)，证明了扩散模型也能遵循 Scaling Laws（扩展定律），是后续 Sora 等视频生成模型的核心基座。

---

## 1. 核心理念：为什么抛弃 U-Net？

在 DiT 之前，以 Stable Diffusion (LDM) 为代表的主流模型都使用 **U-Net**。
*   **U-Net 的优势**：拥有很强的 Inductive Bias（归纳偏置），卷积层的平移不变性非常适合处理图像。
*   **U-Net 的劣势**：网络设计复杂（Downsample, Upsample, Skip Connection），且不清楚如何通过简单地堆参数来获得更好的效果（Scaling property 不明显）。

**DiT 的目标**：解构扩散模型，证明 Transformer 的架构统一性（Architecture Unification）在生成领域同样适用。**只要计算量给够，ViT 能够比 U-Net 学得更好。**

---

## 2. DiT 架构详解 (The DiT Architecture)

DiT 并不是直接在像素空间跑 Transformer（太慢），而是沿用了 Latent Diffusion (LDM) 的两阶段范式。

### 2.1 整体流程
1.  **VAE Encoder**: $x (256\times256\times3) \rightarrow z (32\times32\times4)$。
2.  **Patchify (关键步骤)**: 将 Latent $z$ 切分成 $p \times p$ 的小块。
    *   输入 Latent 尺寸 $I \times I$。
    *   Patch Size 为 $p$。
    *   序列长度 $T = (I/p)^2$。
    *   **复杂度分析（重要勘误）**：这里解释了为什么单纯看“参数量”是不够的。
        *   Transformer 的参数量主要取决于 Hidden Dimension ($d$) 和层数 ($N$)。
        *   但是，**Gflops (计算量)** 受到序列长度 $T$ 的影响极大（Attention 是 $O(T^2)$ 复杂度）。
        *   **Case**: 即使模型参数量不变，如果把 $p$ 从 4 减小到 2，$T$ 会变成原来的 4 倍，Gflops 会暴涨。因此，**Gflops 才是衡量 DiT 图像生成质量的最强指标，而非参数量。**
3.  **DiT Blocks**: 处理序列，预测噪声。
4.  **Depatchify & Decoder**: 还原回噪声图，计算 Loss。

---

## 3. Deep Dive: adaLN-Zero

这是 DiT 能够击败 U-Net 的核心组件。作者探索了 In-context (像 GPT 一样拼 token)、Cross-attention 等方法，最终锁定了 **adaLN-Zero**。

### 3.1 什么是 adaLN (Adaptive Layer Norm)?
标准的 Layer Normalization 公式为：
$$ \text{LayerNorm}(x) = \frac{x - \mu}{\sigma} \cdot \gamma + \beta $$
其中 $\gamma, \beta$ 通常是可以通过反向传播学习的**固定参数**。

而在 **Adaptive** Layer Norm 中，$\gamma$ 和 $\beta$ 不是固定的，而是根据条件（Conditioning）**动态生成**的。在 DiT 中，条件是时间步 $t$ 和 类别标签 $c$。

### 3.2 adaLN-Zero 的数学机制
DiT Block 需要根据 $t$ 和 $c$ 来调节网络对特征的处理。

1.  **输入嵌入 (Embedding)**:
    *   $t_{emb} = \text{MLP}(\text{Sinusoidal}(t))$
    *   $c_{emb} = \text{EmbeddingTable}(c)$
    *   $cond = t_{emb} + c_{emb}$ (直接相加)

2.  **回归参数 (Regression)**:
    将 $cond$ 输入到一个 MLP (SiLU -> Linear) 中，这个 MLP 不止输出一组 $\gamma, \beta$，而是为每个 DiT Block 内部的两个子模块（Attention 和 FFN）分别输出 **缩放 (Scale)** 和 **平移 (Shift)** 参数，以及一个 **门控 (Gate)** 参数。
    
    一个 Block 共输出 **6 个参数**（每个都是向量，维度等于 Hidden Size $D$）：
    *   **Attention 部分**: $\gamma_1, \beta_1, \alpha_1$
    *   **FFN 部分**: $\gamma_2, \beta_2, \alpha_2$

3.  **Modulation (调制过程)**:
    在 DiT Block 内部，残差连接前的计算公式如下（以 Attention 部分为例）：
    
    $x_{norm} = \text{LayerNorm}(x) $
    
    $x_{modulated} = x_{norm} \cdot (1 + \gamma_1) + \beta_1 $
    
    $x_{attn} = \text{SelfAttention}(x_{modulated}) $
    
    $x_{out} = x + \alpha_1 \cdot x_{attn} $

    *注意：这里的 $ \alpha_1$ 作用在残差连接之前，控制整个 Block 的输出强度。*

### 3.3 为什么要 "Zero"? (初始化技巧)
**Zero** 指的是对上述回归 MLP 的**最后一个 Linear 层**进行**零初始化 (Zero-Initialization)**。

*   **初始状态**:
    *   由于权重为 0，MLP 输出的所有 $\gamma, \beta, \alpha$ 初始值全为 **0**。
*   **代入公式**:
    *   $x_{modulated} = x_{norm} \cdot (1 + 0) + 0 = x_{norm}$ (退化为普通 Norm)
    *   $x_{out} = x + 0 \cdot x_{attn} = x$ (**恒等映射**)
*   **意义**:
    这意味着在训练刚开始时，每一个 DiT Block 实际上都是一个**Identity Mapping（恒等映射）**，输入信号可以直接无损地流过整个深层网络。
    这模拟了 ResNet 中的 `Zero Init` 策略，极大地稳定了深层 Transformer 的梯度传播，使得模型可以堆得非常深而不会梯度消失或发散。

---

## 4. 关键数学公式复习

### 4.1 Diffusion Loss
DiT 的本质是预测噪声。虽然推导复杂，但 Loss Function 很简单：

$\mathcal{L}_{simple} = \mathbb{E}_{x_0, \epsilon, t} \left[ \| \epsilon - \epsilon_\theta(x_t, t, c) \|^2 \right]$

*   模型 $\epsilon_\theta$ 接收带噪潜变量 $x_t$，通过 adaLN-Zero 接收 $t, c$，输出预测的噪声。

### 4.2 Classifier-Free Guidance (CFG)
面试必问公式。如何在没有分类器的情况下增强条件控制？

$$ \hat{\epsilon}_\theta(x_t, c) = \epsilon_\theta(x_t, \emptyset) + s \cdot [\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset)] $$

*   $\epsilon_\theta(x_t, \emptyset)$: 无条件预测（训练时用空标签 Null token 得到）。
*   $\epsilon_\theta(x_t, c)$: 有条件预测。
*   $s$: Guidance Scale（通常 > 1）。
*   **直觉**: 计算“有条件”和“无条件”的差值向量（代表了条件 $c$ 的方向），然后在这个方向上放大 $s$ 倍。

---

## 5. Scaling Analysis (实验结论)

论文最核心的发现是**Gflops vs. FID** 的关系。

1.  **Gflops 是唯一真神**:
    不要只看参数量 (Parameters)。要提高图像质量 (降低 FID)，可以通过两种方式增加 Gflops：
    *   **Network Complexity**: 把模型做大 (增加层数 $N$，增加宽度 $d$) -> 增加参数量。
    *   **Sample Complexity**: 把 Patch Size ($p$) 做小 -> 参数量不变，但序列变长，计算量指数级上升。
    
    *结论：DiT-XL/2 (最大模型，最小 Patch) 效果最好。*

2.  **训练效率**:
    大模型 (High Gflops) 并非单纯的浪费算力。实验表明，大模型在训练早期就能以更少的 Step 达到小模型训练很久才能达到的 FID。

---

## 6. 总结与复习 Checkpoint

**为什么 DiT 重要？**
它打通了 Generative Model 和 NLP 架构的壁垒。Sora 的技术报告中明确提到使用了 DiT 架构。

**复习自测题**:
1.  **[架构]** DiT 的输入是什么形状？（答：Sequence of Patches, from Latent Space）。
2.  **[细节]** 为什么参数量相同的情况下，Patch Size 越小，计算量越大？（答：Token 数量 $T$ 增加，Attention 是 $O(T^2)$）。
3.  **[机制]** adaLN-Zero 中的 "Zero" 到底初始化了谁？有什么作用？（答：初始化了生成 $\gamma, \beta, \alpha$ 的 MLP 的最后一层；使得 Block 初始状态为恒等映射，利于深层训练）。
4.  **[采样]** CFG Scale $s=1$ 意味着什么？（答：意味着不使用无分类器引导，等同于标准的条件生成）。
