# 🚀 Prompt Engineering 实战笔记：从 K-Shot 到 Reflexion
在本次 Assignment 1 中，我通过 Python 与 Llama 3.1 8B 模型的交互，深入探索了如何通过优化 Prompt 来显著提升 LLM 在处理复杂任务时的稳定性与准确性。以下是核心技术的总结与最佳实践。

| 技巧名称 | 核心定义 | 使用场景 | 核心技巧 |
| :--- | :--- | :--- | :--- |
| **K-Shot Prompting** 🎯 | 提供 1-k 个示例作为参考 | 字符处理、特定格式输出 | 示例要具有代表性，格式要对齐 |
| **Chain-of-Thought** ⛓️ | 引导模型写出思考步骤 | 数学逻辑、复杂推理 | 使用 "Step by Step" 引导词 |
| **Tool Calling** 🔧 | 引导模型生成结构化指令调用外部工具 | API 调用、数据库查询 | 必须严格定义 JSON 格式与参数 |
| **Self-Consistency** ⚖️ | 运行多次并进行多数投票（Majority Vote） | 数学题、确定性答案任务 | 配合较的高 Temperature 使用 |
| **RAG** 📚 | 将外部知识库内容喂给模型 | 垂直领域问答、文档分析 | 仅允许模型使用 Context 范围内的信息 |
| **Reflexion** 🔄 | 基于错误反馈进行自我修正 | 代码调试、迭代优化 | 反馈信息要包含具体的失败原因 |

💡 深度解析与最佳实践
1. K-Shot Prompting: 突破 Token 限制 🧩
对于模型而言，单词往往是以 Token 形式存在的，这导致它在处理“反转字符串”等任务时容易出错。

- 技巧： 提供清晰的 Input/Output 示例对。
- 注意： 示例的格式必须与最终任务完全一致。

2. Chain-of-Thought (CoT): 推理的基石 🧠
模型在直接输出答案时很容易产生幻觉，让它“说出思考过程”能显著提高逻辑准确率。

- 技巧： 在 System Prompt 中要求模型“先分析已知条件，再列出等式，最后给出答案”。
- 注意： 确保最后一行有明显的标识符（如 Answer: <number>）以便解析。

3. Tool Calling: 实现 AI 智能体 🤖
让模型生成 JSON 格式的工具调用指令。

- 技巧： 在 System Prompt 中提供清晰的工具 Schema（参数类型、功能描述）。
- 注意： 强调 ONLY output valid JSON，防止模型输出多余的解释文本导致解析失败。

4. Self-Consistency: 概率论的应用 📊
单次生成的答案可能有偶然性。通过代码运行 5 次，选出出现次数最多的答案。
- 技巧： 适当调高 Temperature（例如 0.7 - 1.0），增加答案的多样性。
- 场景： 适用于答案空间有限（如数值、True/False）的任务。
5. RAG (Retrieval-Augmented Generation): 知识外挂 📖
模型无法记住所有实时信息。通过检索文档并作为 Context 传入。

- 技巧： 在 Prompt 中加入强制约束：“Use ONLY the provided context”。
- 注意： 示例（Few-Shot）中应演示如何从文档中提取 Base URL 或特定的 Header。

6. Reflexion: 闭环学习 🔄
这是最高级的技巧之一。如果代码运行出错，将 Traceback 或失败原因反馈给模型。

- 技巧： User Prompt 应包含：Previous Code + Failure Reasons + Re-implement Task。
- 注意： 这种迭代非常依赖模型的理解力，反馈信息越具体（例如哪一个测试用例没过），模型修复成功的概率越高。

🌟 额外的心得体会
不要信任默认输出： 即使是 8B 的小模型，通过合理的 K-Shot 和结构化约束，也能完成非常复杂的逻辑任务。

格式检查是关键： 使用正则表达式（Regex）配合提取模型输出的特定部分（代码块、JSON、最终答案）是工程实践中不可或缺的一环。

迭代思维： Prompt Engineering 不是一次性的，需要根据模型的失败案例不断微调 System Prompt 的边界条件。
