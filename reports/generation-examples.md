# TinyStories 文本生成示例

使用第 5,000 步的基线 checkpoint 和训练时的 tokenizer；以下结果在 CPU 上生成。
采样参数为 temperature=0.8、top_p=0.9，每个示例的新增 token 上限为 192。
完整命令见 README；原始输出与结构化记录保存在报告目录。

## story-1

Prompt：`Once upon a time`

随机种子：42；prompt tokens：4；新增 tokens：138；停止原因：生成了 `<|endoftext|>`。

```text
Once upon a time, there was a little girl named Lily. She had a big toy box with a lock. Lily loved her toys very much. One day, she wanted to make her toys into a fun toy box.
Lily started to unpack her toys. But then, she heard a loud noise coming from the toy box. Lily was scared and didn't know what to do. She thought the toy box would be a surprise.
Lily's mom came into the room and saw the toy box. She knew it was not her toy. But she said, "Lily, please be careful. The toy box is broken." Lily was very happy and promised to be more careful with her toys.

```

原始输出：[日志](generation-story-1.txt)。

## story-2

Prompt：`Lily found a small puppy in the garden.`

随机种子：43；prompt tokens：9；新增 tokens：97；停止原因：生成了 `<|endoftext|>`。

```text
Lily found a small puppy in the garden. She was very happy and wanted to play with it. She picked up the puppy and ran around the garden. The puppy was very fast and jumped up and down.
Lily played with the puppy all day. She would run, jump, and chase it. She was so glad she made it so much that she had so much fun. When the sun went down, the puppy went back home. Lily was tired but happy. She knew that she had made a new friend.

```

原始输出：[日志](generation-story-2.txt)。

## 验证记录

- [采样与控制流程检查](./generation-validation.txt)：温度、最小 nucleus、阈值 crossing token、ID 映射、精确阈值、top_p=1、EOS、上下文截取、eval/no_grad、生成上限。
- [真实模型检查](./generation-examples-validation.txt)：两段续写、300-token prompt 截取、零新增 token。
- [同种子可复现性](./generation-repeatability.txt)：相同 CPU 设备、配置和种子得到完全相同的 CLI 输出。
- [结构化示例](./generation-examples.json)。

检查中修正了累计概率相减造成的阈值舍入问题，并明确 top_p=1 保留完整分布。
示例有连贯的儿童故事叙述，但部分情节细节存在不一致；结果来自约 100 MiB 子集上训练的基线模型。
