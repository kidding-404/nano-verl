![nano-verl](docs/logo.png)

# nano-verl

一个从零实现的轻量化 `verl` 风格 RL 训练框架。

## 核心特性

1. **可读性**：`nanoverl` 约 6k 行 vs `verl` 约 90K+ 行
2. **分布式**：以`FSDP+vLLM`作为训练+推理后端，使用`Ray`进行分布式管理，支持 `rollout load balancing`，`dynamic batch`，`remove padding` 等
3. **异步支持**：支持 `one-step-off-policy`异步 （通过`trainer.mode=one_step_off` 设置）

## 安装

1. 下载代码：
```bash
git clone 
cd nano-verl
```

2. 使用`uv`安装依赖： 
```
uv sync
```

3. 找到对应版本的[flash-attn whl](https://github.com/Dao-AILab/flash-attention/releases
)，单独安装：

```bash
uv run pip install <flash_attn_wheel_url>
```

## 快速开始

在`gsm8k`数据集上训练`qwen3-0.6B`：

```bash
uv run python main.py --config configs/gsm8k-qwen3-0.6b-single-gpu.yaml
```

还可以在两张卡上异步训练`qwen3-1.7B`：
```bash
uv run python main.py --config configs/gsm8k-qwen3-1.7b-1p1-async.yaml
```

## Benchmark

**测试配置：**
- Model：Qwen3-4B
- Trainset：DAPO-17K
- Reward: 1/-1 accuracy reward
- Steps：150
- Global batch size：64
- Rollout n：8
- Prompt length：1024
- Response length：8192
- Hardware：1 node, 8 x NVIDIA H100 80GB HBM3

**奖励曲线：**

![nano-verl vs verl reward convergence](docs/reward.png)

**性能比较：**

| Setting             | AIME24 avg16 | AIME24 pass@16 | AIME25 avg16 | AIME25 pass@16 |
| ------------------- | -----------: | -------------: | -----------: | -------------: |
| Qwen3-4B Base        |       0.4333 |         0.7000 |       0.3563 |         0.5333 |
| Qwen3-4B + verl      |       0.5313 |         0.8333 |       0.4417 |         0.6667 |
| Qwen3-4B + nano-verl |        0.535 |         0.8333 |        0.429 |         0.6667 |
