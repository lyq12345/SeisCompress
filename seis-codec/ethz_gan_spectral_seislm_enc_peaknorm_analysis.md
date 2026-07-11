# ETHZ GAN Spectral / SeisLM Encoder PeakNorm 训练结果分析

分析时间：2026-07-10  
日志目录：`/data/seismic/seis-codec-logs/ethz_gan_spectral/seislm_enc_peaknorm/`

## 结论

这次训练相比同组 `clip_normvq` 稳定很多，说明 `peak norm + SeisLM encoder + stable VQ` 方向有效；但最佳点出现在中后段，后 20 多个 epoch 明显退化，而且 checkpoint 保存状态有问题，不能直接信任当前目录里的 `last.ckpt` 代表最终模型。

## 训练配置摘要

- 数据集：ETHZ
- 输入窗口：`3001`
- 归一化：`amp_norm_type=peak`
- batch size：`16`
- 模型：
  - `use_seislm_encoder=true`
  - `freeze_seislm_extractor=false`
  - `use_stable_quantizer=true`
  - encoder / decoder rates 均为 `[2, 2, 2]`
- 训练：
  - `use_gan=true`
  - `use_spectral_loss=true`
  - `use_task_aware_loss=false`
  - learning rate：`1e-4`
  - max epochs：`100`
  - `gradient_clip_g=1000`
  - `gradient_clip_d=10`

## 关键指标

`val/loss` 的定义为：

```text
val/loss = 100 * val/l1 + val/loss_spectral
```

因为本次 `use_task_aware_loss=false`，task-aware loss 没有参与。

| 指标 | 初始 | 最优 | 最后 | 结论 |
|---|---:|---:|---:|---|
| `val/l1/ETHZ` | 0.04622 | 0.02384 @ epoch≈72 | 0.02851 | 最优减半，但最后比最优差 19.6% |
| `val/loss_spectral/ETHZ` | 3.453 | 1.990 @ epoch≈70 | 2.128 | 频谱损失改善明显，但后期回升 |
| `val/loss/ETHZ` | 8.075 | 4.390 @ epoch≈75 | 4.979 | 最后比最优差 13.4% |
| `train/loss_g` | 27.75 | 15.89 | 19.05 | 后期训练本身也在变差 |
| `latent_norm_mean` | 6.1 | — | 170.7 | latent 尺度持续漂移 |
| `latent_norm_max` | 9.3 | peak 3454 | 2304 | 存在严重 outlier / 尺度膨胀 |

按 10 epoch block 看验证集趋势：

| Epoch block | `val/loss` | `val/l1` | `val/loss_spectral` |
|---|---:|---:|---:|
| 00-09 | 6.042 | 0.03275 | 2.767 |
| 10-19 | 5.341 | 0.02890 | 2.451 |
| 20-29 | 5.132 | 0.02797 | 2.335 |
| 30-39 | 4.750 | 0.02587 | 2.164 |
| 40-49 | 4.745 | 0.02602 | 2.143 |
| 50-59 | 4.677 | 0.02582 | 2.094 |
| 60-69 | 4.688 | 0.02606 | 2.082 |
| 70-79 | 4.544 | 0.02504 | 2.040 |
| 80-89 | 4.640 | 0.02566 | 2.074 |
| 90-99 | 4.936 | 0.02777 | 2.159 |

最佳验证点集中在 epoch 70-75 附近，之后整体退化。

## 暴露出的问题

### 1. 后期退化明显

验证集在 epoch 70-75 左右达到最佳，之后 `val/l1`、`val/loss` 都上升。训练侧 `train/l1`、`train/loss_g` 后期也回升，因此不是简单的“继续训练会更好”。

这更像训练后期不稳定或 latent / encoder 表征尺度漂移导致的退化，而不仅仅是传统过拟合。

### 2. latent 尺度持续膨胀

`latent_norm_mean` 从 6.1 涨到 170.7，`latent_norm_max` 达到 2k-3k 量级。

当前 `quantize_stable.py` 中 VQ lookup/loss 在 normalized 空间计算，能避免 VQ loss 直接爆炸，但也使 raw latent magnitude 缺少有效约束。模型可以持续推高 latent 尺度，而 normalized VQ loss 不会强烈惩罚这一点。

这和后期验证指标退化高度相关。

### 3. checkpoint 保存状态异常

event 日志显示训练跑到了：

```text
epoch=99, step=17599
```

但目录下 `last.ckpt` 内部 metadata 是：

```text
epoch=50, global_step=17952
```

文件修改时间也对应 event 中 epoch 50 左右。

因此，当前目录中的 `last.ckpt` 不应被当作最终模型。训练后半段没有在该目录下留下新的 checkpoint，或者日志 / checkpoint 状态发生了不一致。

### 4. 验证流程不是严格 deterministic

`get_val_augmentations()` 中仍使用了随机窗口逻辑，例如：

- `WindowAroundSample(... selection="random")`
- `RandomWindow(...)`

这会让每个 epoch 的验证窗口变化，导致验证曲线带噪声，也会影响 best checkpoint 的选择可靠性。

### 5. GAN 没有崩，但后期 D/G 平衡变差

`loss_d` 基本维持在 3.6-3.9，没有明显 collapse；但 `loss_g_adv` 后期上升，feature matching 仍然是 generator loss 大头。后期 GAN 部分没有继续带来重建收益，反而可能参与了退化。

## 与 `clip_normvq` 的粗略对比

同组 `clip_normvq` 版本出现了明显数值爆炸：

- `val/l1/ETHZ` 最优仍在 336 量级
- `train/latent_norm_mean` 最后达到 `7.99e16`
- `encoder_latent_norm_mean` 最后达到 `4.87e16`

当前 `seislm_enc_peaknorm` 的指标量级恢复正常：

- `val/l1/ETHZ` 最优为 0.02384
- `val/loss/ETHZ` 最优为 4.38956
- latent norm 虽然仍在漂移，但没有到灾难性爆炸

因此当前方案是明显进步，但还没有完全解决 latent 尺度控制问题。

## 改进建议

### 优先级 1：修 checkpoint 和验证流程

- 每次实验使用全新的 `log_version`，避免复用目录。
- 训练结束显式打印并保存：
  - `best_model_path`
  - `best_model_score`
  - `last_model_path`
- 额外保存一个明确的 final checkpoint。
- validation 改为固定窗口或固定 seed。
- 最好预生成 deterministic validation set，避免每个 epoch 验证样本窗口变化。

### 优先级 2：使用 early stopping

当前曲线不支持跑满 100 epoch 后取最后模型。建议：

- 如果目标是综合重建质量，monitor `val/loss/ETHZ`。
- 如果只关心 waveform L1，monitor `val/l1/ETHZ`。
- patience 可先设为 10-15。
- 当前实验的可用模型区间应优先考虑 epoch 70-75。

### 优先级 3：控制 latent 尺度漂移

建议尝试：

- 对 encoder output 或 quantizer `z_e` 加 magnitude regularization，例如：

```text
lambda_latent * mean(z_e ** 2)
```

- 在每个 quantizer `in_proj` 后加 norm / clamp。
- 降低 `gradient_clip_g`，当前 `1000` 太宽，建议先试 `50` 或 `100`。
- 记录更多诊断指标：
  - codebook usage
  - codebook perplexity
  - raw latent RMS
  - input peak 分布
  - per-channel reconstruction error

### 优先级 4：更保守地 fine-tune SeisLM encoder

当前 `freeze_seislm_extractor=false`，SeisLM feature extractor 全量参与训练。建议做 ablation：

- 前 N 个 epoch freeze extractor，只训练 adapter / quantizer / decoder。
- 或者 extractor 使用更小 LR，例如 `1e-5`，其他模块保持 `1e-4`。
- 加 cosine decay 或 ReduceLROnPlateau，避免后期继续大步漂移。

### 优先级 5：调整 GAN 训练策略

建议尝试：

- reconstruction + spectral 预训 20-40 epoch 后再打开 GAN。
- feature matching 权重从 `2.0` 降到 `1.0` 做 ablation。
- 对 discriminator 使用更低 LR 或 update ratio 调整。
- 对 GAN loss 做 warmup，而不是从 epoch 0 全权重加入。

## 下一步推荐实验

建议优先跑一个小型 ablation：

1. deterministic validation + 修 checkpoint。
2. monitor `val/loss/ETHZ`，加 early stopping。
3. `gradient_clip_g=100`。
4. SeisLM extractor 前 20 epoch freeze，之后用较小 LR 解冻。
5. 加 latent magnitude regularization。

如果该实验中：

- `latent_norm_mean` 不再单调飙升；
- `val/loss` 在最佳点附近更平稳；
- checkpoint 能正确保存 best 和 last；

则说明当前主问题基本定位正确。
