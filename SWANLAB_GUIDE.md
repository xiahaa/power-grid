# SwanLab Integration Guide

## 📊 SwanLab for Experiment Tracking

SwanLab is a powerful experiment tracking and visualization tool specifically designed for machine learning experiments. It provides beautiful, interactive dashboards and is particularly suitable for power grid research projects.

---

## 🚀 Quick Start

### 1. Install SwanLab

```bash
pip install swanlab
```

### 2. Enable in Config

Edit your config file (e.g., `configs/ieee33_config.yaml`):

```yaml
logging:
  use_tensorboard: true          # Keep TensorBoard if you want
  use_swanlab: true              # Enable SwanLab
  swanlab_project: "power-grid-estimation"
  swanlab_experiment: "ieee33-graph-mamba"
  log_dir: "logs/ieee33"
  log_freq: 10
  save_visualization: true
```

### 3. Train with SwanLab

```bash
python scripts/train.py --config configs/ieee33_config.yaml
```

SwanLab will automatically:
- Create an experiment in your project
- Log all training metrics
- Track hyperparameters
- Save model checkpoints
- Generate interactive visualizations

---

## 📈 What Gets Logged

### Training Metrics (Per Epoch)

**Losses:**
- `train/loss_total`: Total combined loss
- `train/loss_state`: State estimation loss (V, θ)
- `train/loss_parameter`: Parameter estimation loss (R, X)
- `train/loss_physics`: Physics constraint violation

**Validation Losses:**
- `val/loss_total`, `val/loss_state`, `val/loss_parameter`, `val/loss_physics`

### State Estimation Metrics

**Voltage Magnitude:**
- `val/v_mag_rmse`: Root Mean Squared Error
- `val/v_mag_mae`: Mean Absolute Error
- `val/v_mag_mape`: Mean Absolute Percentage Error

**Voltage Angle:**
- `val/v_ang_rmse`, `val/v_ang_mae`

### Parameter Estimation Metrics

**Line Resistance:**
- `val/r_line_rmse`: RMSE for R estimation
- `val/r_line_mae`: MAE for R estimation

**Line Reactance:**
- `val/x_line_rmse`, `val/x_line_mae`

### Training Dynamics

- `train/learning_rate`: Current learning rate
- `train/epoch`: Current epoch number

### Summary (At End)

- `summary/best_val_loss`: Best validation loss achieved
- `summary/total_epochs`: Total epochs trained
- `summary/training_time_hours`: Total training time
- `summary/model_parameters`: Number of model parameters

---

## 🎛️ Hyperparameters Tracked

SwanLab automatically tracks all hyperparameters from your config:

- `system`: Power system name (ieee33, ieee118)
- `num_buses`: Number of buses
- `pmu_coverage`: PMU coverage ratio
- `batch_size`: Training batch size
- `learning_rate`: Initial learning rate
- `num_epochs`: Maximum epochs
- `spatial_encoder`: GAT configuration
- `temporal_encoder`: Mamba configuration
- `physics_enabled`: Whether physics constraints are used
- `constraint_type`: Soft or hard constraints

---

## 🌐 Accessing SwanLab Dashboard

### Online Dashboard (Recommended)

1. **Create Account** (First time only):
   ```bash
   swanlab login
   ```
   Follow the prompts to create an account at [swanlab.cn](https://swanlab.cn)

2. **View Experiments**:
   - Go to https://swanlab.cn
   - Select your project: `power-grid-estimation`
   - View all experiments and compare results

### Local Dashboard

If you prefer local hosting:

```bash
swanlab watch
```

This opens a local server at `http://localhost:5092`

---

## 📊 Visualization Features

### 1. **Training Curves**

Interactive plots showing:
- Loss curves (train vs validation)
- Metric evolution over epochs
- Learning rate schedule
- Component-wise losses

**Features:**
- Zoom, pan, and hover for details
- Compare multiple experiments
- Smooth curves with adjustable smoothing factor

### 2. **Hyperparameter Comparison**

Compare different configurations:
- Side-by-side metric comparison
- Hyperparameter table
- Best run identification

**Example queries:**
- "Which PMU coverage gives best accuracy?"
- "GAT vs GraphSage performance"
- "Impact of physics weight"

### 3. **System Metrics**

Real-time monitoring:
- GPU utilization
- Memory usage
- Training speed (samples/sec)
- Time per epoch

### 4. **Model Analysis**

- Parameter count
- Training efficiency
- Convergence speed
- Overfitting detection

---

## 🔧 Advanced Usage

### Custom Logging

Add custom metrics in your training script:

```python
import swanlab

# Log custom values
swanlab.log({
    'custom/grid_topology_score': score,
    'custom/constraint_satisfaction': satisfaction_rate,
}, step=epoch)

# Log images
swanlab.log({
    'visualization/voltage_heatmap': swanlab.Image(voltage_map)
}, step=epoch)

# Log text
swanlab.log({
    'notes/observations': 'Interesting pattern observed at epoch 50'
}, step=epoch)
```

### Compare Experiments

```python
# scripts/compare_experiments.py
import swanlab

api = swanlab.Api()
runs = api.runs("power-grid-estimation")

for run in runs:
    print(f"{run.name}: {run.summary['val/v_mag_rmse']:.6f}")
```

### Export Results

```python
# Get experiment results
api = swanlab.Api()
run = api.run("power-grid-estimation/ieee33-graph-mamba")

# Export to DataFrame
import pandas as pd
history = run.history()
df = pd.DataFrame(history)
df.to_csv('experiment_results.csv')
```

---

## 🎯 Best Practices

### 1. **Naming Conventions**

Use descriptive experiment names:

```yaml
# Good
swanlab_experiment: "ieee33-gat4heads-mamba3layers-physics0.1"

# Bad
swanlab_experiment: "experiment1"
```

### 2. **Project Organization**

Organize by research phase:

```yaml
# Baseline experiments
swanlab_project: "power-grid-baselines"

# Main experiments
swanlab_project: "power-grid-estimation"

# Ablation studies
swanlab_project: "power-grid-ablation"
```

### 3. **Tags and Notes**

Add tags in code:

```python
swanlab_run = swanlab.init(
    project="power-grid-estimation",
    experiment_name="ieee33-graph-mamba",
    tags=["baseline", "full-model", "ieee33"],
    notes="Initial baseline with standard hyperparameters"
)
```

### 4. **Checkpointing**

Save important artifacts:

```python
# Save model to SwanLab
swanlab.save('checkpoints/ieee33/best_model.pt')

# Save config
swanlab.save('configs/ieee33_config.yaml')
```

---

## 📋 Example Workflow

### Scenario: Hyperparameter Tuning

```bash
# Experiment 1: Baseline
python scripts/train.py --config configs/ieee33_config.yaml

# Experiment 2: More GAT heads
# Edit config: num_heads: 8
python scripts/train.py --config configs/ieee33_config.yaml

# Experiment 3: Stronger physics constraint
# Edit config: physics_weight: 0.2
python scripts/train.py --config configs/ieee33_config.yaml

# Compare all experiments in SwanLab dashboard
# Find best configuration
```

### Scenario: Ablation Study

```yaml
# configs/ablation_*.yaml

# No physics
physics:
  enabled: false

# Soft constraints only
physics:
  constraint_type: "soft"

# Hard constraints
physics:
  constraint_type: "hard"
```

Run all and compare:

```bash
for config in configs/ablation_*.yaml; do
    python scripts/train.py --config $config
done
```

View side-by-side comparison in SwanLab.

---

## 🔍 Troubleshooting

### SwanLab not logging?

**Check:**
1. SwanLab installed: `pip list | grep swanlab`
2. Config enabled: `use_swanlab: true`
3. Network connection (for online mode)

**Debug:**
```python
import swanlab
print(swanlab.__version__)  # Should be >= 0.3.0
```

### Login issues?

```bash
# Re-authenticate
swanlab login --relogin

# Use API key directly
swanlab login --api-key YOUR_API_KEY
```

### Dashboard not showing metrics?

**Ensure metrics are logged:**
```python
# Check if swanlab_run is initialized
if swanlab_run:
    swanlab.log({'test': 123}, step=1)
```

---

## 📊 SwanLab vs TensorBoard

| Feature | SwanLab | TensorBoard |
|---------|---------|-------------|
| **Cloud Hosting** | ✅ Built-in | ❌ Manual setup |
| **Experiment Comparison** | ✅ Easy | ⚠️ Complex |
| **Hyperparameter Tracking** | ✅ Automatic | ⚠️ Manual |
| **Model Registry** | ✅ Yes | ❌ No |
| **Collaboration** | ✅ Easy sharing | ⚠️ Difficult |
| **Offline Mode** | ✅ Yes | ✅ Yes |
| **Custom Plots** | ✅ Rich | ⚠️ Limited |

**Recommendation:** Use both!
- TensorBoard: Local debugging, quick checks
- SwanLab: Experiment tracking, comparison, sharing

---

## 🎓 Learning Resources

### Official Documentation
- [SwanLab Docs](https://docs.swanlab.cn)
- [Python API Reference](https://docs.swanlab.cn/api)
- [Examples](https://github.com/swanhubx/swanlab-examples)

### Video Tutorials
- [Getting Started with SwanLab](https://www.bilibili.com/video/BV1...)
- [Advanced Features](https://www.bilibili.com/video/BV1...)

### Community
- [GitHub](https://github.com/swanhubx/swanlab)
- [Discord](https://discord.gg/swanlab)

---

## 📝 Summary

SwanLab integration provides:

✅ **Automatic metric logging** - No manual tracking
✅ **Beautiful dashboards** - Interactive visualizations
✅ **Experiment comparison** - Side-by-side analysis
✅ **Hyperparameter tracking** - All configs saved
✅ **Cloud storage** - Access from anywhere
✅ **Collaboration** - Easy sharing with team
✅ **Reproducibility** - Full experiment history

**Get started now:**

```bash
pip install swanlab
python scripts/train.py --config configs/ieee33_config.yaml
```

Visit your dashboard at https://swanlab.cn and explore your experiments! 🚀

---

*Last updated: 2026-01-18*
*SwanLab version: 0.3.0+*


 SwanLab 集成完成！
我已经成功将 SwanLab 实验跟踪工具集成到你的 Physics-Informed Graph Mamba 项目中。以下是完整的修改总结：
📝 修改文件列表
1. 依赖更新 (requirements.txt)
✅ 添加 swanlab>=0.3.0
2. 配置文件 (2个文件)
✅ configs/ieee33_config.yaml - 添加 SwanLab 配置
✅ configs/ieee118_config.yaml - 添加 SwanLab 配置
新增配置项：
logging:  use_swanlab: true  swanlab_project: "power-grid-estimation"  swanlab_experiment: "ieee33-graph-mamba"
3. 训练脚本 (scripts/train.py)
✅ 导入 SwanLab（带错误处理）
✅ 初始化 SwanLab 实验
✅ 自动记录所有训练指标
✅ 自动跟踪超参数
✅ 训练结束时保存摘要
4. 新文档 (2个文件)
✅ SWANLAB_GUIDE.md - 400+行完整使用指南
✅ SWANLAB_INTEGRATION.md - 集成实现总结
5. 测试脚本 (scripts/test_swanlab.py)
✅ 验证 SwanLab 安装
✅ 测试初始化和日志记录
✅ 检查配置文件集成
6. README 更新 (README.md)
✅ 添加实验跟踪章节
✅ SwanLab vs TensorBoard 对比
🎯 记录的指标
每个 Epoch 自动记录：
训练损失：
train/loss_total, train/loss_state, train/loss_parameter, train/loss_physics
验证损失：
val/loss_total, val/loss_state, val/loss_parameter, val/loss_physics
状态估计指标：
val/v_mag_rmse, val/v_mag_mae, val/v_mag_mape
val/v_ang_rmse, val/v_ang_mae
参数估计指标：
val/r_line_rmse, val/r_line_mae
val/x_line_rmse, val/x_line_mae
训练动态：
train/learning_rate, train/epoch
训练结束摘要：
summary/best_val_loss, summary/total_epochs, summary/training_time_hours, summary/model_parameters
🚀 快速开始
1. 安装 SwanLab
pip install swanlab
2. 测试集成
python scripts/test_swanlab.py
3. 训练模型（SwanLab 自动记录）
python scripts/train.py --config configs/ieee33_config.yaml
4. 查看仪表板
在线：访问 https://swanlab.cn （首次需要 swanlab login）
本地：运行 swanlab watch
✨ 主要优势
☁️ 云端托管：随时随地访问实验结果
📊 自动跟踪：无需手动记录指标和超参数
🔄 实验对比：轻松比较不同配置的性能
👥 团队协作：方便分享和讨论结果
📱 移动访问：支持手机查看实验进度
🎨 美观交互：现代化的可视化界面
🔌 无侵入式：如果未安装，自动降级不影响训练
🎯 使用场景
超参数调优
# 实验 1：基线python scripts/train.py --config configs/ieee33_config.yaml# 实验 2：调整 GAT heads# 修改 config: num_heads: 8python scripts/train.py --config configs/ieee33_config.yaml# 实验 3：增强物理约束# 修改 config: physics_weight: 0.2python scripts/train.py --config configs/ieee33_config.yaml# 在 SwanLab 仪表板中对比所有实验
消融研究
在 SwanLab 中对比有无物理约束的影响
比较不同空间编码器（GAT vs GraphSage）
分析 Mamba vs LSTM 的性能差异
📊 仪表板功能
访问 https://swanlab.cn 后，你将看到：
概览页
实验状态、运行时间、系统指标
超参数完整列表
GPU/内存使用情况
图表页
训练/验证损失曲线
所有指标随时间变化
学习率调度可视化
可缩放、平滑、对比
对比页
多实验并排比较
超参数表格
最佳运行识别
文件页
模型检查点
配置文件
生成的图表
📚 文档资源
快速上手：SWANLAB_GUIDE.md - 完整使用指南
集成细节：SWANLAB_INTEGRATION.md - 实现总结
官方文档：https://docs.swanlab.cn
测试脚本：scripts/test_swanlab.py
⚙️ 配置选项
如果需要临时禁用 SwanLab：
# configs/ieee33_config.yamllogging:  use_swanlab: false  # 改为 false
或自定义项目/实验名称：
logging:  use_swanlab: true  swanlab_project: "my-custom-project"  swanlab_experiment: "experiment-v2"
🎉 总结
✅ SwanLab 已完全集成
✅ 所有指标自动记录
✅ 超参数自动跟踪
✅ 向后兼容（可选安装）
✅ 完整文档和测试
现在你可以：
运行 pip install swanlab 安装
执行 python scripts/test_swanlab.py 测试
开始训练并在云端查看漂亮的实验仪表板！
祝实验顺利！ 🚀📊
如有任何问题，查看 SWANLAB_GUIDE.md 获取详细帮助。