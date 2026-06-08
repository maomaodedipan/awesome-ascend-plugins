---
name: migration-ascend-torchnpu-skills-troubleshooting
description: Diagnoses and resolves common errors during Ascend NPU model migration. Invoke when encountering NPU runtime errors, precision anomalies, OOM issues, environment conflicts, or installation failures.
version: 1.0.0
---

# Skill: 昇腾NPU迁移故障诊断

你是一位专注于昇腾NPU迁移问题诊断的工程师。本Skill覆盖常见错误模式、诊断方法和解决方案。

## 一、环境类问题

### 1.1 `torch.npu.is_available()` 返回 False

| 可能原因 | 诊断命令 | 解决方案 |
|---------|---------|---------|
| CANN环境变量未设置 | `echo $ASCEND_HOME` | `source /usr/local/Ascend/ascend-toolkit/set_env.sh` |
| 驱动未安装或损坏 | `npu-smi info` | 重新安装驱动，检查 `/usr/local/Ascend/driver` |
| Docker未挂载驱动库 | `ls /usr/local/Ascend/driver/lib64/` | 启动时添加 `-v` 挂载参数 |
| torch_npu版本不匹配 | `pip show torch-npu` | 对照版本兼容性矩阵重新安装 |
| PyTorch与torch_npu版本不对应 | `python -c "import torch; print(torch.__version__)"` | 严格按兼容性矩阵匹配版本 |
| CANN版本与torch_npu不匹配 | `cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg` | 选择匹配的版本组合 |
| NPU设备权限不足 | `ls -la /dev/davinci*` | 将用户加入 `HwHiAiUser` 组 |
| 安装了CUDA版PyTorch | `pip show torch | grep cu` | x86_64：重装CPU版PyTorch |

### 1.2 `npu-smi info` 无输出

| 可能原因 | 诊断命令 | 解决方案 |
|---------|---------|---------|
| 驱动未安装 | `ls /usr/local/Ascend/driver` | 安装昇腾驱动 |
| 驱动安装后未重启 | — | `reboot` |
| 内核升级后驱动不可用 | `uname -r` | 锁定内核版本，重装驱动 |
| NPU卡硬件故障 | 检查服务器指示灯 | 联系硬件运维 |

### 1.3 CANN安装失败

| 可能原因 | 诊断命令 | 解决方案 |
|---------|---------|---------|
| 磁盘空间不足 | `df -h /usr/local/Ascend` | 确保 >9GB，清理后重装 |
| 系统依赖缺失 | `gcc --version; cmake --version` | 安装系统依赖包 |
| 先装了kernels再装toolkit | — | 卸载后按 toolkit→kernels 顺序重装 |
| 架构不匹配 | `uname -m` | 确认下载的安装包与架构一致 |

### 1.4 第三方库安装失败

| 可能原因 | 诊断命令 | 解决方案 |
|---------|---------|---------|
| ARM架构无预编译wheel | `uname -m` | 编译安装 |
| 国内网络不通 | `curl -I https://pypi.org` | `-i https://pypi.tuna.tsinghua.edu.cn/simple` 或阿里源 |
| 版本冲突 | `pip check` | 放宽版本约束，逐个安装并记录 |
| requirements含CUDA后缀 | `grep "+cu" requirements.txt` | 替换为纯PyTorch版本号 |

---

## 二、运行时错误

### 2.1 `RuntimeError: ... not implemented for NPU`

**根因**：代码调用了torch_npu尚未适配的接口。

**诊断**：
```python
try:
    output = model(input_npu)
except RuntimeError as e:
    if "not implemented for" in str(e) and "NPU" in str(e):
        import re
        match = re.search(r"'(.*?)'", str(e))
        if match:
            print(f"不支持的接口: {match.group(1)}")
```

**解决流程**：
1. 提取报错接口名称
2. 查阅PyTorch原生接口支持度文档
3. 查阅torch_npu扩展接口文档
4. 按优先级寻找替代方案：扩展接口 > 原生组合 > 自定义实现

### 2.2 `RuntimeError: CUDA error: ...`（残留CUDA引用）

**根因**：代码中仍有未替换的CUDA调用。

**诊断**：
```bash
grep -rn "\.cuda()" --include="*.py" .
grep -rn "torch\.cuda\." --include="*.py" .
grep -rn 'backend.*=.*"nccl"' --include="*.py" .
grep -rn "DataParallel" --include="*.py" .
grep -rn "torch\.backends\.cudnn" --include="*.py" .
```

**解决**：对照设备映射替换规则逐一处理。

### 2.3 OOM (Out of Memory)

| 可能原因 | 诊断 | 解决方案 |
|---------|------|---------|
| 数据并行batch过大 | 查看 `batch_size` | 减小batch_size |
| 显存泄漏 | `torch.npu.memory_allocated()` 趋势 | 检查 `del` 和 `empty_cache()` |
| 混合精度未开启 | 检查代码 | 开启AMP（NPU上不是可选项） |
| 模型过大 | 参数量估算 | 使用梯度累积、模型并行 |

```python
# 显存监控
import torch_npu
print(f"Allocated: {torch.npu.memory_allocated() / 1024**3:.2f} GB")
print(f"Cached: {torch.npu.memory_reserved() / 1024**3:.2f} GB")
torch.npu.empty_cache()  # 释放缓存
```

### 2.4 分布式训练报错

| 现象 | 可能原因 | 解决方案 |
|------|---------|---------|
| `backend 'nccl' not found` | 使用了nccl后端 | 改为 `backend='hccl'` |
| `DataParallel not supported` | 使用了DP模式 | 改为DDP |
| HCCL通信超时 | 网络或配置问题 | `export HCCL_CONNECT_TIMEOUT=1800` |
| 多卡初始化失败 | 缺少 `init_process_group` 的 `world_size`/`rank` | 确保参数正确传递 |

---

## 三、精度问题

### 3.1 NPU输出与CPU基线偏差过大

| 偏差量级 | 可能原因 | 诊断方法 | 解决方案 |
|---------|---------|---------|---------|
| 完全不一致 | 设备映射遗漏 | 对比代码diff | 补全缺失的映射 |
| 较大差异(>1e-2) | 混合精度类型错误 | 检查AMP配置 | FP16 vs BF16选择 |
| 中等差异(1e-3~1e-2) | 算子实现差异 | 逐层对比 | 对差异层进行特殊处理 |
| 微小差异(<1e-3) | 正常浮点计算误差 | — | 可接受，无需处理 |

### 3.2 逐层精度诊断工具

```python
def diagnose_precision(model_cpu, model_npu, sample_input, atol=1e-5):
    """逐层对比CPU/NPU输出，定位精度偏差来源"""
    model_cpu.eval()
    model_npu.eval()
    
    x_cpu = sample_input.clone()
    x_npu = sample_input.clone().npu()
    
    for name, layer_cpu in model_cpu.named_children():
        layer_npu = dict(model_npu.named_children())[name]
        
        x_cpu = layer_cpu(x_cpu)
        x_npu = layer_npu(x_npu)
        
        diff = abs(x_cpu - x_npu.cpu()).max().item()
        status = "✅" if diff < atol else "⚠️"
        print(f"{status} Layer {name}: max_diff={diff:.2e}")
```

### 3.3 训练Loss不收敛

| 可能原因 | 诊断方法 | 解决方案 |
|---------|---------|---------|
| 学习率需调整 | 对比CPU上loss曲线 | 适当调整lr |
| 混合精度导致梯度消失 | 关闭AMP测试 | 使用GradScaler动态调整scale |
| 随机种子未固定 | 对比多次运行 | 固定 `torch.manual_seed` 及相关种子 |
| 算子精度差异累积 | 逐层诊断 | 对关键层使用FP32 |

### 3.4 `torch.npu.amp.GradScaler` 问题

```python
# GradScaler在NPU上scale增长过快可能导致inf/nan
scaler = torch.npu.amp.GradScaler(
    init_scale=2**8,     # 降低初始scale（默认2**16）
    growth_interval=2000, # 增加增长间隔
)
```

---

## 四、数据加载问题

### 4.1 pin_memory 不可用

NPU不支持CUDA风格的 `pin_memory`：

```python
# ❌ 错误写法
dataloader = DataLoader(dataset, pin_memory=True)

# ✅ 正确写法
dataloader = DataLoader(dataset, pin_memory=False)
```

### 4.2 数据在设备间传输报错

```python
# 确保数据和模型在同一设备上
device = torch.device('npu:0')
model = model.to(device)
for data, target in dataloader:
    data, target = data.to(device), target.to(device)
    output = model(data)
```

---

## 五、模型保存与加载问题

### 5.1 保存NPU模型

```python
# ❌ 错误：直接保存NPU模型，无法在CPU/GPU加载
torch.save(model.state_dict(), 'model.pth')

# ✅ 正确：保存前转移到CPU
torch.save(model.cpu().state_dict(), 'model.pth')
model.npu()  # 恢复到NPU
```

### 5.2 加载GPU模型到NPU

```python
# 加载GPU训练的权重到NPU
checkpoint = torch.load('gpu_model.pth', map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])
model.npu()
```

---

## 六、快速诊断清单

遇到问题时按以下顺序排查：

```
1. [ ] npu-smi info 有输出？         → 否：驱动问题
2. [ ] torch.npu.is_available()?     → 否：CANN/NPU环境问题
3. [ ] import torch_npu 无报错？     → 否：版本匹配问题
4. [ ] 代码中无 .cuda() 残留？        → 否：设备映射未完成
5. [ ] 代码中无 torch.cuda.* 残留？   → 否：接口替换未完成
6. [ ] backend 是 hccl 不是 nccl？   → 否：分布式配置问题
7. [ ] 无 DataParallel 引用？         → 否：需改为DDP
8. [ ] 无 torch.backends.cudnn 引用？ → 否：需删除
9. [ ] 混合精度用的是 torch.npu.amp？ → 否：需替换
10.[ ] CPU基线已验证通过？            → 否：先解决CPU问题
```

<verification>
故障解决后确认：
- [ ] 错误日志中不再出现相同报错
- [ ] 相同输入反复运行结果一致
- [ ] 精度偏差在可接受范围内
- [ ] 已记录根因和解决方案到迁移报告
</verification>
