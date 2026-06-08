# NPU 迁移报告

> 生成时间：{YYYY-MM-DD HH:MM}
> 原始项目：{项目名称/代码仓地址}
> 目标脚本：{训练/推理/评估脚本路径}
> NPU环境：{服务器IP/型号，或"无NPU环境，仅代码修改"}

---

## 一、环境搭建步骤

### 1.1 基础环境

| 项目 | 配置 |
|------|------|
| 服务器架构 | aarch64 / x86_64 |
| 操作系统 | Ubuntu 22.04 / openEuler 24.03 / ... |
| NPU型号 | Ascend 910B / 910C / 310P / ... |
| NPU数量 | {N} 卡 |
| Docker版本 | {version}（如使用Docker） |

### 1.2 版本信息

| 组件 | 版本 | 安装方式 |
|------|------|---------|
| Ascend HDK（驱动） | {version} | {安装命令} |
| CANN | {version} | {Docker镜像 / 手动安装} |
| Python | {version} | {conda / 系统自带} |
| PyTorch | {version} | {pip / whl包} |
| torch_npu | {version} | {pip / whl包} |
| 其他关键依赖 | {package==version} | {安装命令} |

### 1.3 环境搭建命令

```bash
# 1. Docker镜像拉取（如使用）
docker pull --platform={arch} swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:{tag}

# 2. 容器启动
docker run --name npu_dev -dit --privileged ... bash

# 3. 进入容器
docker exec -it npu_dev bash

# 4. 配置CANN环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 5. 安装PyTorch
pip install ...

# 6. 安装torch_npu
pip install ...

# 7. 安装项目依赖
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 8. 环境验证
python -c "import torch; import torch_npu; print(torch.npu.is_available())"
```

### 1.4 环境验证结果

```
[粘贴 torch.npu.is_available() 输出]
[粘贴 npu-smi info 输出]
[粘贴 import torch; print(torch.__version__) 输出]
```

---

## 二、代码迁移内容

### 2.1 修改清单

| # | 文件 | 行号 | 修改类型 | 修改前 | 修改后 | 原因 |
|---|------|------|---------|--------|--------|------|
| 1 | {path} | L{12} | 设备映射 | `.cuda()` | `.npu()` | NPU设备适配 |
| 2 | {path} | L{45} | 接口替换 | `torch.cuda.amp.autocast()` | `torch.npu.amp.autocast()` | 混合精度适配 |
| 3 | ... | ... | ... | ... | ... | ... |

### 2.2 接口等价替换详细说明

#### 替换 {N}：{原始接口} → {替换接口}

**数学定义**：
- 原始接口：{数学公式或行为描述}
- 替换接口：{数学公式或行为描述}

**等价性证明**：
{论证过程}

**数值验证**：
| 测试用例 | 原始输出 | NPU输出 | 绝对误差 | 相对误差 |
|----------|---------|---------|---------|---------|
| 用例1 | {value} | {value} | {value} | {value} |
| 用例2 | {value} | {value} | {value} | {value} |

**结论**：误差在可接受范围内 / 需要进一步处理

### 2.3 设备映射汇总

| CUDA（原始） | NPU（迁移后） | 涉及文件数 |
|-------------|-------------|----------|
| `.cuda()` | `.npu()` | {N} |
| `torch.cuda.*` | `torch.npu.*` | {N} |
| `backend="nccl"` | `backend="hccl"` | {N} |
| `DataParallel` | `DistributedDataParallel` | {N} |
| `torch.backends.cudnn.*` | 删除 | {N} |
| 其他 | ... | ... |

### 2.4 未修改项说明

| 文件/代码 | 不修改原因 |
|----------|-----------|
| {path} | {原因} |

---

## 三、验证结果

### 3.1 CPU基线结果

```
[粘贴CPU运行的关键输出]
[训练loss曲线截图路径或文字描述]
[推理输出样本]
```

### 3.2 NPU运行结果

```
[粘贴NPU运行的关键输出]
[训练loss曲线截图路径或文字描述]
[推理输出样本]
```

### 3.3 精度对比

| 指标 | CPU基线 | NPU结果 | 绝对误差 | 相对误差 | 是否通过 |
|------|---------|---------|---------|---------|---------|
| {指标1} | {value} | {value} | {value} | {value} | ✅/❌ |
| {指标2} | {value} | {value} | {value} | {value} | ✅/❌ |

**通过标准**：
- FP32：最大相对误差 < 1e-5
- FP16/BF16：最大相对误差 < 1e-2

**结论**：精度验证通过 / 精度存在偏差，说明如下：{...}

### 3.4 性能对比

| 指标 | CPU基线 | NPU结果 | 加速比 |
|------|---------|---------|--------|
| 单次推理耗时(ms) | {value} | {value} | {ratio}x |
| 单epoch训练耗时(s) | {value} | {value} | {ratio}x |
| 显存占用(MB) | {value} | {value} | — |

### 3.5 （可选）NPU亲和性调优

| 调优项 | 调优前耗时 | 调优后耗时 | 提升比例 |
|--------|----------|----------|---------|
| NpuFusedOptimizer | {value} | {value} | {ratio}% |
| npu_confusion_transpose | {value} | {value} | {ratio}% |

---

## 四、问题与解决方案

| 问题 | 现象 | 根因 | 解决方案 | 状态 |
|------|------|------|---------|------|
| {问题描述} | {报错信息} | {根因分析} | {解决步骤} | 已解决/已规避/待解决 |

---

## 五、附录

### A. 完整环境变量

```bash
export ASCEND_HOME=/usr/local/Ascend
export ASCEND_LOG_LEVEL=3
export ASCEND_RT_VISIBLE_DEVICES="0"
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:32
```

### B. 迁移后代码仓库信息

- 分支：{branch}
- Commit：{hash}
- 修改文件数：{N}

### C. 验证命令速查

```bash
# 环境验证
npu-smi info
python -c "import torch; import torch_npu; print(torch.npu.is_available())"

# 精度验证脚本
python verify_precision.py --cpu-baseline {path} --npu-output {path}

# 性能测试
python benchmark.py --device npu --iterations 100
```
