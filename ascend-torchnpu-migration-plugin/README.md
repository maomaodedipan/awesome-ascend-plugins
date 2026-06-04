# ascend-torchnpu-migration-plugin

将传统深度学习模型（PyTorch）通过 `torch_npu` 适配层迁移至华为昇腾NPU的Claude Code插件，覆盖环境搭建、代码迁移、精度验证、报告输出的完整流程。

> **注意：** 本插件仅覆盖 `torch_npu` 方式迁移。不包括ATC模型编译、ACL原生开发、MindSpore迁移等方案。

---

## 安装

```bash
git clone https://github.com/ascend-ai-coding/awesome-ascend-plugins.git
cd awesome-ascend-plugins
claude --plugin-dir ./ascend-torchnpu-migration-plugin
```

启动后运行 `/reload-plugins` 加载 skill。

---

## 使用方式

调用本插件的 skill 时，使用带命名空间前缀的 skill 名称发起迁移任务：

```
使用 /ascend-torchnpu-migration-plugin:migration-ascend-torchnpu 技能，在环境 IP：<IP> 账号：<user> 密码：<password> 的 NPU 服务器上，拉取 <代码仓URL>，创建新容器，跑通 <模型名称> 模型。
```

### 基础迁移

```
使用 /ascend-torchnpu-migration-plugin:migration-ascend-torchnpu 技能，在环境 IP：<IP> 账号：<user> 密码：<password> 的 NPU 服务器上，拉取 <代码仓URL>，创建新容器，跑通 <模型名称> 模型。
```

**示例（YOLO26）：**

```
代码仓：https://github.com/ultralytics/ultralytics
环境IP：175.100.100.60 密码：root/ Huawei@XXXX
使用 /ascend-torchnpu-migration-plugin:migration-ascend-torchnpu 技能，在给定环境上，新建容器，跑通 yolo26 模型
```

### 指定版本与目录

```
基于 <代码仓URL> 代码仓，跑通 <模型名称> 模型。
环境 IP：<IP> 密码：<password>，在机器上用 NPU 跑通代码仓给的第一个用例，使用 /ascend-torchnpu-migration-plugin:migration-ascend-torchnpu 技能。
要求使用新的容器和镜像，镜像采用最新的 CANN 镜像，PyTorch 采用 <版本>。
远端服务器的文件统一保存在：<目录>。
```

### Quick Start 代码迁移

```
环境 IP：<IP> 密码：<password>
使用 /ascend-torchnpu-migration-plugin:migration-ascend-torchnpu 技能，SSH 登录到上面的环境中，使用 docker 容器，加载 diffusers 库，迁移并在 NPU 上跑通以下代码：

```python
from diffusers import DiffusionPipeline
import torch
pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")
pipeline("An image of a squirrel in Picasso style").images[0]
```
```

### 裸机部署

```
在环境 IP：<IP> 密码：<password> 的 NPU 服务器上，裸机跑通 <代码仓URL> 模型，使用 /ascend-torchnpu-migration-plugin:migration-ascend-torchnpu 技能
```

### Prompt 关键要素

| 要素 | 必填 | 说明 |
|------|------|------|
| Skill 名称 | ✅ | `/ascend-torchnpu-migration-plugin:migration-ascend-torchnpu` |
| NPU服务器连接信息 | ✅ | IP、账号、密码 |
| 代码仓地址 | ✅ | GitHub / Gitee / ModelScope URL |
| 模型名称/目标 | ✅ | 要跑通的模型名称或用例 |
| 容器/裸机 | 建议 | 新建容器 or 裸机部署 |
| PyTorch 版本 | 可选 | 不指定则自动匹配兼容版本 |
| 文件保存目录 | 可选 | 避免污染服务器其他目录 |

---

## 迁移结果

迁移完成后，AI 将生成一份完整的迁移报告，包含：

- 环境搭建步骤（驱动/CANN/PyTorch/torch_npu 安装命令、环境变量配置）
- 代码迁移内容（每处修改的前后对比、位置、原因、接口等价替换说明）
- 验证结果（CPU/GPU 基线运行结果、NPU 运行结果、精度对比数据）

---

## 已验证模型

以下模型已通过本 Skill 在实际 NPU 环境（Ascend 910B3）上验证通过：

| 模型 | 平台 | 模型 | 方式 | 状态 |
|------|------|------|------|------|
| YOLO26 | OpenCode | DeepSeek V4 Pro | Docker容器 | ✅ |
| ResNet50 (mmpretrain) | OpenCode | DeepSeek V4 Pro | Docker容器 | ✅ |
| Whisper | OpenCode | DeepSeek V4 Pro | 裸机 | ✅ |
| CosyVoice3 | OpenCode | DeepSeek V4 Flash Free | Docker容器 | ✅ |
| Qwen-Image | OpenCode | DeepSeek V4 Flash | Docker容器 | ✅ |
| Qwen2-Audio-7B | OpenCode | DeepSeek V4 Flash Free | Docker容器 | ✅ |
| Stable Diffusion | Trae | GLM-5.1 | Docker容器 | ✅ |
| PP-OCRv4 (Paddle) | OpenCode | DeepSeek V4 Flash Free | Docker容器 | ✅ |

---

## 相关链接

- 昇腾官方文档：https://www.hiascend.com/document/
- torch_npu 官方仓库：https://gitcode.com/Ascend/pytorch
- 原始 Skill 代码：https://github.com/ascend-ai-coding/awesome-ascend-skills
