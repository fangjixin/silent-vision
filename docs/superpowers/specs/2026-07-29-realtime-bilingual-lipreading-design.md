# Silent Vision 实时中英双语唇语系统设计规格

**日期：** 2026-07-29  
**状态：** 已完成设计讨论，等待书面规格确认  
**目标：** 构建一个单用户、匿名会话的实时唇语识别系统。本机浏览器采集摄像头 JPEG 帧，通过 WebSocket 发送到远程 AMD ROCm 服务器；服务器完成嘴部检测、中英文唇语候选生成、MiniCPM-o 4.5 语义判断和无副作用 Agent 动作生成，再通过同一连接返回可分阶段验证的结果。

## 1. 范围

### 1.1 第一版包含

- 本机 Chrome、Safari 或 Edge 浏览器调用 `MediaDevices.getUserMedia()` 打开摄像头。
- 浏览器按目标 25 FPS 将 JPEG 二进制帧发送给远程 FastAPI 服务。
- 服务端使用 MediaPipe 检测面部关键点并裁剪 96×96 嘴部图像。
- 每个匿名 session 在内存中维护最近 75 个有效嘴部帧。
- 缓冲区首次达到 75 帧后触发推理，此后每增加 25 个有效帧再次触发滑动窗口推理。
- 英文 AV-HuBERT 与中文 CMLR VSR 分别输出候选文本和分数。
- MiniCPM-o 4.5 结合双引擎候选、分数和抽样嘴部帧，输出 `zh`、`en` 或 `unknown` 以及对应文本。
- Agent 将语义结果映射成 `respond`、`confirm` 或 `unknown` 结构化动作，只返回建议，不执行系统命令、外部 API、设备控制或文件操作。
- 浏览器展示摄像头、WebSocket、视觉检测、缓冲、两个唇语引擎、MiniCPM 和 Agent 的独立阶段状态。
- 单机 Radeon 7900 上只允许一个活动推理 session。
- 使用 `/workspace/persistence` 下的 NFS 目录持久化模型、缓存、诊断报告和服务日志。

### 1.2 第一版不包含

- 用户注册、登录、账户或跨连接身份。
- AV-HuBERT、CMLR VSR 或 MiniCPM 的训练、微调和数据集生产。
- 根据脸型、肤色、身份、姓名或其他外貌特征猜测语言。
- 音频采集、音频识别或音视频融合识别。
- 多用户 GPU 调度、分布式推理或水平扩容。
- Agent 的真实副作用。
- 对任意人物、光照、角度或口音承诺固定准确率。
- 将研究用途模型用于未获授权的商业部署。

## 2. 已确认的技术决策

1. 推理服务器使用 AMD Radeon 7900 和 ROCm；MiniCPM-o 4.5 的 AMD ROCm 适配已经由部署方完成，项目只验证其运行状况，不重新实现移植。
2. 采用模块化单体 FastAPI 服务，而不是多进程模型微服务。
3. 英文使用 AV-HuBERT video-only 检查点。
4. 中文使用 `mpc001/Visual_Speech_Recognition_for_Multiple_Languages` 提供的 CMLR visual-only 检查点及可选语言模型。
5. 中文与英文检查点分别推理；第一版不声称支持单句中的中英 code-switching。
6. MiniCPM-o 负责候选融合、语义理解和结构化推理，不把英文结果翻译成中文后冒充中文唇语结果。
7. 低置信度、候选冲突或缺少充分视觉证据时输出 `unknown`。
8. 目标节奏是 25 FPS、75 帧窗口、25 帧步长；所有值通过配置修改。
9. 第一版验收一个活动 session；第二个活动 session 得到 `SERVER_BUSY`。
10. 域名和 TLS 当前未配置。开发访问使用 SSH 本地端口转发；正式公网访问必须使用 HTTPS/WSS。

## 3. 部署拓扑

```text
本机浏览器
  摄像头 -> Canvas JPEG 编码 -> WebSocket 客户端
                                  |
                                  | SSH 隧道开发访问 / WSS 正式访问
                                  v
远程 AMD ROCm 服务器
  FastAPI Gateway -> Session Manager -> MediaPipe -> 75 帧窗口
                                                   |
                              +--------------------+--------------------+
                              |                                         |
                              v                                         v
                       AV-HuBERT (en)                             CMLR VSR (zh)
                              |                                         |
                              +--------------------+--------------------+
                                                   v
                                            MiniCPM-o 4.5
                                                   v
                                               Agent Policy
                                                   v
                                           WebSocket JSON 结果
```

FastAPI 同时提供前端静态文件。虽然 HTML 和 JavaScript 来自服务器，摄像头访问和 JPEG 编码仍在用户本机浏览器执行。开发阶段在本机执行：

```bash
ssh -L 8000:127.0.0.1:8000 user@rocm-server
```

随后打开 `http://localhost:8000`。服务端只监听 `127.0.0.1:8000`。正式部署时由反向代理提供域名、HTTPS 和 WSS。

## 4. 组件边界

### 4.1 Frontend

- `frontend/index.html`：页面结构、阶段状态区、候选结果区、最终结果区和运行指标区。
- `frontend/camera.js`：请求摄像头权限，将视频绘制到隐藏 canvas，按配置生成 JPEG，并控制开始和停止采集。
- `frontend/websocket.js`：创建匿名 session、连接 WebSocket、发送二进制 JPEG、处理服务端事件、心跳和指数退避重连。

前端不加载 AI 模型，不在浏览器推断语言，不保存视频。

### 4.2 FastAPI Gateway

- `backend/main.py`：应用工厂、lifespan、静态文件、路由注册、健康检查和模型容器初始化。
- `api/session.py`：`POST /api/sessions`，创建短生命周期匿名 UUID。
- `api/websocket.py`：`/ws/{session_id}`，处理 JSON 控制消息和二进制 JPEG，发送阶段事件。

WebSocket 协程只负责协议和编排。阻塞的图像与模型推理通过线程执行器运行，避免阻塞 ASGI 事件循环。

### 4.3 Session Manager

- `session/manager.py`：管理单个活动连接、内存帧、序号、最后活动时间、推理任务和最新待推理窗口。

每个 session 持有：

```text
session_id
connection_state
deque[mouth_frame, maxlen=75]
accepted_frame_count
last_inference_frame_count
active_inference_task
latest_pending_window
created_at
last_seen_at
```

如果新窗口到达时推理仍在运行，旧的待处理窗口被新窗口替换。系统不建立无限队列。

### 4.4 Vision

- `vision/face.py`：MediaPipe Face Landmarker/Face Mesh 适配器，返回归一化嘴部相关 landmarks 和检测状态。
- `vision/mouth.py`：计算带边距且限制在图像范围内的嘴部区域，裁剪、灰度化并缩放到 96×96。

检测失败的帧不会进入 75 帧缓冲。服务端周期性返回归一化 `mouthBox`，前端用它叠加可视化嘴部框。

### 4.5 Lip Reading

- `lip/base.py`：定义统一 `LipReader` 协议和 `LipReadingCandidate` 数据结构。
- `lip/avhubert.py`：封装 AV-HuBERT video-only 模型、输入布局、归一化、解码和英文候选分数。
- `lip/cmlr.py`：封装 CMLR visual-only 模型、中文字符解码和候选分数。
- `lip/inference.py`：按固定次序调用两个适配器，隔离单引擎故障并产生 `BilingualCandidates`。

统一候选包含：

```json
{
  "model": "cmlr",
  "language": "zh",
  "text": "请帮我打开灯",
  "confidence": 0.74,
  "rawScore": -0.83,
  "latencyMs": 186
}
```

不同模型的原始分数不可直接比较。各适配器负责将 token 平均对数概率或模型可用分数映射为 0–1 的本模型置信度，并保留 `rawScore` 供诊断。无法得到可靠分数时，`confidence` 为 `null`，MiniCPM 不得将其视为高置信度。

### 4.6 MiniCPM Semantic Interpreter

- `llm/minicpm.py`：封装已完成 ROCm 适配的 MiniCPM-o 4.5 加载和推理 API。

输入包括两路候选、置信度、窗口统计以及从 75 张嘴部图像均匀抽样的视觉帧。输出必须通过 Pydantic schema 校验：

```json
{
  "language": "zh",
  "text": "请帮我打开灯",
  "confidence": 0.78,
  "reason": "中文候选与视觉序列一致"
}
```

系统提示明确规定：

- 只根据嘴部运动和候选内容判断语言。
- 禁止根据外貌和身份判断语言。
- 禁止将翻译结果描述为唇语转写。
- 证据不足时返回 `unknown` 和空文本。
- 输出只允许一个 JSON 对象，不附加 Markdown。

### 4.7 Agent

- `agent/agent.py`：纯函数式策略层，将经过验证的语义结果映射为 `respond`、`confirm` 或 `unknown`。

```json
{
  "type": "agent.result",
  "action": "respond",
  "language": "zh",
  "text": "请帮我打开灯",
  "arguments": {},
  "requiresConfirmation": false
}
```

第一版 Agent 没有工具注册表，也没有操作系统、网络、文件或设备执行能力。

## 5. 生命周期与数据流

1. 页面加载后调用 `POST /api/sessions`。
2. 服务返回 UUID 和过期时间；前端只把 UUID 写入当前标签页的 `sessionStorage`。
3. WebSocket 连接成功后，服务返回 `session.ready` 和当前运行参数。
4. 用户点击开始，前端发送 `stream.start`，随后发送 JPEG 二进制消息。
5. 服务校验消息大小、JPEG 格式和图像尺寸，记录服务端接收序号和时间。
6. Vision Pipeline 返回人脸状态、嘴部框和 96×96 嘴部图像。
7. 有效嘴部图像进入长度为 75 的 deque；浏览器收到 `vision.result` 和 `buffer.progress`。
8. 第 75 个有效帧产生第一个不可变窗口快照；此后每 25 个有效帧产生新快照。
9. GPU 锁串行保护双 VSR 和 MiniCPM 推理。推理在后台任务中运行，WebSocket 仍可接收帧和心跳。
10. 服务依次发送 `lip.candidates`、`semantic.result` 和 `agent.result`。
11. 用户点击停止或连接断开时，服务取消未开始任务，释放窗口引用并移除 session。

## 6. WebSocket 协议

### 6.1 客户端到服务器

- 二进制消息：一张完整 JPEG。服务器接收时间作为该帧时间戳。
- `stream.start`：开始新的内存缓冲并返回生效参数。
- `stream.stop`：停止接收推理帧并清空缓冲。
- `ping`：应用层心跳。

### 6.2 服务器到客户端

所有 JSON 消息都包含 `type`、`sessionId`、`timestamp`。

- `session.ready`
- `stream.started`
- `frame.accepted`
- `vision.result`
- `buffer.progress`
- `inference.started`
- `lip.candidates`
- `semantic.result`
- `agent.result`
- `metrics.update`
- `error`
- `pong`

错误格式为：

```json
{
  "type": "error",
  "sessionId": "uuid",
  "timestamp": "2026-07-29T12:00:00Z",
  "stage": "vision",
  "code": "FACE_NOT_FOUND",
  "message": "当前画面未检测到清晰人脸",
  "recoverable": true
}
```

## 7. 错误处理与降级

- JPEG 损坏、超限或尺寸非法：丢弃单帧并返回可恢复错误。
- 没有脸、检测到多张脸或嘴部区域无效：不缓冲该帧，提示用户调整位置。
- 实际 FPS 偏低：继续收集有效帧并报告实际 FPS，不用复制帧伪造输入。
- AV-HuBERT 单独失败：保留中文候选，标记 `degradedModels: ["avhubert"]`。
- CMLR 单独失败：保留英文候选，标记 `degradedModels: ["cmlr"]`。
- 两个 VSR 引擎均失败：不调用 MiniCPM，返回 `LIP_MODELS_FAILED`。
- MiniCPM 超时、异常或非法 JSON：返回两路原始候选，Agent 输出 `unknown`。
- 候选冲突且语义结果低于阈值：输出 `unknown`，不选择看似通顺的文本。
- GPU OOM：记录显存诊断，释放窗口引用，调用 ROCm PyTorch 对应的缓存清理接口一次，并返回 `GPU_OUT_OF_MEMORY`；当前窗口不自动重试。
- WebSocket 断开：立即清除 session 和帧。前端指数退避重连后获取新 session，不恢复旧视频缓冲。
- 第二个活动推理连接：返回 `SERVER_BUSY` 并关闭连接。

## 8. 健康检查与可观测性

FastAPI lifespan 依次初始化 MediaPipe、AV-HuBERT、CMLR VSR 和 MiniCPM。所有必需组件就绪后服务才通过 readiness。

- `GET /health/live`：进程和事件循环可响应。
- `GET /health/ready`：列出每个模型的加载状态、设备和失败原因，不泄露文件系统凭证。
- `GET /health/models`：仅在开发配置开启，返回模型名称、检查点摘要、dtype 和最近一次推理耗时。

指标至少包括接收 FPS、有效嘴部帧比例、缓冲深度、丢弃帧数、各阶段耗时、端到端耗时、GPU 设备名称和显存使用量。默认日志不记录 JPEG、嘴部图像或完整识别文本。

## 9. 持久化与隐私

NFS 根目录固定为：

```text
/workspace/persistence/silent-vision/
├── models/
│   ├── avhubert/
│   ├── cmlr/
│   └── minicpm-o-4_5/
├── cache/
│   ├── huggingface/
│   └── torch/
├── reports/
│   ├── benchmarks/
│   └── diagnostics/
└── logs/
```

Docker Compose 将该目录绑定到容器内相同绝对路径。启动时检查目录存在、可读写和模型文件可读。模型权重不进入 Docker 镜像和 Git。

摄像头 JPEG、96×96 嘴部帧、session 状态、MiniCPM 临时视觉输入和完整识别文本不持久化。必须生成临时媒体文件时使用 `/tmp/silent-vision/{session_id}`，在窗口结束、断开连接及正常停机时清除。

## 10. 安全约束

- UUID 由服务端使用密码学安全随机源生成。
- session 创建后短时间内未连接则过期；断开后立即失效。
- WebSocket 校验 `Origin`；允许列表来自配置。
- 限制 JPEG 字节数、最大分辨率、有效帧率和控制消息大小。
- 模型路径只能来自服务端环境变量，客户端不能提交路径、prompt 或模型参数。
- Agent 输出通过枚举和 Pydantic schema 限制。
- 开发服务只监听 loopback，并通过 SSH 隧道访问。
- 正式远程访问必须使用 HTTPS/WSS；匿名模式仍需反向代理连接数和速率限制。

## 11. 配置

核心环境变量：

```text
PERSISTENCE_ROOT=/workspace/persistence/silent-vision
AVHUBERT_CHECKPOINT=/workspace/persistence/silent-vision/models/avhubert/model.pt
CMLR_CHECKPOINT=/workspace/persistence/silent-vision/models/cmlr/model.pth
CMLR_LANGUAGE_MODEL=/workspace/persistence/silent-vision/models/cmlr/language-model.pth
MINICPM_MODEL_PATH=/workspace/persistence/silent-vision/models/minicpm-o-4_5
CAPTURE_FPS=25
WINDOW_FRAMES=75
INFERENCE_STRIDE=25
MOUTH_SIZE=96
MAX_JPEG_BYTES=1048576
MAX_FRAME_WIDTH=1920
MAX_FRAME_HEIGHT=1080
MODEL_CONFIDENCE_THRESHOLD=0.55
ALLOWED_ORIGINS=http://localhost:8000
LOG_TRANSCRIPTS=false
```

前端默认从 `window.location` 推导 HTTP 和 WebSocket 地址；可选运行时配置只用于跨源开发，不把服务器地址写死在 JavaScript 中。

## 12. 测试策略

### 12.1 默认自动测试

默认测试使用假模型，不要求 ROCm 或下载权重：

- session 创建、过期、连接冲突和断开清理；
- JPEG 校验与解码；
- landmarks 到嘴部框、边界裁剪和 96×96 输出；
- 75 帧 deque、25 帧步长及最新窗口替换；
- 候选类型和双引擎单边降级；
- MiniCPM 合法 JSON、非法 JSON、超时和低置信度；
- Agent 三种动作映射；
- WebSocket 事件 schema、顺序和错误码；
- 假推理模式下的完整 FastAPI WebSocket 流程。

### 12.2 浏览器端到端测试

浏览器测试启动假模型服务并使用虚拟摄像头媒体：

- 摄像头授权与预览；
- WebSocket 建连和重连；
- 二进制 JPEG 上传；
- 嘴部框、缓冲进度和阶段状态渲染；
- 中英文候选、语义结果和 Agent 动作显示；
- 停止采集后不再发送帧。

### 12.3 ROCm 和模型测试

硬件测试使用 `rocm` 与 `model_integration` pytest 标记，不进入无 GPU 的默认 CI：

1. PyTorch 能发现 Radeon 7900，并记录 ROCm 版本、设备名和显存。
2. MediaPipe 对真实摄像头帧持续产生有效嘴部区域。
3. 许可允许的英文样本经 AV-HuBERT 得到非空英文候选。
4. 许可允许的中文样本经 CMLR VSR 得到非空中文候选。
5. MiniCPM 输出通过 schema 校验，语言为 `zh`、`en` 或 `unknown`。
6. Agent 只产生允许的枚举动作。
7. 完整链路首次在 75 个有效帧后启动，此后按 25 个有效帧步长启动。
8. 连续运行 10 分钟，无持续显存增长、无无限任务队列、WebSocket 保持可响应。

测试媒体不提交 Git。仓库提供 manifest 格式记录本地测试文件路径、预期语言和授权来源。

## 13. 分阶段交付与验收

### Phase 0：项目骨架与假模型

服务、配置、健康检查、统一 schema 和假模型全部运行；默认测试不需要 GPU。

### Phase 1：摄像头与 WebSocket

本机浏览器通过 SSH 隧道打开页面、获得摄像头权限并持续上传 JPEG；页面显示连接状态、发送 FPS 和服务端接收计数。

### Phase 2：MediaPipe 与嘴部缓冲

页面叠加服务端返回的嘴部框，显示有效帧比例和 0–75 缓冲进度；无脸帧不进入缓冲。

### Phase 3：ROCm 双语唇语识别

AV-HuBERT 英文样本与 CMLR 中文样本分别产生候选；单引擎故障时另一引擎仍可返回；GPU 指标和阶段耗时可见。

### Phase 4：MiniCPM 与 Agent

MiniCPM 从候选和视觉帧生成严格语义结果；Agent 输出允许的无副作用动作；完整链路事件按顺序显示。

### Phase 5：容器与运行验收

Docker Compose 在 ROCm 服务器加载 NFS 模型，SSH 隧道访问成功，完成 10 分钟单 session 稳定性测试并生成诊断报告。

## 14. 模型与许可证

- AV-HuBERT：使用项目批准的 video-only 检查点，部署时记录来源 URL、文件哈希和许可证。
- CMLR VSR：使用 `mpc001/Visual_Speech_Recognition_for_Multiple_Languages` 的 CMLR visual-only 模型。上游说明代码仅用于比较或基准等非商业用途；部署前必须确认实际用途符合授权。
- MiniCPM-o 4.5：使用部署方已完成 AMD ROCm 适配的本地模型，记录版本、权重来源、dtype、量化方式和文件哈希。

模型引用：

- AV-HuBERT: <https://github.com/facebookresearch/av_hubert>
- Multilingual VSR/CMLR: <https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages>
- MiniCPM-o 4.5: <https://huggingface.co/openbmb/MiniCPM-o-4_5>
- Browser camera secure-context requirement: <https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia>

## 15. 成功标准

系统成功的定义是：在一台 Radeon 7900 ROCm 服务器和一台带摄像头的本机浏览器之间，能够通过匿名单 session 连续传输帧；每个流水线阶段都能单独观察成功、降级或失败；中英文专用模型各自生成候选；MiniCPM 在不依据外貌猜测语言的前提下选择 `zh`、`en` 或拒识；Agent 返回 schema 合法且无副作用的动作；断开连接后不保留视频帧或 session 数据。
