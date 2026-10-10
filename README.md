# 知识库问答系统

基于 LangGraph 的知识库问答系统：文档导入（MinerU 解析 → 切片 → BGE-M3 向量化 → Milvus 入库）
与检索问答（意图识别 → HyDE → 混合检索 → RRF 融合 → DashScope 重排 → LLM 流式作答）。

检索链路之外还配了三套生产级保障，用法见 [检索体系与运维能力](#检索体系与运维能力)：
**召回评测体系**（LLM 反向出题的评测集 + 召回率跌幅超 2% 就拦住发布的门禁）、
**索引蓝绿重建**（显式 HNSW 调参 + 别名原子切换 + 新旧索引一致性校验）、
**监控告警与线上抽检**（Prometheus/Grafana + 慢查询日志 + 每 6 小时自动巡检索引健康度）。

| 组件 | 形态 | 说明 |
| --- | --- | --- |
| import-api | 容器（本仓库） | FastAPI，端口 8000，文件上传 + 导入流水线 + 索引/巡检运维接口 |
| query-api | 容器（本仓库） | FastAPI，端口 8001，检索问答 + SSE 流式 |
| scheduler | 容器（本仓库） | 索引巡检常驻进程；默认不加载 BGE-M3，内存占用远小于两个 API（开了自动重建才会用到） |
| Milvus / etcd / MinIO(内部) | 容器 | `deploy/milvus/docker-compose.yml`，端口 19530 |
| MongoDB | 容器 | 会话历史、查询日志、索引注册表、评测与巡检记录，端口不对外 |
| MinIO(业务) | 容器 | 图片对象存储，宿主机 9010 / 控制台 9011 |
| Prometheus / Alertmanager / Grafana | 容器 | `deploy/monitoring/`，端口 9090 / 9093 / 3000 |
| BGE-M3 | 容器内进程加载 | 本地模型，bind-mount 进容器，约 2.3GB |
| LLM / 重排 / MinerU | 远端 API | DeepSeek、DashScope `qwen3-rerank`、`mineru.net`，**不容器化** |

两个 API 服务与 scheduler 共用同一个镜像，仅 `command` 不同。

---

## Docker 部署

### 0. 前置条件

- Docker Engine 24+，Compose **v2.20+**（根 compose 用 `include` 复用 Milvus 栈）。实测通过：Docker 29.7.2 + Compose v5.5.1。
- 磁盘：CPU 镜像约 2GB / GPU 镜像约 7GB，另加 Milvus、Mongo、MinIO 数据卷。
- 内存：≥8GB。BGE-M3 会在 import-api 与 query-api 中**各加载一份**，CPU 部署按 ×2 估算。
- 目标机能访问公网 API：`api.deepseek.com`（或 `.env` 里配的 `OPENAI_API_BASE`）、`dashscope.aliyuncs.com`、`mineru.net`。
- GPU 部署额外需要：NVIDIA 驱动 + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。

### 1. 准备 BGE-M3 模型（一次性）

容器不下载模型（`MODELSCOPE_OFFLINE=1`），模型目录由宿主机 bind-mount 进去。

默认取仓库下的 `./models/BAAI/bge-m3`（即 modelscope 的缓存布局），目录内需含 `config.json`、
权重文件（`model.safetensors` 或 `pytorch_model.bin`）、`tokenizer.json`、`sentencepiece.bpe.model`。

**已有缓存的机器**：在 `.env` 里追加一行指向现成目录，免拷贝。

```dotenv
BGE_M3_DIR=D:/ai_models/modelscope_cache/models/BAAI/bge-m3
```

> 仓库自带 `tool/download_bgem3.py` 就是这个用途，但里面 `cache_dir` 写死为 Windows 路径。

**新机器**：下载到仓库 `./models` 下，正好落在默认位置。

```bash
pip install modelscope
python -c "from modelscope.hub.snapshot_download import snapshot_download; print(snapshot_download('BAAI/bge-m3', cache_dir='./models'))"
```

境内网络下载失败时改用 HF 镜像（必须禁 Xet，否则大文件 401）：

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
  python -c "from huggingface_hub import snapshot_download; print(snapshot_download('BAAI/bge-m3', local_dir='./models/BAAI/bge-m3'))"
```

### 2. 配置 `.env`

`.env` 已随仓库提供，容器内这些键会被 compose 的 `environment` **覆盖**，无需手改：

`MILVUS_URL` `MONGO_URL` `MINIO_ENDPOINT` `DATA_BASED_ROOT_DIR` `BGE_M3_PATH`
`BGE_DEVICE` `BGE_FP16` `MODELSCOPE_CACHE` `HF_HOME` `MODELSCOPE_OFFLINE` `HF_HUB_OFFLINE`

必须确认的键：

```dotenv
OPENAI_API_KEY=sk-...        # LLM + DashScope 重排共用
OPENAI_API_BASE=https://api.deepseek.com
MINERU_API_TOKEN=...         # MinerU 文档解析
TEXT_RERANK_MODEL=qwen3-rerank
```

> ⚠️ `.env` 含真实密钥。它被 `.dockerignore` 排除，**不会**进镜像层；容器运行时由 compose 的
> `env_file` 注入。不要把镜像推到公共仓库。

检索与运维相关的可选键（意图识别、索引参数、监控、巡检）都有代码内默认值，
不改也能跑；要调时见 [新增配置项](#7-新增配置项)。

**图片 URL 的特殊处理**：`MINIO_ENDPOINT` 同时用于「容器连 MinIO」和「浏览器加载图片」，
两个视角的地址不同，按访问方式二选一：

- **只在部署机本机浏览器访问**：保持默认 `host.docker.internal:9010`。
  Windows 宿主机需在 `C:\Windows\System32\drivers\etc\hosts` 加一行 `127.0.0.1 host.docker.internal`
  （Linux / macOS 的 Docker Desktop 默认可解析）。
- **从别的机器访问**：在 `.env` 追加 `APP_MINIO_ENDPOINT=<部署机局域网IP>:9010`。
  变量名故意与 `MINIO_ENDPOINT` 不同——compose 会拿根目录 `.env` 做 `${}` 插值，同名会被 `.env` 里的 `localhost` 值顶掉。

不改这项的后果：导入与问答流程正常，但答案里的图片在浏览器中加载失败。

### 3. 启动

```bash
docker compose up -d --build
docker compose ps          # 11 个服务均应为 healthy / running
```

> `scheduler` 是常驻循环进程，没配 healthcheck，状态显示 `running` 即为正常；
> 不想要自动巡检时 `docker compose stop scheduler` 即可，其余服务不受影响。

境内网络请先按 [§7](#7-境内网络镜像加速) 预热镜像，**逐个 pull 而非并发**
（并发拉大镜像时镜像站把 blob 302 到 R2，易瞬时 `i/o timeout`，单独重试即可成功）。

### 4. 验证

```bash
curl http://localhost:8000/health    # {"ok":true}
curl http://localhost:8001/health    # {"ok":true}
```

- 导入页：`http://<部署机IP>:8000/import.html`（接口文档 `/docs`）
- 问答页：`http://<部署机IP>:8001/chat.html`（接口文档 `/docs`）
- 监控看板：`http://<部署机IP>:3000`（默认 `admin/admin`，已开匿名只读，生产请改密码并关匿名）
- 索引状态：`curl http://localhost:8000/admin/index/status`（别名指向、各版本状态、重建进度）
- 原始指标：`curl http://localhost:8001/metrics`（Prometheus 抓取的就是这个端点）

BGE-M3 是**首次请求时**才加载（懒加载单例），所以第一次上传/提问会明显卡顿几十秒，属正常现象。
用 `docker compose logs -f query-api` 可看到加载日志。

### 5. 使用 GPU

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

叠加文件做两件事：给两个 API 服务加 `gpus: all`，并把 `BGE_DEVICE` 改为 `cuda:0`、`BGE_FP16` 改为 `True`。

GPU 版 torch 体积大得多，构建时需切换 torch 源（默认装的是 CPU 版 `torch==2.14.0+cpu`）：

```bash
TORCH_SPEC=torch==2.14.0 TORCH_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple \
  docker compose -f docker-compose.yml -f docker-compose.gpu.yml build
```

验证容器能看到显卡：

```bash
docker compose exec query-api python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 6. 端口一览

| 端口 | 服务 |
| --- | --- |
| 8000 | import-api（导入 + `import.html` + `/admin/*` 运维接口） |
| 8001 | query-api（问答 + `chat.html`） |
| 19530 / 9091 | Milvus / Milvus 健康检查与自身指标 |
| 9010 / 9011 | 业务 MinIO API / 控制台 |
| 9090 | Prometheus（指标与告警规则状态） |
| 9093 | Alertmanager（告警通知） |
| 3000 | Grafana（看板 `知识库问答系统 · 检索体系总览`） |

MongoDB 与 Milvus 内部 MinIO 不对外发布端口，仅容器网络内可达。
监控栈整体可摘：删掉根 `docker-compose.yml` 里 `include` 的
`deploy/monitoring/docker-compose.monitoring.yml` 那一行即可下线，业务服务无感知。

### 7. 境内网络：镜像加速

仓库里的 compose 与 Dockerfile 一律写**官方镜像名**，境内机器用「先经镜像站拉取、再打回原标签」
的方式预热，无需修改任何文件：

```bash
M=docker.m.daocloud.io
docker pull $M/library/python:3.12-slim            && docker tag $M/library/python:3.12-slim python:3.12-slim
docker pull $M/library/mongo:7                     && docker tag $M/library/mongo:7 mongo:7
docker pull $M/minio/minio:RELEASE.2023-03-20T20-16-18Z && docker tag $M/minio/minio:RELEASE.2023-03-20T20-16-18Z minio/minio:RELEASE.2023-03-20T20-16-18Z
docker pull $M/milvusdb/milvus:v2.5.5              && docker tag $M/milvusdb/milvus:v2.5.5 milvusdb/milvus:v2.5.5
docker pull quay.m.daocloud.io/coreos/etcd:v3.5.18 && docker tag quay.m.daocloud.io/coreos/etcd:v3.5.18 quay.io/coreos/etcd:v3.5.18
```

可选镜像站：`docker.1panel.live`、`hub.rat.dev`。

pip 源默认已指向清华（`PIP_INDEX_URL` 可覆盖）。若 `download.pytorch.org` 不可达，
用 `TORCH_INDEX` 指向可达的 PyTorch wheel 源。

### 8. 日常运维

```bash
git pull && docker compose up -d --build     # 更新代码并重建
docker compose logs -f import-api query-api  # 看日志
docker compose logs -f scheduler             # 看索引巡检日志
docker compose restart query-api             # 重启单个服务
docker compose down                          # 停止（保留数据卷）
docker compose down -v                       # 停止并删除全部数据（向量库、历史、图片全丢）

# 改完告警规则/看板后热加载（compose 已开 --web.enable-lifecycle）
docker compose exec prometheus wget -qO- --post-data='' http://localhost:9090/-/reload
curl -s http://localhost:9090/api/v1/rules | python -m json.tool   # 看规则是否加载、有无 firing
```

数据都在命名卷里，重建镜像不影响：`milvus_data`（向量）、`mongo_data`（会话历史、查询日志、
索引注册表、评测与巡检记录）、`app_minio_data`（图片）、`uploads`（上传原件与解析中间产物）、
`eval_results`（评测结果 JSON，两个 API 与 scheduler 共享，`kb_eval_recall` 指标靠它）、
`etcd_data`、`minio_data`、`prometheus_data`、`alertmanager_data`、`grafana_data`。

备份向量库请对 Milvus 做 collection 导出，或直接备份卷：
`docker run --rm -v zhangguithinktank_milvus_data:/d -v ${PWD}:/bk alpine tar czf /bk/milvus_data.tgz -C /d .`

### 9. 排错

| 现象 | 原因与处理 |
| --- | --- |
| `import-api` 起不来，日志报 Milvus/Mongo 连接失败 | 依赖服务未 healthy。compose 已配 `depends_on: service_healthy`，若手动 `up` 单个服务请先把依赖起全 |
| 日志出现 `Minio init failed` 且后续图片全部 404 | `utils/minio_utils.py` 在导入时建桶，失败后 `minio_client` 永久为 `None`（不重试）。重启该服务即可 |
| 问答能出文字但图片裂开 | 见 §2 图片 URL 说明，设置 `APP_MINIO_ENDPOINT` 或宿主机 hosts |
| 构建时 `torch` 拉取极慢或超时 | 默认从 `download.pytorch.org` 取 CPU 轮子，改用 `TORCH_INDEX` |
| 容器内尝试联网下模型并卡住 | 检查 `BGE_M3_DIR` 挂载是否正确：`docker compose exec query-api ls /models/bge-m3` |
| GPU 机器上 `torch.cuda.is_available()` 为 False | 构建时没切 `TORCH_SPEC`（装成了 CPU 版），或宿主机缺 nvidia-container-toolkit |
| Grafana 看板没有召回率曲线 | `eval/results/` 里还没有结果文件。先跑一次 `python -m eval.run_eval --limit 5 --tag smoke`；容器部署要确认 `eval_results` 卷被两个 API 与 scheduler 共享 |
| `/metrics` 里缺自定义 `kb_*` 指标 | `.env` 里 `METRICS_ENABLED=false`，或指标采集器读不到 Milvus/Mongo（缺哪项会在 `python -m utils.metrics_utils` 的自测里列出来） |
| Alertmanager 日志报推送失败 | `ALERT_WEBHOOK_URL` 默认值故意指向一个没人监听的端口，这样推不出去会报错而不是静默丢告警。钉钉/飞书需要一个签名转发网关，把它填进去 |
| 巡检每轮都只出 `alert_only` | 报告里的 `[!]` 行会写明原因，通常是 Milvus/Mongo 连不上——采集不到信号时**不会**报健康 |
| 重建报「已有在建版本」 | 上次重建异常退出留下了 building 记录。确认新集合确实没用后 `python -m processor.index_manager.rebuild --mark-failed --version N`，或加 `--drop-on-failure` 清掉 |

---

## 不用 Docker 的本地跑法

```bash
uv sync                                              # 或 pip install -r requirements.txt
uvicorn web.api.import_service:app --port 8000       # 另开一个终端
uvicorn web.api.query_service:app --port 8001
python -m processor.index_manager.scheduler --loop   # 可选：索引巡检常驻
```

需自行准备 Milvus、MongoDB、MinIO，并保证 `.env` 里的地址指向它们。
`eval/` 与 `processor/index_manager/` 下的命令都是本地直接跑的 CLI，见下一节。

---

## 检索体系与运维能力

改造分五期落地，每期都能单独用、单独演示、单独回退。下面按「先接哪个」的顺序说明。

### 0. 链路全景

```
用户问题
  └─ 意图识别（三层漏斗：关键词/正则 → BGE-M3 原型相似度 → LLM 兜底）
       ├─ OUT_OF_SCOPE ─────────────────────────────► 模板直答，不检索
       └─ 其它意图 → 产出 strategy（融合权重 / HNSW ef / 各路开关 / 截断条数）
            ├─ dense + sparse 混合检索（Milvus WeightedRanker，权重来自 strategy）
            ├─ HyDE 假设文档检索（strategy.enable_hyde）
            └─ 联网搜索（strategy.enable_web_search）
                 └─ RRF 融合 → DashScope 重排 + 断崖截断 → LLM 流式作答

每次问答都会留下：Prometheus 指标（节点级 / 单路级耗时与命中）+ Mongo query_log（含慢查询标记）

旁路（都不在请求路径上，挂了不影响问答）：
  eval/                            离线评测集 + 召回率门禁        ← 改检索参数前后各跑一次
  processor/index_manager/         参数网格 / 蓝绿重建 / 一致性校验 ← 换索引参数时用
  processor/index_manager/scheduler 每 6h 巡检 → 判定 → 可选触发重建
  deploy/monitoring/               Prometheus + Alertmanager + Grafana
```

### 1. 意图识别三层漏斗

**为什么要分层**：精确查询（问型号参数）该重稀疏词面匹配，语义查询（问怎么修）该重稠密向量，
一套权重必然让其中一类吃亏。但每个问题都调 LLM 判意图又太贵太慢，所以做成漏斗：便宜的先过，过不了的才往下走。

| 层 | 手段 | 成本 | 什么情况走到这 |
| --- | --- | --- | --- |
| L1 | 关键词子串 + 型号正则（如 `HAK180`） | 0 | 所有问题；命中即定类，置信度 1.0 |
| L2 | BGE-M3 把问题与每类 4~5 条「原型句」算余弦 | 一次本地 embedding | L1 没命中 |
| L3 | LLM 输出 JSON 定类（带 L2 的倾向作提示） | 一次 LLM 调用 | L2 最高分 < 0.75，或与次高分差 < 0.05 |

L1 的优先级是刻意的：**超范围（且不含型号）→ 歧义 → 语义 → 精确**。
L3 调用失败时回退 `SEMANTIC` 而不是 `OUT_OF_SCOPE`——宁可多检索一次，也不能把正经问题挡在门外。

L2 算出的稠密/稀疏向量直接存进 state 给检索节点复用，**整条链路只 embedding 一次**。

策略表（`config/intent_config.py`，改这里就改了检索行为）：

| 意图 | dense:sparse | HNSW ef | HyDE | 联网 | 单路/融合条数 | 典型问题 |
| --- | --- | --- | --- | --- | --- | --- |
| EXACT | 0.3 : 0.7 | 64 | ✗ | ✗ | 10 / 5 | HAK180 的功率是多少 |
| SEMANTIC | 0.8 : 0.2 | 128 | ✓ | ✓ | 10 / 5 | 转印颜色偏浅怎么办 |
| AMBIGUOUS | 0.6 : 0.4 | 128 | ✓ | ✓ | 20 / 10 | HAK180 和 HAK200 有什么区别 |
| OUT_OF_SCOPE | — | — | — | — | 不检索，模板直答 | 你好 / 讲个笑话 |
| DEFAULT | 0.8 : 0.2 | 索引默认 | ✓ | ✓ | 10 / 5 | 关掉意图识别时的行为 |

HyDE 与联网的开关**不改图的边**：节点自己读 strategy 早退返回空结果，所以图拓扑与改造前一致。

```bash
# 自测四类问题，打印命中的层级/类别/置信度。需要本地 BGE-M3（L2）与可达的 LLM API（L3）
python -m processor.query_processor.nodes.node_intent_recognition
```

`POST /query` 支持直接指定意图跳过识别：`{"query": "...", "intent": "EXACT"}`；
传 `top_k` 会同时覆盖融合条数与重排上限。

### 2. 召回评测体系与发布门禁

**评测集怎么来**：人工标注太贵，改成让 LLM 从语料反向出题——抽一段切片，让它出「答案就在这段里」的问题。
5 个场景 × 每场景 200 条，再抽 10% 人工核对。

| 场景 | 目录名 | 期望意图 |
| --- | --- | --- |
| 参数查询 | `param_lookup` | EXACT |
| 操作步骤 | `operation_guide` | SEMANTIC |
| 故障排查 | `troubleshoot` | SEMANTIC |
| 对比歧义 | `comparison` | AMBIGUOUS |
| 长文综合 | `long_doc_qa` | SEMANTIC |

> **标准答案存的是业务三元组 `(file_title, title, part)` + `content_hash`，不是 `chunk_id`。**
> `chunk_id` 是 Milvus 自增 id，索引重建一次就全变；三元组在业务上唯一定位一段资料，
> 所以评测集与历史查询日志在重建之后依然可用。

指标：`recall@k`、`precision@k`、`MRR`、`NDCG@k`，按场景聚合并给总均值。

```bash
python -m eval.generate_dataset --dry-run                            # 只验抽样链路，不调 LLM
python -m eval.generate_dataset --scenario param_lookup --limit 20   # 先小量试跑
python -m eval.generate_dataset                                      # 补齐到配额（断点续跑）
python -m eval.sample_review export --ratio 0.1                      # 导出 10% 待人工审
python -m eval.sample_review apply --review eval/review/xxx.json     # 回写结论，rejected 不计入评测
python -m eval.sample_review status                                  # 看审核进度
python -m eval.run_eval --tag baseline                               # → eval/results/{时间戳}_baseline.json
python -m eval.gate --baseline <基线文件> --candidate <候选文件>
```

**门禁规则**：逐个场景 + 两个总均值（样本加权 / 场景等权）分别对比 `recall@10`，
**最差的一项**比基线低 0.02（2 个百分点）以上 → 打印逐项对比表并 `exit 1`。
蓝绿重建默认强制调用它，跌了就拦下不切换。阈值在 `eval/config.py` 的 `gate_recall_drop`，
临时改用 `--max-drop`；两份结果口径不可比（评测模式、数据集版本、样本量差太多）时 `exit 2` 而不是放行。

`run_eval` 常用开关：`--limit N`（冒烟）、`--no-rerank`（只测召回层，看 RRF 输出）、
`--use-intent`（走意图漏斗）、`--ef-search / --dense-weight / --sparse-weight`（覆盖策略）、
`--scenario`（可重复传）、`--collection`（指定物理集合，用于新旧对比）。

### 3. 索引选型与蓝绿重建

**选型**：建集合原本用 `AUTOINDEX`（Milvus 自己挑索引类型和参数，不可控也没法调优），现在显式声明：

| 字段 | 索引 | 参数 | 为什么 |
| --- | --- | --- | --- |
| `dense_vector` | HNSW | M=32, efConstruction=256, COSINE | 万级~百万级文档 + 要求低延迟高召回，HNSW 综合表现最好；IVF 系列省内存但召回要靠 nprobe 换，调参更麻烦 |
| `sparse_vector` | SPARSE_INVERTED_INDEX | DAAT_MAXSCORE, IP | BGE-M3 的词权重向量，MaxScore 剪枝遍历更快 |

查询期用多大的 `ef` 不在这里配，而在意图策略表里（不同意图不同 ef），避免同一份参数出现在两处。

**调参**：

```bash
python -m processor.index_manager.benchmark --build      # 建临时集合跑 M × efConstruction × ef_search 网格
python -m processor.index_manager.benchmark --ef-only    # 只跑 ef_search（不重建，最快）
```

网格默认 `M ∈ {16,32,48} × efConstruction ∈ {128,256} × ef_search ∈ {32,64,128,256}`，
输出 markdown 表（recall / p95 延迟）供定参，跑完自动删临时集合（`--keep` 保留）。

**版本化与别名**：查询侧永远只认 `.env` 里的 `CHUNKS_COLLECTION=kb_chunks`，
这个名字被改造成**别名**，物理集合叫 `kb_chunks_v1 / _v2 / ...`。切换索引 = 改别名指向，原子且瞬间完成。

已有集合先做一次迁移（**执行前停掉两个 API 服务**：rename 期间 `kb_chunks` 这个名字短暂不存在，在跑的查询会报错）：

```bash
python -m processor.index_manager.migrate_alias                    # 只看计划
python -m processor.index_manager.migrate_alias --yes --verify-dml # 执行并插探针验证别名可写
python -m processor.index_manager.migrate_alias --rollback --yes   # 撤回迁移
```

**蓝绿重建**：

```bash
python -m processor.index_manager.rebuild --status                       # 别名指向、各版本状态、可清理版本
python -m processor.index_manager.rebuild                                # 只打印计划
python -m processor.index_manager.rebuild --yes --prepare-only           # 建好验好停在 ready，等人工切
python -m processor.index_manager.rebuild --yes                          # 全流程（含切换）
python -m processor.index_manager.rebuild --switch --version 2 --yes     # 人工上线
python -m processor.index_manager.rebuild --rollback --version 1 --yes   # 回滚
python -m processor.index_manager.rebuild --cleanup --yes                # 清理过了排水期的旧版本
```

状态机（记录在 Mongo `index_registry`，跨进程可见，所以接口和 CLI 看到的是同一份）：

```
building → backfilling → validating → ready → active
                                     ↘ failed
旧 active → draining（保留 INDEX_RETIRE_AFTER_DAYS 天，期间可回滚）→ retired
```

> **一致性怎么保证**：顺序是「**先开双写，再搬存量**」，且搬运按 `file_title` 分组、
> 搬之前先删新集合里的同名文件。这样两种交错情况都是对的：文件在搬运前被重新导入，
> 搬到时先删再拷的是同一份新数据；文件在搬运后被重新导入，双写会删掉搬来的旧版再写新版。
> 既不会重复，也不会被覆盖成旧数据。双写期间 building 侧写失败只记日志并置 `failed`，
> **不阻塞**线上 active 的导入。

搬运模式：`--mode move_vectors`（直接搬旧集合里的向量，快、不花模型算力，要求 embedding 模型没变）
或 `--mode re_embed`（读正文重算向量，慢且花钱，换了 embedding 模型才用）。

**切换前的两道闸门**（`validate.py`，也可单独跑）：

| 校验项 | 通过条件 |
| --- | --- |
| 行数 | 新旧集合行数相等（两种搬运模式都查） |
| gold 命中率 | 取样 N 条（默认 50），新旧命中率差 ≤ 1 个百分点 |
| top10 Jaccard | ≥ 0.85；`move_vectors` 模式 ≥ 0.95（逐条搬运本该几乎一致） |
| 文件覆盖 | 双写期新导入的 `file_title` 在两个集合都查得到 |

```bash
python -m processor.index_manager.validate --old kb_chunks_v1 --new kb_chunks_v2
```

**索引健康度**（不跑评测也能看的代理指标）：

```bash
python -m processor.index_manager.quality --json
```

判定项：sealed segment > 64（碎片化，查询要扫的段变多、延迟上升）、
实际行数与注册表偏差 > 1%（导入漏写或双写失败）、索引类型与配置不符（这批数据没吃到调优参数）。

### 4. 监控告警与慢查询日志

两个 API 都暴露 `/metrics`（`prometheus-fastapi-instrumentator` + 自定义 `kb_*` 指标），
Milvus 自身指标直接从 `standalone:9091` 抓。

| 指标 | 类型 | 看什么 |
| --- | --- | --- |
| `kb_node_latency_seconds{flow,node}` | H | 哪个图节点慢 |
| `kb_node_errors_total{flow,node}` | C | 哪个节点在报错 |
| `kb_retrieval_latency_seconds{path}` | H | dense/hyde/web 各路耗时（定位是哪一路拖慢整体） |
| `kb_retrieval_hits{path}` | H | 各路命中条数 |
| `kb_query_latency_seconds{intent}` | H | 整条链路 p99（含 LLM 生成） |
| `kb_query_total{intent,status}` | C | QPS 与失败率 |
| `kb_slow_query_total{reason}` | C | 慢查询数（`reason=total` 整体慢 / `path` 单路慢） |
| `kb_rerank_output_size` | H | 重排断崖截断后真正交给 LLM 的条数 |
| `kb_import_tasks_total{status}` / `kb_import_running_tasks` | C / G | 导入吞吐与堆积 |
| `kb_index_rows{collection,role}` / `kb_index_sealed_segments` | G | 索引规模与碎片化 |
| `kb_rebuild_status{logical,version,status}` | G | 各版本处于什么状态（当前状态置 1） |
| `kb_eval_recall{scenario,k,mode}` | G | 最近一次评测的召回率（`mode=offline/online`） |

> **一条贯穿全项目的约定**：检索**失败**绝不记成「0 命中」或「耗时很短」的样本——
> 那会把故障画成"变快了/没结果"。失败只进 `kb_node_errors_total` 与 query_log 的 error 字段。
> 同理，巡检采集不到信号时报 `alert_only` 而不是 `healthy`；本轮没评出的线上结果标记为 stale，
> 不参与重建判定。**「没量到」永远不等于「健康」。**

13 条告警规则（`deploy/monitoring/prometheus/alert.rules.yml`）：
`TargetDown`、`QueryP99High`、`RetrievalSlow`、`QueryErrorRateHigh`、`SlowQueryBurst`、`NodeErrorBurst`、
`RecallDrop`（离线 recall@10 同比跌 2%）、`OnlineRecallLow`（线上抽样相关率 < 70%）、
`RebuildFailed`、`IndexFragmented`、`MilvusMemHigh`、`AppMemHigh`、`ImportErrorRate`。

通知走 Alertmanager 的 webhook。钉钉/飞书机器人要求的签名与 body 格式和 Alertmanager 不一样，
需要一个转发网关，在 `.env` 里设 `ALERT_WEBHOOK_URL` 指过去。默认值故意指向一个没人监听的端口：
推不出去会在 Alertmanager 日志里报错，而不是把告警悄悄丢掉。

Grafana 已 provision 好数据源与看板 `知识库问答系统 · 检索体系总览`，17 个面板覆盖：
查询 QPS / p99 延迟 / 失败率 / 慢查询数、按意图拆分的 QPS 与延迟分位、单路检索 p95 与平均命中数、
节点耗时与失败次数、离线与线上召回率、索引规模与碎片化、蓝绿版本状态、常驻内存、重排输出与导入任务、HTTP 接口延迟。
阈值在**三处**是对应的（`.env` ↔ `config/*.py` ↔ `alert.rules.yml`），
改一处记得同步，`alert.rules.yml` 文件头列了对应关系。

**慢查询日志**（Mongo `query_log`）：每次问答记一条（`QUERY_LOG_SAMPLE_RATE` 可抽样），
整体超 `SLOW_QUERY_TOTAL_MS` 或任一路超 `SLOW_QUERY_PATH_MS` 时打 `slow=true`。
字段含意图与置信度、生效的 strategy、各路命中数/耗时/参数、RRF 与重排后的切片三元组、总耗时。

> 刻意**不存** 1024 维查询向量：可随时重算、体积大、对排查无益，只存检索参数。
> 写日志也绝不拖垮请求——所有异常只记不抛，Mongo 挂了查询照常返回。

### 5. 线上抽样评测与索引巡检

离线评测集衡量的是「对理想问题的表现」，真实用户的问题更短、更口语、带错别字，
甚至问的是知识库里根本没有的东西。所以补一层线上抽检：从 `query_log` 抽真实查询，
把当时真正交给大模型的那几条切片回查出来，让 LLM 当裁判按 0/1/2 打分
（2 = 能直接回答，1 = 沾边但不足以回答，0 = 无关）。

```bash
python -m eval.online_eval --dry-run                # 只抽样打印，不调 LLM 也不落库
python -m eval.online_eval --sample 30 --seed 42    # 跑一轮 → Mongo eval_online + eval/results/{ts}_online.json
python -m eval.online_eval --sample 30 --json
```

产出的 `recall@1 / recall@3` 是**相关率代理指标**，不是离线那个真召回率（没有 ground truth 算不出来），
看板上靠 `mode` 标签区分两条线——两条线差距拉大本身就是信号。
有效判定率 `judge_rate < 50%` 时 `exit 3`，这轮结果不采信。

**巡检调度器**把这些串成闭环：

```bash
python -m processor.index_manager.scheduler                  # 手动跑一轮
python -m processor.index_manager.scheduler --no-online      # 只看索引侧信号，不调 LLM
python -m processor.index_manager.scheduler --status 5       # 看最近 5 轮结论
python -m processor.index_manager.scheduler --loop           # 常驻，每 SCHEDULER_INTERVAL_HOURS 小时一轮
docker compose up -d scheduler                               # 容器里常驻
```

每轮三步：① 线上抽样评测 → ② 采集索引质量 + query_log 延迟/慢查询 + 注册表状态 → ③ 判定。
结论落 Mongo `scheduler_runs`，所以 CLI 跑的轮次在运维接口里也看得见。

| 信号 | 结论 |
| --- | --- |
| 线上相关率连续 2 轮 < 70% | 建议重建（可自动触发） |
| sealed segment > 64 | 建议重建（碎片化，压实能救） |
| 实际索引类型与配置不符（如还是 AUTOINDEX） | 建议重建（这批数据没吃到调优参数） |
| p99 延迟 > 1.5 × 基线 | 建议重建（需先设 `LATENCY_P99_BASELINE_MS`，否则不判） |
| 行数与注册表偏差 > 1% | **只告警**——重建只会把不一致的数据原样搬一遍，根因在导入/双写 |
| 空检索条数 ≥ max(3, 抽样数的 20%) | **只告警**——知识库里压根没有相关资料，该补语料 |
| 注册表读不到 / 已有在建版本 | **只告警**，并说明本轮判定不完整 |
| 什么都没量到（Milvus 连不上） | `alert_only`，绝不报 healthy |
| 线上信号是历史值（本轮没评出新结果） | 标记 stale，只展示，不参与重建判定 |

> **自动重建的边界**：`SCHEDULER_AUTO_REBUILD=true` 时最多做到 `--prepare-only`
> （建新集合 + 双写 + 一致性校验 + 召回门禁，停在 `ready`）。**切别名永远留给人确认**——
> 线上事故里最贵的就是「自动切换切错了」。

### 6. 运维接口（import-api，端口 8000）

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/admin/index/status?quality=true` | 别名指向、各版本状态、重建进度、可清理版本（`quality=true` 附索引质量指标，稍慢） |
| POST | `/admin/index/rebuild` | 后台发起蓝绿重建，立刻返回 202；body 支持 `mode/prepare_only/skip_eval/eval_limit/M/ef_construction/drop_on_failure` |
| POST | `/admin/index/switch` | 人工把别名切到指定版本，回滚也是它 |
| GET | `/admin/scheduler/status?limit=3` | 本轮巡检执行状态 + 最近几轮判定结论 |
| POST | `/admin/scheduler/run` | 后台跑一轮巡检；body 支持 `sample/auto_rebuild/force/no_online` |
| GET | `/metrics` | Prometheus 指标 |

重建与巡检**互斥**：任一在跑时发起另一个返回 409。
原因很实在——重建期间采到的全是过渡状态（双写中、行数对不上），判定结论没有意义。

### 7. 新增配置项

全部有代码内默认值，`.env` 里不写也能跑。

| 键 | 默认 | 作用 |
| --- | --- | --- |
| `INTENT_ENABLED` | true | 意图识别总开关，false = 改造前行为 |
| `HNSW_M` / `HNSW_EF_CONSTRUCTION` | 32 / 256 | 建索引参数 |
| `INDEX_DOUBLE_WRITE_ENABLED` | true | 重建期间新旧集合双写 |
| `INDEX_BACKFILL_BATCH` | 200 | 存量搬运批大小 |
| `INDEX_REBUILD_MODE` | move_vectors | 默认搬运模式（另一个是 `re_embed`） |
| `INDEX_RETIRE_AFTER_DAYS` | 7 | 旧版本保留几天再删（回滚窗口） |
| `INDEX_VALIDATE_SAMPLES` | 50 | 一致性校验取样条数 |
| `INDEX_VALIDATE_HIT_DIFF` | 0.01 | gold 命中率最大允许差 |
| `INDEX_VALIDATE_JACCARD` / `_MOVE` | 0.85 / 0.95 | top10 Jaccard 下限 |
| `INDEX_MAX_SEALED_SEGMENTS` | 64 | 碎片化告警阈值 |
| `INDEX_P99_DEGRADE_RATIO` | 1.5 | p99 相对基线的劣化倍数阈值 |
| `METRICS_ENABLED` | true | 关掉后不记任何自定义指标 |
| `QUERY_LOG_ENABLED` | true | 关掉后不写 query_log |
| `SLOW_QUERY_TOTAL_MS` / `SLOW_QUERY_PATH_MS` | 3000 / 1000 | 慢查询判定阈值 |
| `QUERY_LOG_SAMPLE_RATE` | 1.0 | 查询日志抽样率（高 QPS 时调低） |
| `LATENCY_P99_BASELINE_MS` | 0 | p99 基线，0 表示不判延迟劣化 |
| `SCHEDULER_INTERVAL_HOURS` | 6 | 巡检周期 |
| `SCHEDULER_AUTO_REBUILD` | false | 判定出「该重建」后要不要真动手 |
| `SCHEDULER_PREPARE_ONLY` | true | 自动触发时只做到 ready，不切别名 |
| `SCHEDULER_EVAL_LIMIT` | 20 | 自动触发时门禁每场景取样条数 |
| `SCHEDULER_ONLINE_SAMPLE` | 30 | 每轮线上抽样条数，0 表示跳过 |
| `ALERT_WEBHOOK_URL` | `http://host.docker.internal:5001/alerts` | Alertmanager 推送地址（默认端口故意没人监听） |
| `GRAFANA_ADMIN_USER` / `_PASSWORD` | admin / admin | Grafana 首次登录 |
| `GRAFANA_ANONYMOUS` | true | 免登录只读看板，生产建议 false |

评测侧的阈值（门禁跌幅、场景配额、抽查比例、线上相关率阈值与连续轮数）在 `eval/config.py`，
不走环境变量——它们是评测口径的一部分，改了就该在代码评审里看得见。

### 8. 回退手段

| 出问题 | 怎么办 |
| --- | --- |
| 意图判错 | `.env` 里 `INTENT_ENABLED=false` 重启 query-api，行为完全等同改造前 |
| 监控栈有副作用 | 删根 compose 里 monitoring 那一行 `include`，整体下线 |
| 新索引不对 | `rebuild --rollback --version <旧版本> --yes`，一条命令切回 |
| 重建失败留下垃圾 | `rebuild --mark-failed --version N`，或发起时带 `--drop-on-failure` |
| 指标/日志拖慢请求 | `METRICS_ENABLED=false` / `QUERY_LOG_ENABLED=false`（两者本就异常只记不抛，不会打断请求） |
| 巡检乱触发重建 | `SCHEDULER_AUTO_REBUILD=false`（默认值），或 `docker compose stop scheduler` |

### 9. 首次接入顺序

1. `migrate_alias`（先看计划）→ 停两个 API → `--yes --verify-dml` → 起服务，确认 `/query` 正常
2. `generate_dataset --dry-run` → `--limit 20` 试跑 → 全量生成 → `sample_review` 抽 10% 人工核对
3. `run_eval --tag baseline`，这份结果文件留好当基线
4. `benchmark --build` 定 HNSW 参数，写回 `.env`
5. `rebuild --yes --prepare-only` → 确认 validate 与 gate 都过 → `--switch --version 2 --yes`
6. `docker compose up -d`（含监控栈）→ Grafana :3000 确认有数据 → 故意触发一条告警验证通知链路
7. `scheduler --sample 20` 跑一轮完整巡检，确认结论合理后再 `docker compose up -d scheduler` 常驻

> 第 5 步之前磁盘上会同时存在两份向量，先确认容量够。
