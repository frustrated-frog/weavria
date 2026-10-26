# 软件发布就绪评测适配器

该目录提供 Weavria 的第二个领域适配器，用于验证任务语义义务覆盖机制并未绑定 FinScope。
它通过一个只读、强类型的发布档案能力，评估发布决策、质量阻塞和回滚准备；只有用户明确
要求时，才把用户可见变更与客户通知提升为 `explicit_only` 信息义务。

该适配器目前只服务于冻结数据集、确定性跨层测试和显式真实模型探针，不是发布管理产品，
也不连接生产缺陷系统、CI、通知渠道或外部写操作。

运行确定性测试：

```bash
uv run pytest tests/release_readiness_app
```

运行真实模型探针：

```bash
export WEAVRIA_MODEL_BASE_URL="https://provider.example/v1"
export WEAVRIA_MODEL_NAME="model-name"
export WEAVRIA_MODEL_API_KEY="..."
export WEAVRIA_MODEL_JSON_MODE="true"
uv run python examples/live_release_readiness.py --coverage model --case all
```

探针支持与 FinScope 基线相同的 `--coverage off|profile|model`，以及
`--understanding fixed|live|forced-broad`。`core-go-no-go` 是必须排除用户变更和客户通知的
负例；`explicit-release-communication` 是必须选择这两个证据面的正例。

报告把 required fields 精确度与完整需求契约精确度分开。冻结档案只能证明稳定事实；若真实
规划器把需求声明为 `current`，默认结构证据会保持 Need 为 `partial`，最终结果可以是
`unverifiable`。不要仅凭 `generated_at` 字段把 Fixture 宣称为调用时最新。
