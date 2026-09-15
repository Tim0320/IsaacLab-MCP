# 受限 Dofbot 垂直訓練流程

這是目前第一條可執行的閉環，不是通用的任意 shell 或任意 Isaac Lab task launcher。

```text
EnvironmentContract
  → submit_dofbot_training_run
  → 可見 Kit + RecordVideo (750 frames，約 15 秒)
  → get_training_run_status
  → TrainingRunRecord
  → Verification Agent
  → EvidenceBundle: PASS / FIX_REQUIRED / RETHINK
```

## MCP actions

| Tool | 用途 | 安全界線 |
| --- | --- | --- |
| `submit_dofbot_training_run` | 提交 Dofbot grasp 或 lift | 只接受兩個 task、固定 CUDA 0、4 environments 預設、可見 Kit、750-frame 影片；拒絕任意 command、路徑與 `--headless`。 |
| `get_training_run_status` | 讀取持久化狀態與 artifacts | 不啟動、不控制程序。 |
| `cancel_training_run` | 寫入取消請求 | worker 只停止它自己建立的 launcher process tree，不依外部 PID。 |

`run_name` 只允許英數、`_`、`-`，防止被當成路徑或 command fragment。`max_iterations` 最少為 24，確保 32 steps/iteration 足以錄滿 750 frames。

## 執行順序

1. Isaac Sim Engineer 先建立並呼叫 `validate_environment_contract` 驗證 Dofbot asset、joint、physics 和 Sim evidence。
2. Isaac Lab Engineer 將同一份 contract 傳入 `submit_dofbot_training_run`。runner 再次驗證 schema、重新計算 fingerprint，並將不可變 contract snapshot 寫入 `training_projects/runtime_jobs/<job-id>/`。
3. Orchestrator 以 `get_training_run_status` 輪詢。`running` 或 `cancellation_requested` 不是完成；不要從 console、reward 或 checkpoint 名稱推論結果。
4. 成功時 worker 只在同時找到 `model_*.pt` 與 MP4 時，才產生 `training_run_record.json`。任一缺失會標記 `failed`。
5. Verification Agent 仍需獨立執行 evaluation，將數值、影片與行為條件寫入 `EvidenceBundle`。TrainingRunRecord 只證明這場訓練產出了可追蹤 artifacts，不證明 policy 已學會。

## 目前限制

- 只支援 `Isaac-Grasp-Cube-Dofbot-v0` 與 `Isaac-Lift-Cube-Dofbot-v0`；G1、TM6S 與公司新資產仍需下一個 allow-listed runner。
- worker 結束或 server 重啟後，狀態仍可讀取，因為 manifest、status、log 和 record 都保存於 job directory；但只由 worker 寫 terminal state。
- 取消是終止由 worker 建立的 launcher process tree，不修改 `D:\IsaacLab`、USD 或已產生的訓練 artifacts。
