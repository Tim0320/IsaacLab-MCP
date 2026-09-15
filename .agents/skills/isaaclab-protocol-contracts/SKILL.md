---
name: isaaclab-protocol-contracts
description: Create, validate, and route Isaac Sim/Isaac Lab multi-agent protocol documents in this repository. Use for EnvironmentContract, TrainingRunRecord, EvidenceBundle, SceneChangeRequest, version linkage, or evidence handoffs; do not use to launch training or directly edit a USD scene.
---

# Isaac Lab Protocol Contracts

讓每個 Agent 只交接可驗證的資料，不把文字敘述、reward 曲線或 checkpoint 存在誤當成已完成任務。

完整欄位、範例與驗證規則見 `F:\IsaacLab-MCP\docs\MULTI_AGENT_PROTOCOL.md`；本 Skill 只保留實際操作時需要的路由與邊界。

## 邊界

- 實作與文件在 `F:\IsaacLab-MCP`；`D:\IsaacLab` 是 runtime/reference，只讀。
- `validate_environment_contract`、`validate_evidence_bundle`、`validate_scene_change_request` 是純資料驗證：不啟動 Kit、訓練、不讀寫 USD，也不會替任何 Agent 產生成功證據。
- `submit_dofbot_training_run` 只支援兩個 allow-listed Dofbot task，固定使用可見 Kit、CUDA 0 與約 15 秒 `RecordVideo`。它不接受任意 command、路徑或 `--headless`，也不代表其他 Isaac Lab task 已 MCP 化。
- Isaac Sim Engineer 擁有場景與物理事實；Isaac Lab Engineer 只能引用已驗證的 `EnvironmentContract`，不能猜測或覆寫 USD、drive、collision 或 joint limit。

## 交接順序

1. **Sim → Lab：EnvironmentContract**：先填資產 URI/hash、prim path、關節名稱與 limits、控制與物理 timestep、物理 read-back，再呼叫 `validate_environment_contract`。失敗時回給場景擁有者修正。
2. **Lab → Verification：TrainingRunRecord**：以 `submit_dofbot_training_run` 啟動的 Dofbot job 在實際結束後會自動寫入 record；`get_training_run_status` 回傳路徑。`status=completed` 必須同時帶 checkpoint 和影片，否則 worker 標記 failed。
3. **Verification → Orchestrator：EvidenceBundle**：每個成功條件都填 pass/fail 與對應 artifact。`decision=PASS` 時，所有 criteria 必須通過；否則使用 `FIX` 或 `RETHINK`。
4. **Lab → Sim：SceneChangeRequest**：當 reward、觀測或 action 已無法合理補救時，提出請求而非直接改場景。它必須列出原因、預期影響、風險與 rollback；以 `validate_scene_change_request` 先驗證格式。

## 版本完整性

- 對每份已通過的合約保存 validator 回傳的 fingerprint。TrainingRunRecord 必須引用完全相同的 EnvironmentContract fingerprint。
- USD、articulation、joint limit、control dt 或 physics dt 有任一變動，就建立新的 EnvironmentContract；不要沿用舊 fingerprint。
- 影片要可見、長度 10–20 秒（通常約 15 秒），並和 evaluation JSON、checkpoint、log 一起保存。單一 reward、畫面或檔案都不足以判定成功。

## 何時改用其他 Skill

- 設計 action、observation、reward、curriculum、PPO 或可見錄影：使用 `$isaaclab-rl-training`。
- telemetry 顯示 joint 變化、但 RTX 模型沒有動：使用 `$isaaclab-joint-visual-sync`，先修 articulation/visual sync，再談 RL。
- 需要實際修改或 read-back Isaac Sim stage：使用 Isaac Sim 對應的 live workflow；這個 Skill 不能取代它。
