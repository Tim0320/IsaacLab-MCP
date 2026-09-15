---
name: isaaclab-rl-training
description: Design, implement, train, and independently validate Isaac Lab reinforcement-learning tasks in this repository. Use for assets, observations, actions, rewards, PPO settings, curriculum, or recorded training evidence; not for a frozen viewport with moving joint telemetry.
---

# Isaac Lab RL training

將可執行的訓練、可量測的成功條件與可見錄影一起交付。reward 變高或 checkpoint 存在，都不能單獨代表任務學會。

## 專案邊界

- 實作、測試、log 與 Skill 都在 `F:\IsaacLab-MCP`。
- `D:\IsaacLab` 是 Isaac Lab 3.0.0 runtime/reference：只讀取它的 launcher 和 API，不修改、清理或複製其中的檔案。
- Kit-bound 程式使用 `C:\isaacsim\python.bat` 或本專案既有 PowerShell launcher。訓練保持可見，不使用 `--headless`。
- 若 joint 數值有變但 RTX 模型沒有動，改用 `$isaaclab-joint-visual-sync`，不要把它當作 PPO 或 reward 問題。
- 若要交接 Sim 場景事實、訓練結果或驗證證據，改用 `$isaaclab-protocol-contracts`；其現有 MCP tools 只驗證文件，不能啟動訓練或直接改場景。

## 建立新訓練的順序

1. **資產與物理**：在 `env_cfg.py` 定義 USD、關節 drive、collision、摩擦、質量與 reset pose。先以一個 environment 驗證關節、末端與物體確實能到達任務姿態。
2. **動作與觀測**：action 要有清楚的物理意義並在 joint limits 內；observation 至少包含完成任務所需的相對位置、速度或抓取狀態。避免 policy 只靠 world 座標或看不到關鍵物體。
3. **分段 curriculum**：把「能接近／拿穩」和「拿著移動／抬起」拆成 task。前一段要經獨立 evaluation 通過，才拿 checkpoint warm-start 下一段；每一段用不同 experiment/run 名稱保留結果。
4. **reward 與 termination**：先給任務本體最大的正向 reward，再加少量 shaping（接近、姿勢、穩定）。失敗事件要能 reset 並記錄，例如掉落、手離開、底座碰地。每個 reward 都應對應一個可量測的成功指標。
5. **先小後大**：先用 1–4 environments 檢查 reset、動作與數值，再用 32–64 environments 訓練。不要在未驗證資產或 reward 前直接跑長訓練。

## 參數的實用規則

- 小型手臂：先縮小 arm action scale，讓探索不會立刻掃飛物體；確認可夾後才放大抬升關節的 scale。
- 速度任務：先站立／慢走，再逐段加速，例如 `0–0.35 → 0.35–0.50 → 0.50–0.60 m/s`。只在前段持物率、存活率都通過時升級。
- 重物：重量力用 `F = m × 9.81` 快估；15 kg 約為 147 N。雙手平均至少要承擔約 74 N，實際 force cap 還要預留加速與姿勢誤差。
- reward 權重不是固定配方。先看各 term 的 episode contribution；若一個 shaping term 壓過任務成功 term，先降低它，不要只增加總 reward。
- PPO：探索太亂先降低 `init_std` 或 action scale；收斂慢才逐步調整 iteration、learning rate 或網路大小。保留相同 seed 作 A/B 比較。

詳見 [快速查表](references/quick-reference.md)：已有數值、公式、搜尋指令與每個 PPO 參數的用途。

## 訓練、錄影與判定

使用現有 wrapper，不要手動拼出容易遺漏錄影設定的指令：

```powershell
& .\scripts\train_dofbot_with_video.ps1 -Task Isaac-Grasp-Cube-Dofbot-v0 -NumEnvs 32 -MaxIterations 40 -RunName grasp_visible
& .\scripts\train_g1_run_with_video.ps1 -Task Isaac-Carry-Box-G1-v0 -NumEnvs 64 -MaxIterations 64 -RunName carry_visible
```

每次訓練必須用 Isaac Lab `RecordVideo` 保留 10–20 秒 MP4；本機已驗證 50 FPS，所以 750 frames 約 15 秒。訓練後回報 log、checkpoint、MP4 與獨立 evaluation JSON。

成功條件至少包含：物體質量或高度等物理 read-back、連續持物／到達時間、成功率或掉落率、termination 統計與可見影片。Dofbot 的 `model_39.pt` 只驗證夾取；`model_98.pt` 沒有通過吊升，不得稱為 lift success。G1 15 kg 範例使用雙手的順應式抓取輔助；它不是單靠手指接觸摩擦的真實抓取，報告時必須明確標註。
