# G1 people-interaction RL

這組實驗使用 Unitree G1 作為可受力、可訓練的人形 agent。Isaac Sim `People` 角色主要是動畫角色，適合當視覺目標或行人；它不是具備 actuator、joint drive 與接觸控制的 RL 機器人，因此不能直接把 People 角色丟給 PPO 學跑步或拿東西。

## 目前能力判定

| 技能 | Isaac Lab task | 現況 | 判定 |
| --- | --- | --- | --- |
| 跑步 | `Isaac-Velocity-Flat-G1-v0` | 有 reward、RSL-RL PPO runner 與 G1 locomotion asset | 可直接 RL |
| 拿東西 | `Isaac-PickPlace-Locomanipulation-G1-Abs-v0` | 有 G1 雙手、物件與 Robomimic BC-RNN，但 `rewards=None` | 可收 demonstration，尚不可直接 PPO |
| 追人 | `Isaac-Chase-Person-G1-v0` | 專案已新增 moving Reallusion Worker target、安全距離 reward 與 RSL-RL runner | 已通過 PPO pipeline 與畫面驗證，尚未收斂 |

## 自主行為架構

不把三件事塞進同一個 policy。先分別訓練，再由高階 selector 根據目標切換技能：

1. `run` 學平衡、速度追蹤與轉向。
2. `pick` 先固定下半身，學接近、合手、穩定夾取與抬起。
3. `chase` 把移動人物的相對 XY 與 heading error 轉成 G1 預訓練 locomotion 熟悉的 `(forward, lateral, yaw rate)` command，再由 policy 輸出 37 維 joint-position target；只有人物碰撞代理量到接觸力才算成功。
4. selector 執行 `跑到物件 -> 拿起 -> 追到並碰觸人物`，失敗時只重試目前階段。

這裡的「自主思考」指 policy 根據 observation 選 action，以及 selector 根據任務狀態選 skill。RL 不會自行理解「人」或「東西」的語意；必須用 observation、reward、termination 與 curriculum 明確定義。

## 安全邊界

- 追人只在模擬內執行。
- 紅色 Worker 保留作為外觀，另用同步移動、底部離地 `0.1 m` 的 kinematic capsule 提供碰撞。
- capsule 的獨立 contact sensor 達 `1 N` 才觸發 `person_contact` 成功並終止 episode。capsule 不碰地，因此 sensor 的 `net_forces_w` 只會在 G1 碰到人物時升高。
- G1 傾倒超過 `0.8 rad` 或 root 高度低於 `0.45 m` 時判定跌倒；觸碰人物本身不算跌倒。
- 拿取尚未穩定前固定 pelvis，不同時訓練走路與手部操作。

## 驗證與第一階段訓練

檢查本機 task registration：

```powershell
& 'C:\isaacsim\python.bat' '.\scripts\inspect_people_rl_tasks.py' `
  --viz kit --device cuda:0 `
  --experience 'F:\IsaacLab-MCP\apps\isaaclab.dofbot.demo.kit' `
  --output '.\artifacts\people_rl_inspection.json'
```

執行第一階段 G1 跑步 PPO，並用 Isaac Lab 內建 `RecordVideo` 錄製 15 秒：

```powershell
& '.\scripts\train_g1_run_with_video.ps1' `
  -Task Isaac-Velocity-Flat-G1-v0 `
  -NumEnvs 64 -MaxIterations 32 `
  -RunName g1_run_visible_recorded
```

追人 task 使用同一個可見錄影 launcher：

```powershell
& '.\scripts\train_g1_run_with_video.ps1' `
  -Task Isaac-Chase-Person-G1-v0 `
  -NumEnvs 64 -MaxIterations 32 `
  -RunName g1_chase_feasibility_visible
```

chase 預設從本機官方 G1 flat-locomotion `checkpoint.pt` warm-start，讓 agent 先保有站立與走路能力，再 fine-tune 跟隨行為。只有比較從零訓練時才使用 `-FromScratch`。

訓練後用單一 environment 做可見 evaluation，關閉 observation corruption 與隨機推力，並錄製 15 秒：

```powershell
& '.\scripts\play_g1_chase_with_video.ps1'
```

launcher 固定使用可見 Kit 視窗、`cuda:0`、單 GPU，拒絕 `--headless`。`VideoLength` 限制 500–1000 frames，在已驗證的 50 FPS runtime 對應 10–20 秒。

## 2026-09-10 至 2026-09-11 實機驗證結果

- `inspect_people_rl_tasks.py` 在可見 Kit 中確認：run 與 chase 均有 reward 和 RSL-RL entry point；pick 有 Robomimic entry point，但 `rewards=None`。
- G1 run 的 32-iteration PPO smoke run 正常結束並產生 15.02 秒影片。這只驗證訓練管線；另用 Isaac Lab 預訓練 checkpoint 播放 15.00 秒，確認模型能實際移動。
- 第一版 G1 chase 從隨機關節 policy 開始，16 environments、32 iterations 後仍頻繁觸發 `base_contact`，距離誤差約 3 m。影片雖確認人物與關節會動，agent 尚未正常跑起來。
- 修正版將人物位置轉成 bounded velocity command，保留原始 G1 velocity-tracking rewards，並預設載入官方 locomotion checkpoint。後續影片與 metrics 應以 warm-start run 為準。
- 修正版以 16 environments fine-tune 32 iterations。mean reward 最高 `20.70`、mean episode length 最高 355 steps；大部分訓練期間的 `base_contact` 約 `0.06–0.12`，最後一批為 `0.1875`，相較第一版約 `1.0` 已明顯改善。
- `model_1030.pt` 的單一 environment evaluation 關閉 observation corruption 與隨機推力。15.00 秒影片的 1、5、10、14 秒抽幀皆顯示 G1 保持站立並轉向、前進或側移，人物也持續改變位置。這證明 locomotion 已接通；精準距離跟隨仍需更長訓練與多 seed evaluation。
- 加速階段將人物軌跡速度從 `0.35 m/s` 提高到 `0.65 m/s`，距離控制 gain 從 `0.8` 提高到 `1.0`。G1 velocity command 上限維持官方 locomotion checkpoint 已涵蓋的 `1.0 m/s`，降低突然外插到未知步態的風險。
- 加速版由 `model_1030.pt` 接續 fine-tune 128 iterations，輸出 `model_1157.pt`。末段 mean reward 約 `40.5`、mean episode length 約 750 steps，約 `74%` 回合跑滿時間，`base_contact` 約 `0.26`。單一 environment 的 15.00 秒可見 evaluation 顯示 G1 全程站立，並持續轉向、前進或側移追隨移動人物。`0.65 m/s` 是人物軌跡速度；G1 實際速度由距離 command 動態決定，不應解讀成全程固定 `0.65 m/s`。
- Worker 的原始 MDL 在本機顯示缺少 `OmniRLBase`，幾何仍以紅色 fallback material 正常顯示。這不影響 command/reward 計算，但正式展示前應補齊材質或改用本地人形資產。
- 2026-09-11 起，chase 的成功定義改為實際接觸。舊 `model_1157.pt` 只證明跟隨能力，必須在新 collision/contact task 中重新 fine-tune 並達到 deterministic evaluation 的接觸成功率門檻，才能宣稱通過。
- 接觸版 command 在距離控制外加入 `0.25 m/s` closing bias，避免 G1 與 `0.65 m/s` 人物形成固定落後距離；`person_contact` 後立即結束 episode。
- `model_1411.pt` 是目前保留的最佳接觸 checkpoint。64 environments 的最後一批 `person_contact=0.984375`，訓練期間最高為 `1.0`；同批 `fallen_orientation=0.015625`、`fallen_height=0.0`，已超過 80% 的訓練批次門檻。這個數字不是多 seed 的獨立 deterministic 評估，兩者不可混為一談。
- `model_1411.pt` 的可見單環境回放錄製 15.00 秒；影格顯示 G1 接近並與人物重疊，接觸後約在 9.5 秒重置到遠處，再開始下一回合。這個重置由 `1 N` contact sensor termination 觸發，不是距離門檻。
- 後續 32-iteration 記錄驗證確認 `Metrics/success_rate` 能在實際接觸 reset batch 顯示 `1.0`。該短 run 最後 `person_contact=0.7923`，低於 `model_1411.pt`，所以不取代最佳 checkpoint。
- 內建 `RecordVideo` 的正式訓練與回放影片皆為約 15 秒。launcher 現在預設只在 step 0 錄影，並拒絕任何不足 750 frames 的尾段錄影觸發。
