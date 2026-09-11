# Isaac Lab RL 快速查表

## 先找哪裡

```powershell
# 找資產、質量、reward、termination 與速度範圍
rg -n "mass=|weight=|termination|break_distance|lin_vel_x" isaaclab_mcp/runtime_tasks

# 找既有訓練與獨立評估入口
rg -n "RecordVideo|video_length|WarmStartCheckpoint|passed" scripts

# 找可重用 task id
rg -n "Isaac-.*-v0" isaaclab_mcp/runtime_tasks
```

## 必填設定與意思

| 要調什麼 | 放在哪裡 | 先怎麼定 |
| --- | --- | --- |
| USD、質量、摩擦、碰撞、drive | `env_cfg.py` 的 scene/asset config | 先讀回 mass、關節名稱與碰撞；不要只相信 USD 路徑存在。 |
| action scale/offset | `actions.*` | offset 放已驗證初始姿勢；scale 先小，確認可行後只放大真正要移動的關節。 |
| observation | `observations.policy.*` | 放相對位置、速度、目標與抓取狀態；不放無法取得的未來資訊。 |
| reward | `rewards.*` 與 `mdp.py` | 主要成功項最大；接近、姿勢、平衡當 shaping；每項都要有對應量測。 |
| termination | `terminations.*` | 掉落、抓取分離、底座撞地等不可恢復狀態；不要把 time limit 當成功。 |
| PPO | `agents/rsl_rl_ppo_cfg.py` | 先沿用同類 task；只改一兩項，固定 seed 比較。 |

## 快速計算

| 問題 | 算式 | 例子 |
| --- | --- | --- |
| 錄影 frame 數 | `秒 × FPS` | `15 × 50 = 750 frames` |
| 訓練至少要幾 iteration | `ceil(VideoLength / steps_per_iteration)` | Dofbot：`ceil(750 / 32) = 24`；G1：`ceil(750 / 24) = 32` |
| 負載重量力 | `m × 9.81 N` | `15 kg = 147.15 N` |
| 雙手平均承擔力 | `m × 9.81 / 2` | `15 kg = 73.58 N/hand` |
| 物體成功率 | `成功 episode / 全部 episode` | 獨立 evaluation，不以 training reward 代替。 |

錄影限制：`VideoLength` 必須是 500–1000；`VideoInterval >= VideoLength`。G1 wrapper 為避免尾端不完整片段，最安全是將 interval 設為大於訓練總步數，只錄 step 0 的一段。

## 本專案可參考的起始值

| 情境 | 已用起始值 | 使用理由 |
| --- | --- | --- |
| Dofbot 夾取 | 64 envs、8 秒 episode、arm scale `0.05/0.08`、grasp reward 12 | 先縮小探索，優先形成穩定夾取。 |
| Dofbot 吊升 | 64 envs、20 秒 episode、lift progress 12、lift pose 8 | 只在夾取 checkpoint 通過後使用。 |
| G1 追人 | 人物速度 `0.65 m/s`、接觸門檻 `1 N`、max forward speed `1.0 m/s` | success 以 contact sensor 量到碰撞力判定，不以距離接近代替。 |
| G1 持 15 kg | 64 envs、速度 `0–0.35 m/s`、hold reward 10、grasp-lost termination | 先拿穩，再加入速度。 |
| G1 加速 | 速度 `0.35–0.80 m/s`、velocity reward 4 | 曾出現退化；建議改成較細的速度階段。 |

## PPO 參數白話

| 參數 | 用途 | 先調整時機 |
| --- | --- | --- |
| `num_steps_per_env` | 每次更新前每個環境收集的步數 | episode 太短或回饋太慢時再調。 |
| `max_iterations` | 更新次數與訓練長度 | 短跑只驗證流程；長跑才判斷是否學會。 |
| `actor.hidden_dims` / `critic.hidden_dims` | policy / value 網路容量 | 任務變複雜且曲線停滯時才增加。 |
| `init_std` | 動作探索幅度 | 一直撞飛、掉落時降低；卡住才小幅增加。 |
| `learning_rate` | 每次更新幅度 | 曲線劇烈震盪時降低；不要與多個參數同時改。 |
| `gamma` / `lam` | 多遠的未來 reward 影響現在動作 | 長期持物或長路徑任務可維持 `0.99 / 0.95` 基準。 |
| `entropy_coef` | 鼓勵探索的強度 | 太早收斂成單一錯誤動作時微調。 |
| `clip_param` / `desired_kl` | 限制 policy 更新幅度 | 訓練不穩時優先檢查，不要先放大網路。 |

## 最小驗收清單

1. 開始前：資產可載入、關節與物體 pose 合理、質量與碰撞 read-back 正確。
2. 訓練後：有 checkpoint、10–20 秒 MP4、沒有把 reward 當成功。
3. 獨立 evaluation：重新載入 checkpoint，量測成功率、持物／高度、速度誤差、termination；使用不同於 training log 的判定程式。
4. 報告：列出通過與未通過項目。任何輔助抓取、固定物體或簡化物理都要明講。

追人任務額外檢查：人要同時有可見模型與碰撞 proxy；「靠近」只作 dense reward，通過條件必須讀取 contact sensor 的力值。現有實作在 `isaaclab_mcp/runtime_tasks/g1_people_chase/`。
