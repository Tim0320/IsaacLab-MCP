# IsaacLab-MCP

在 Windows 上以 Model Context Protocol 提供 Isaac Lab 的環境診斷與工作流程工具。

目前骨架以本機 `D:\IsaacLab` 為開發基線，已驗證版本為 Isaac Lab 3.0.0，搭配 `C:\isaacsim` Isaac Sim 6.0.1。路徑可由環境變數覆寫，不會寫死在程式碼中。

## 架構

MCP server 在獨立 Python `.venv` 執行，避免把 Kit runtime 載入 MCP server 行程。Dofbot 的受限 MCP worker 會再以 `C:\isaacsim\python.bat` 啟動可見 Isaac Lab；其他 task 仍維持手動 launcher。

目前 named tools：

| Tool | 功能 | Runtime |
| --- | --- | --- |
| `get_isaac_lab_capabilities` | 回報目前已實作與尚未實作的能力 | MCP Python |
| `get_isaac_lab_status` | 檢查路徑、版本、launcher、source packages 與 Git 狀態 | MCP Python，唯讀 |
| `list_isaac_lab_tasks` | 從 task package 的 `gym.register(id=...)` 靜態列出 task IDs | MCP Python，唯讀 |
| `design_lifting_training` | 設計吊升任務的資產、控制、RL 訊號、curriculum、PPO 與 metrics | MCP Python，唯讀 |
| `design_people_rl_program` | 規劃 G1 跑步、拿取、追人三項技能並回報各自的 RL readiness | MCP Python，唯讀 |
| `create_lifting_training_project` | 預覽或產生 training design packet，預設 `preview=true` | MCP Python，寫入指定 training root |
| `validate_lifting_training_project` | 驗證 packet schema、資產參照與 training readiness | MCP Python，唯讀 |
| `validate_environment_contract` | 驗證 Isaac Sim → Isaac Lab 的版本化環境資料契約 | MCP Python，唯讀 |
| `validate_evidence_bundle` | 驗證 evidence 是否鎖定指定 environment、run、checkpoint 與 criteria | MCP Python，唯讀 |
| `validate_scene_change_request` | 驗證 Lab → Sim 的受控能力請求 | MCP Python，唯讀 |
| `submit_dofbot_training_run` | 提交白名單 Dofbot grasp／lift 訓練 | 可見 Kit、CUDA 0、固定 15 秒 `RecordVideo` |
| `get_training_run_status` | 讀取持久化 job 狀態與 artifacts | MCP Python，唯讀 |
| `cancel_training_run` | 要求 worker 停止其 own launcher process tree | MCP Python，受限寫入 |

`list_isaac_lab_tasks` 是靜態 discovery，不會啟動 Kit；動態產生的 registry 項目不在此階段保證完整。

## 新訓練設計流程

1. 呼叫 `design_lifting_training`，輸入用途、資產路徑、joint/body 名稱、負載、目標高度與安全工作負載。
2. 檢查回傳的 `missing_requirements` 與 `readiness`。缺少真實資產時仍可保存 draft，但不可宣稱可訓練。
3. 呼叫 `create_lifting_training_project`。先保留預設 `preview=true`，確認路徑後再用 `preview=false` 寫入。
4. 呼叫 `validate_lifting_training_project`，重新檢查 schema 與目前資產是否存在。

預設輸出到 `training_projects/<project-name>/`，此資料夾已排除在 Git 外，避免公司資產路徑或內部訓練資訊意外提交。可用 `ISAACLAB_MCP_TRAINING_ROOT` 改到其他位置。

## Multi-Agent Protocol

Protocol v1 以 `EnvironmentContract`、`TrainingRunRecord`、`EvidenceBundle` 與 `SceneChangeRequest` 建立 Sim → Lab → Verification → Correction 的版本化交接。三個 validator 維持純 Python、唯讀；另有僅支援 Dofbot 的受限可見 runner，不讀寫 USD。完整規格位於 [`docs/MULTI_AGENT_PROTOCOL.md`](docs/MULTI_AGENT_PROTOCOL.md)，執行順序見 [`docs/VERTICAL_TRAINING_WORKFLOW.md`](docs/VERTICAL_TRAINING_WORKFLOW.md)。

釣具吊升輸入範例位於 [`examples/fishing_tackle_lift_request.json`](examples/fishing_tackle_lift_request.json)。
完整設計原理見 [`docs/TRAINING_DESIGN_LOGIC.md`](docs/TRAINING_DESIGN_LOGIC.md)。

## Dofbot cube-lift runtime task

目前 Dofbot runtime 分成 `Isaac-Grasp-Cube-Dofbot-v0` 與 `Isaac-Lift-Cube-Dofbot-v0`。它使用 Isaac Sim 6.0.1 預設資產 `Robots/Yahboom/Dofbot/dofbot.usd`，控制 4 個 arm joints、wrist twist 與二元 gripper action，並在場景中加入 DexCube。

訓練先讓 policy 反覆練習夾取；掉落時只重設失敗的環境並立刻再試。通過獨立夾取門檻後，才接續訓練吊起。資產、reward、retry 邏輯與目前驗證結果見 [`docs/DOFBOT_CUBE_LIFT.md`](docs/DOFBOT_CUBE_LIFT.md)。

啟動可見的單機器手臂動作演示：

```powershell
& 'C:\isaacsim\python.bat' '.\scripts\demo_dofbot_lift.py' `
  --viz kit --device cuda:0 `
  --experience '.\apps\isaaclab.dofbot.demo.kit' `
  --duration 300 `
  --status-file '.\artifacts\dofbot_demo_status.json' `
  --capture-dir '.\artifacts\dofbot_viewport_motion'
```

這個專用 Kit experience 固定使用單張 GPU，並避免 geometry streaming 讓 Fabric articulation transform 停留在舊畫面。演示會讓 arm、wrist 與夾爪週期運動，並輸出四張 viewport capture 供畫面層驗證。

驗證 reset、step 與夾爪開合：

```powershell
& 'C:\isaacsim\python.bat' '.\scripts\validate_dofbot_lift_task.py' `
  --device cuda:0 --num-envs 2 --settle-steps 20 `
  --output '.\artifacts\dofbot_lift_validation.json'
```

執行可見的 grasp pre-training。專案 launcher 會固定啟用 Isaac Lab 內建 `RecordVideo`，每次 run 的 MP4 儲存在 `logs/rsl_rl/<experiment>/<run>/videos/train/`：

```powershell
& '.\scripts\train_dofbot_with_video.ps1' `
  -Task Isaac-Grasp-Cube-Dofbot-v0 `
  -NumEnvs 32 -MaxIterations 40 `
  -RunName safe_approach_grasp_visible
```

預設每 1000 environment steps 錄製 750 frames；已驗證的 50 FPS 輸出長度為 15 秒。`-VideoLength` 限制在 500–1000 frames，也就是 10–20 秒。launcher 會拒絕錄影彼此重疊、訓練步數不足，以及 `--headless`。

目前通過嚴格獨立評估的是 stage-one `model_39.pt`：4/4 environments 穩定夾取至少 0.5 秒，最長 6.78 秒。stage-two lift checkpoint 尚未通過，不能宣稱已完成吊起。

## G1 people-interaction RL

G1 跑步沿用 Isaac Lab 內建 PPO task。G1 拿取 task 目前是 teleop／Robomimic 路線且沒有 reward；專案自有的 chase task 會同步移動人物外觀與離地 kinematic collision capsule，只有人物 capsule 的 contact force 達 `1 N` 才算成功。各技能先分開訓練，再由高階 selector 組合。能力矩陣、課程順序與可見視窗錄影指令見 [`docs/PEOPLE_RL.md`](docs/PEOPLE_RL.md)。

目前最佳 chase checkpoint 為 `model_1411.pt`：64 environments 的最後一批接觸成功率為 `98.4375%`，同批方向跌倒率為 `1.5625%`、高度跌倒率為 `0%`；15 秒可見回放也確認碰觸後會重置並開始下一回合。

## TM6S retained reach proxy

`Isaac-Reach-TM6S-Lift-Proxy-v0` 直接載入 `E:\CMC\mod\TMroboot\tm6s.usd`，控制 6 個 arm joints，讓 `tool0` 追蹤高處目標。可用 `TM6S_USD_PATH` 覆寫資產位置。

這個 task 保留作為第一階段 reach proxy。原始 USD 沒有夾爪，而且 `link_6`、`flange_link`、`tool0` 沒有 collision prim，因此它不代表已完成物件抓取或接觸吊升驗證。詳細證據與限制見 [`docs/TM6S_LIFT_PROXY.md`](docs/TM6S_LIFT_PROXY.md)。

先驗證 reset 與 step：

```powershell
& 'C:\isaacsim\python.bat' '.\scripts\validate_tm6s_task.py' `
  --device cuda:0 --steps 2 --num-envs 2 `
  --output '.\artifacts\tm6s_task_validation.json'
```

執行可產生完整 10 秒影片的 PPO 錄影短跑：

```powershell
& 'C:\isaacsim\python.bat' `
  'D:\IsaacLab\scripts\reinforcement_learning\rsl_rl\train.py' `
  --task Isaac-Reach-TM6S-Lift-Proxy-v0 `
  --external_callback isaaclab_mcp.runtime_tasks.register_tasks `
  --num_envs 4 --max_iterations 32 --device cuda:0 --seed 42 `
  --video --video_length 500 --video_interval 1000 `
  --viz kit
```

## 安裝

```powershell
Set-Location 'F:\IsaacLab-MCP'
uv sync --dev
& '.\scripts\check_environment.ps1'
```

## 啟動

本機 stdio：

```powershell
& '.\scripts\run_mcp_server.ps1'
```

Codex 或其他 MCP client 可使用以下設定：

```json
{
  "mcpServers": {
    "isaac-lab-mcp": {
      "command": "F:\\IsaacLab-MCP\\.venv\\Scripts\\isaaclab-mcp-server.exe",
      "env": {
        "ISAACLAB_PATH": "D:\\IsaacLab"
      }
    }
  }
}
```

可選的 Streamable HTTP：

```powershell
$env:ISAACLAB_MCP_TRANSPORT = 'streamable-http'
$env:ISAACLAB_MCP_HTTP_HOST = '127.0.0.1'
$env:ISAACLAB_MCP_HTTP_PORT = '8010'
& '.\scripts\run_mcp_server.ps1'
```

## 驗證

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
uv build
```

## 開發邊界

- 不修改 `D:\IsaacLab` 的 source tree。
- 不把 API keys、`.env`、MCP client 設定、training logs 或 checkpoints 納入 Git。
- MCP job-control 目前只支援兩個 Dofbot task：可提交、監看與取消；G1、TM6S 與新公司資產仍需明確新增 allow-listed runner。
- 專案已包含一個可由 Isaac Lab CLI 載入的 TM6S runtime smoke task；它與 MCP job-control 能力分開。
