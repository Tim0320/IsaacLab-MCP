# Dofbot cube-lift runtime validation

## 使用資產

Isaac Sim 6.0.1 預設資產：

`https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/Robots/Yahboom/Dofbot/dofbot.usd`

2026-09-08 下載供唯讀檢查的 USD snapshot SHA-256：`52c524ebb26c38a3d164daee10f6cac0f15487fce5408a38c0c94199a37f1303`。

## 資產檢查

- Articulation root：`/arm`
- 單位：1 stage unit = 1 meter，Z-up
- Arm：`joint1` 到 `joint4`
- Wrist：`Wrist_Twist_RevoluteJoint`
- 主動 gripper joints：`Finger_Left_01_RevoluteJoint`、`Finger_Right_01_RevoluteJoint`
- 被動 finger linkage joints：4
- Rigid bodies：12
- Collision prims：Dofbot 本體 12 個，另含資產內的 ground plane

左右主動 finger joints 的 close command 分別為 `-0.50 rad` 與 `+0.50 rad`。20 個 settle steps 的指尖距離量測如下：

- Open：`0.060324 m`
- Closed：`0.041756 m`
- Reopened：`0.058062 m`

因此已驗證 gripper command 方向與 PhysX 開合行為。

## 可見畫面同步驗證

`scripts/demo_dofbot_lift.py` 使用 `apps/isaaclab.dofbot.demo.kit` 啟動單 GPU、Fabric-enabled 的互動視窗。預設控制週期會對 5 個 arm/wrist joints 發出正弦 joint-position action，並每 100 steps 切換夾爪開合。

不能只用 joint position 判定畫面正確：PhysX joint state 可能已更新，但 RTX viewport 仍讀到舊的 articulation transforms。這個專用 experience 隔離可視化設定、關閉 geometry streaming，演示腳本另外在 steps 90、190、290、390 擷取 viewport。四張影像必須出現不同連桿姿態，才算完成畫面層驗證。

```powershell
& 'C:\isaacsim\python.bat' '.\scripts\demo_dofbot_lift.py' `
  --viz kit --device cuda:0 `
  --experience '.\apps\isaaclab.dofbot.demo.kit' `
  --duration 300 `
  --status-file '.\artifacts\dofbot_demo_status.json' `
  --capture-dir '.\artifacts\dofbot_viewport_motion'
```

## 兩階段 RL 定義

第一階段 `Isaac-Grasp-Cube-Dofbot-v0` 只練習接近、在正確距離閉爪，以及維持方塊位於兩指之間。第二階段 `Isaac-Lift-Cube-Dofbot-v0` 才加入離地、吊升目標與姿勢引導。兩者共用相同的 39 維 observation 與 6 維 action，checkpoint 可以接續使用。

- Object：Isaac Sim DexCube，scale `0.4`，質量覆寫為 `0.03 kg`
- Action：5 維 arm/wrist joint-position action，加 1 維 binary gripper action
- Observation：11 維 joint position、11 維 joint velocity、3 維指尖中點到方塊的位移、7 維 target pose、6 維 previous action、1 維 fingertip gap
- Grasp reward：指尖接近、正確閉爪時機、穩定夾取、掉落、頻繁開合懲罰
- Lift reward：保留 grasp reward，再加入持物高度進度、已驗證抬升姿勢、離地與目標追蹤
- Retry：方塊低於 `0.045 m` 時，只 reset 失敗的 vectorized environment；PPO 與其他環境繼續執行
- Episode：grasp 8 秒，lift 20 秒

## 可見訓練與獨立驗證

所有下列 runtime 都使用 Isaac Sim 可見 Kit 視窗、`cuda:0`，沒有使用 headless。

```powershell
& '.\scripts\train_dofbot_with_video.ps1' `
  -Task Isaac-Grasp-Cube-Dofbot-v0 `
  -NumEnvs 32 -MaxIterations 40 `
  -RunName safe_approach_grasp_visible
```

這個 launcher 固定加入 Isaac Lab `train.py` 的 `--video`、`--video_length` 與 `--video_interval`，使用內建 `gym.wrappers.RecordVideo`。預設每 1000 environment steps 錄製 750 frames；已驗證的 50 FPS 輸出長度為 15 秒。`VideoLength` 限制為 500–1000 frames，也就是 10–20 秒，輸出到：

`logs/rsl_rl/dofbot_cube_grasp_pretrain/<timestamp>_<run-name>/videos/train/*.mp4`

每次執行結束時，launcher 會列出實際 run、影片資料夾與所有 MP4；成功訓練卻沒有影片時會回傳失敗。訓練總 steps 不足以錄滿一段影片、錄影區段互相重疊或使用 `--headless` 時，也會在啟動前拒絕執行。

2026-09-09 的 stage-one `model_39.pt` 獨立 deterministic evaluation 結果：4/4 environments 連續穩定夾取至少 0.5 秒，最長 6.78 秒，close commands 有 99.2% 發生在方塊附近。證據位於 `artifacts/dofbot_safe_approach_model39_visible_evaluation.json` 與對應 PNG。

```powershell
& 'C:\isaacsim\python.bat' '.\scripts\evaluate_dofbot_grasp_policy.py' `
  --task-stage grasp --checkpoint '<model_39.pt>' `
  --num-envs 4 --steps 1000 `
  --video-length 750 `
  --video-dir '.\artifacts\dofbot_safe_approach_model39_video' `
  --output '.\artifacts\dofbot_safe_approach_model39_visible_evaluation.json' `
  --capture '.\artifacts\dofbot_safe_approach_model39_visible.png' `
  --viz kit --device cuda:0 `
  --experience '.\apps\isaaclab.dofbot.demo.kit' `
  --kit_args '--/renderer/multiGpu/enabled=false --/physics/cudaDevice=0'
```

第二階段 60-iteration guided lift 短跑尚未通過：獨立評估的最高方塊高度仍為 `0.065 m`，2 cm retained lift 為 0/4，且該 checkpoint 的穩定夾取為 0/4。因此目前認可的成果是「重複穩定夾取」，不是「已學會吊起」。
