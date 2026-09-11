---
name: isaaclab-joint-visual-sync
description: Diagnose and fix Isaac Sim or Isaac Lab cases where actions and joint-state values change but an articulated robot stays visually frozen in the RTX viewport. Use for Fabric, Hydra, geometry-streaming, scene-delegate, or GPU synchronization problems; do not use when joint states themselves are static or when the request is only about RL convergence.
---

# Isaac Lab joint and viewport synchronization

找出停止更新的層級，修正後同時提供數值與可見畫面證據。不要把 joint telemetry 變化直接當成模型已經動作。

## 為什麼 joint 有移動，模型仍不會動

Isaac Lab 的控制與顯示會經過不同層級：

1. policy 或測試腳本產生 action。
2. actuator 將 action 轉成 joint position、velocity 或 effort target。
3. PhysX 更新 articulation joint state 與 rigid-body pose。
4. Fabric 或 USD 將 PhysX pose 提供給 Hydra scene delegate。
5. RTX viewport 繪製新的 link transforms。

畫面上的 joint 數值、graph 點位或 `robot.data.joint_pos` 只證明前幾層有更新。若 PhysX state 已變，但 Fabric、USD scene delegate、geometry streaming 或 RTX viewport 仍讀取舊 transform，數值會移動，機器人外觀會停在原姿勢。

這種狀況和 RL reward 設計是兩條不同問題線。先修正 articulation 到 viewport 的同步，再評估 reward、success rate 或 policy convergence。

## 先定位停止更新的層級

保留現有 USD、訓練輸出與 Git dirty files。先用 1 個 environment、固定相機和明顯但符合 joint limits 的週期 action 重現問題。

每個相位至少記錄：

- 實際送出的 action。
- `robot.data.joint_pos`，不要只記 target。
- 每個受控 joint 在整個週期內的 `max - min`。
- PhysX body pose 或 end-effector pose。
- 同一固定相機的 viewport capture。

依證據分流：

- action 沒變：檢查 policy、action sampling 或測試程式。
- action 有變但 joint state 沒變：檢查 actuator mapping、joint name、limits、drive stiffness、simulation stepping 與 reset。
- joint state 與 body pose 有變但 viewport 靜止：檢查 Fabric、scene delegate、geometry streaming、Kit experience 與 GPU 選擇。
- viewport 會動但物件沒有被抓起：轉查 gripper contact、collision、mass、friction、reward 與 termination。

## 處理 viewport transform 不同步

對這個專案的 Isaac Sim 6.0.1 與 Isaac Lab 3.0.0：

1. 需要 Fabric renderer 時，保持 `cfg.sim.use_fabric = True`。關閉 Fabric 可能讓 PhysX joint state 繼續更新，但 Fabric-based viewport 失去動態 transform 來源。
2. 使用明確 GPU ordinal，例如 `--device cuda:0` 與 `--/physics/cudaDevice=0`。不要用 `/physics/cudaDevice=-1`。多 GPU 自動選擇可能讓 physics 與 renderer 位於不同 context。
3. 可見驗證先使用單 GPU：`--/renderer/multiGpu/enabled=false`。
4. 檢查 Kit log 是否包含下列訊息：

   ```text
   /rtx/hydra/readTransformsFromFabricInRenderDelegate and geometry streaming are enabled together
   ```

   這代表 render delegate 與 geometry streaming 的組合可能讓動態物件停留在舊 transform。warning 本身不能證明畫面一定失敗，仍要以 viewport capture 判定。
5. 需要固定可視化設定時，建立專用、可自行解析 dependency 的 Kit experience。若 experience 放在 Isaac Lab checkout 外，確認 `[settings.app.exts].folders` 能找到實際的 Isaac Lab `source` 資料夾。
6. 修改 `UJITSO.geometry`、`app.useFabricSceneDelegate` 或 `rtx.hydra.readTransformsFromFabricInRenderDelegate` 後必須重新啟動 Kit。不要只從設定文字宣告成功。

本專案已驗證的 Dofbot 路徑：

- Demo：[`scripts/demo_dofbot_lift.py`](../../../scripts/demo_dofbot_lift.py)
- Kit experience：[`apps/isaaclab.dofbot.demo.kit`](../../../apps/isaaclab.dofbot.demo.kit)
- Runtime 說明：[`docs/DOFBOT_CUBE_LIFT.md`](../../../docs/DOFBOT_CUBE_LIFT.md)

這組 experience 使用單 GPU、固定 Physics CUDA ordinal、Fabric transforms，並在本機設定 `UJITSO.geometry = false`。把這組設定當成本專案的已驗證基準；其他 Isaac Sim 版本或資產仍需重新驗證。

## 可見驗證標準

至少擷取一個完整動作週期內的四個相位。相機、解析度、燈光與場景保持相同，比對 robot links 的姿態，不要依賴 UI 數字、selection highlight、時間或其他會變動的畫面元素。

通過條件：

- action 與實際 joint state 都有非零活動範圍。
- 四張 viewport captures 至少呈現兩個清楚不同的 link poses。
- status file 持續增加 simulation step，沒有 reset 將姿態立即蓋回。
- 延長執行後正常結束，沒有 `force_update`、`PhysXGpu` 或 teardown crash。
- 記錄使用的 Isaac Sim、Isaac Lab、GPU ordinal、Kit experience 與殘留 warning。

若桌面被其他視窗遮住，使用 Kit 內部 viewport capture。桌面錄影只能證明螢幕當下顯示的內容，無法可靠證明被遮住的 Isaac Sim viewport。

## 不要誤判 RL 成果

正弦 joint action 或手動 open/close 演示只驗證以下能力：

- USD articulation 可以載入。
- joint mapping 與 actuator command 有效。
- PhysX state 會更新。
- rendered links 能跟隨 articulation transforms。

一次 PPO iteration 只驗證 rollout、reward 計算、backpropagation 與 checkpoint 流程能執行。要宣告學會抓取或吊升，另外量測完整 episode 的 success rate、物件離地高度、目標追蹤、掉落率與獨立 evaluation。
