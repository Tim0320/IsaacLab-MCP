# 吊升任務的 RL 設計邏輯

這份文件說明 `design_lifting_training` 如何把「讓機構自行吊起物件」拆成 Isaac Lab 可以逐步實作與驗證的訓練資訊。

## 1. 先定義成功

訓練不能只寫「把物件吊起來」。可量測的成功條件是：

- payload 到達目標高度或目標位置；
- 位置誤差小於 0.03 m；
- 移動速度小於 0.05 m/s；
- 連續穩定 0.5 s；
- 過程中沒有掉落、超載或進入禁止碰撞區。

這避免 policy 只讓 payload 快速掠過目標，也能把 `success_rate` 變成可比較的評估指標。

## 2. 資產決定模擬能否成立

至少需要三類資產：

1. Lifting mechanism：有 rigid bodies、collision、mass、inertia、可控制 joint 與 joint limits 的 USD 或 URDF。
2. Hook/end effector：機構中可識別的 body 與 attachment frame。
3. Payload：有 collision、真實質量、center of mass、inertia 與 attachment point 的 rigid-body USD。

Environment asset 可稍後加入，但正式安全訓練需要地面、障礙物與禁止區域的 collision geometry。

檔案存在只代表第一層檢查通過。真正執行前仍需在 Isaac Lab runtime 裡 read back prim、joint、body、質量與 collision。

## 3. MDP 的五個部分

每一個 simulation step 都遵循以下循環：

1. Observation：policy 看見 joint position/velocity、payload 到 target 的向量、payload 速度、hook 距離與上一個 action。
2. Action：policy 輸出 `[-1, 1]`，再映射成有速度上限的 lift command。
3. Physics：Isaac Lab 更新機構、payload、接觸與重力。
4. Reward：根據接近目標、上升進度、穩定程度、安全與耗能計分。
5. Termination：成功、掉落、超載、危險碰撞或 timeout 時結束 episode。

policy 用 PPO 重複收集許多 episodes，讓能得到較高累積 reward 的動作逐漸更常出現。

## 4. Reward 為什麼要拆開

只有成功獎勵時，初期幾乎所有 episode 都拿不到訊號。設計因此同時使用：

- Dense reward：`target_distance` 與 `upward_progress` 告訴 policy 哪個方向比較好。
- Stability reward：`stable_payload` 抑制晃動。
- Sparse bonus：`success_bonus` 強調真正完成任務。
- Smoothness penalty：`action_rate` 避免馬達命令跳動。
- Efficiency penalty：`energy` 降低不必要輸出。
- Safety penalties：`unsafe_collision`、`overload`、`dropped_load` 給高代價並終止 episode。

權重只是起始值。後續要看各 reward term 的實際量級、成功率、掉落率與動作曲線再調整，不能只看總 reward。

## 5. 為什麼分三階段

繩索、魚線或 hook contact 會增加柔性體、接觸不連續、擺動與數值穩定性問題。直接從完整 cable physics 開始通常難以分辨是 policy 沒學會，還是物理與資產設定不穩。

預設 curriculum：

1. Fixed attachment：payload 固定在 hook，先學垂直控制與煞停。
2. Compliant attachment：加入擺動、質量變化與小幅柔順性。
3. Contact or cable：前兩階段穩定後，再換成經驗證的 hook contact 或 cable dynamics。

每階段以 `success_rate`、`drop_rate`、`unsafe_collision_rate` 等門檻決定是否前進。

## 6. Domain randomization 的用途

訓練時小幅改變 payload mass、joint friction、motor strength、初始位置與 sensor noise，避免 policy 只記住一組完美模擬參數。隨機範圍必須來自設備量測或合理公差；範圍過大會讓任務變得不可學。

## 7. Readiness gate

- `design_ready`：MDP、PPO、curriculum 與 metrics 已完整描述。
- `asset_ready`：真實資產、joint/body 名稱與安全工作負載已確認。
- `training_ready`：前兩項都成立。

目前 generator 產生的是 training design packet。它不會自行產生可信的公司設備幾何、猜測 joint 名稱，也不會把模擬結果當成實機安全證明。
