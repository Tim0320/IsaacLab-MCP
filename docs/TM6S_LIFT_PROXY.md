# TM6S lift-proxy runtime validation

## 驗證範圍

資產：`E:\CMC\mod\TMroboot\tm6s.usd`

SHA-256：`e92de4e018cf0d9b213d6594066b49f4b8c1ee69b659dba5de90c6c6e73b137a`

本 task 驗證 TM6S 的 Isaac Lab articulation、6 維 joint-position action、reward、observation 與 RSL-RL PPO 串接。原始 USD 全程唯讀。

## 資產檢查結果

- Default prim：`/root`
- Articulation root：`/root/body`
- 單位：1 stage unit = 1 meter，Z-up
- 可控制關節：`shoulder_1_joint`、`shoulder_2_joint`、`elbow_joint`、`wrist_1_joint`、`wrist_2_joint`、`wrist_3_joint`
- 末端 body：`tool0`
- Collision prim 只存在於 `link_0` 到 `link_5`
- 沒有 gripper joint，也沒有 `link_6`、`flange_link` 或 `tool0` 的碰撞幾何

因此目前 task 使用高處 end-effector pose tracking 作為 lift proxy。加入有碰撞幾何的夾爪或吸盤、payload 與 grasp/contact 驗證後，才能升級成物件吊升 task。

## RL 定義

- Action：6 個 joint-position commands，每個 policy output 乘上 `0.25 rad` 後加到 default joint pose。
- Observation：6 維 joint position、6 維 joint velocity、7 維 target pose、6 維 previous action，共 25 維。
- Reward：末端位置誤差、細緻位置追蹤、姿態誤差、action-rate penalty、joint-velocity penalty。
- Command：`tool0` 在 x `[0.25, 0.55] m`、y `[-0.35, 0.10] m`、z `[0.55, 1.10] m` 內追蹤目標。
- Termination：6 秒 episode timeout。

## 2026-09-07 執行證據

Runtime：`C:\isaacsim\python.bat`，Isaac Sim 6.0.1，Isaac Lab 3.0.0，device `cuda:0`。

Reset/step validation：2 個環境執行 2 steps，action shape `[2, 6]`，policy observation shape `[2, 25]`，所有 observation 為有限值，沒有 termination 或 truncation。

RSL-RL smoke run：4 個環境、1 iteration、32 total steps、seed 42，training time 6.2 秒，產生 `model_0.pt`。該次結果為 mean reward `-0.04`、position error `0.8455 m`、success rate `0.0`。

這次結果只證明程式碼可執行並完成一次 optimizer update。1 iteration 不足以評估收斂、穩定性、泛化、抓取成功率或實機安全性。
