# Multi-Agent Protocol v1

這份 Protocol 讓 Isaac Sim、Isaac Lab 與 Verification Agent 用可追蹤的文件交接，而不是用自然語言猜測資產、joint 或 checkpoint 的來源。

## 四份文件

| 文件 | 方向 | 回答的問題 |
| --- | --- | --- |
| `EnvironmentContract` | Sim → Lab | 這個已驗證的世界有哪些 asset、joint、sensor、physics 與限制？ |
| `TrainingRunRecord` | Lab → Verification/Audit | 這次對哪個 contract、用何種 config、訓練出什麼 checkpoint？ |
| `EvidenceBundle` | Verification → Orchestrator/Audit | 哪個 environment、run、checkpoint 是否通過每一項 criteria？ |
| `SceneChangeRequest` | Lab → Orchestrator → Sim | Lab 觀察到什麼事實、推測什麼原因、需要哪一種新能力？ |

每份文件都有 `schema_version`、`provenance` 與 deterministic SHA-256 fingerprint。相同內容一定會產生相同 fingerprint；任何欄位改變都需要新的版本。

## Environment Contract 的界線

`EnvironmentContract` 只描述由 Isaac Sim Engineer 驗證過的世界事實：USD／asset hash、articulation root、joint limit/control mode、sensor、physics/control timestep、workspace、已知限制與 Sim evidence。

它不能包含 PPO、reward 權重或 Lab Agent 推測。Lab 端的 Training Specification 必須引用 `environment_id`、`contract_version` 與 `contract_fingerprint`，不能自行猜測 joint 名稱。

## Evidence 與修正

`EvidenceBundle` 必須同時引用 structured、visual、behavioral 與 training evidence，並鎖定唯一的 environment version、training run 與 checkpoint。`PASS` 只有在所有 criteria 都是 `PASS` 時才可使用。

`SceneChangeRequest` 必須分開填寫：

- `observed_evidence`：可量測的事實與 artifact。
- `suspected_cause`：Lab Agent 的假設，不能當成事實。
- `requested_capability`：需要的能力，不可直接指定 USD 或 joint 的修改數值。

因此 Lab Agent 能提出「目標必須在不超出 rail limit 的情況下可達」，但不能直接把 rail 改成某個長度。真正的 scene decision 仍由 Isaac Sim Engineer 負責。

## MCP validation tools

目前是純 Python、唯讀的 validator：

- `validate_environment_contract`
- `validate_evidence_bundle`
- `validate_scene_change_request`

它們不啟動 Kit、不讀取或修改 USD、不寫檔，也不啟動 training job。下一個 execution milestone 才會讓 training job 產生 `TrainingRunRecord`。
