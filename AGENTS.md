# IsaacLab-MCP workspace instructions

- Treat `F:\IsaacLab-MCP` as the MCP project repository.
- Treat `D:\IsaacLab` as the configured Isaac Lab runtime checkout. Inspect it before use and do not modify it unless the user explicitly asks.
- Run Isaac Lab or Kit-bound Python through `D:\IsaacLab\isaaclab.bat -p`. Do not import `omni`, `pxr`, or `isaaclab` with a generic Python runtime.
- Keep credentials, local MCP client files, training logs, checkpoints, and generated outputs out of Git.
- Record Git status before and after changes. Do not commit or push unless the user explicitly requests it.
- Keep Isaac Sim documentation MCP, Isaac Sim live control, and this Isaac Lab MCP server as distinct routes.
- Run every Dofbot grasp or lift training through `scripts\train_dofbot_with_video.ps1`. It keeps the Kit window visible, enables Isaac Lab's built-in `RecordVideo`, verifies that an MP4 was produced, and reports the run's `videos\train` directory. Do not use `--headless` for these training runs.
