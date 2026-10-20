# handle_basket 提交包（2026-10-10）

这是 `handle_basket` 的独立 ACT 提交包，包含策略适配器、配置、启动器、完整 checkpoint、来源记录和 SHA-256 校验文件，无需训练数据或历史评测目录即可加载模型。

## 模型

- 源 checkpoint：`outputs/act_handle_basket_mixed1020_bs8_v1/checkpoints/080000/pretrained_model`
- 训练步数：80,000；训练 batch size：8；ACT `chunk_size=50`、`n_action_steps=50`
- 输入：三路 RGB（`cam_high`、左右腕部相机）以及机器人 `qpos/qvel/qf`
- 输出：14 维双臂绝对关节动作；夹爪索引为 6、13，并映射到环境物理范围
- 策略只读取公开观测，不读取物体真值位姿、任务内部状态或随机种子

checkpoint 下的 `config.json`、`model.safetensors`、`train_config.json` 均从上述目录原样复制。

## 运行

先按照官方项目安装 RoboSynChallenge 和 ACT/LeRobot 运行环境。在本提交包根目录执行：

```bash
ROBOSYNCHALLENGE_ROOT=/path/to/official/RoboSynChallenge \
PYTHON_BIN=/path/to/python \
bash policy/act_handle_basket_submission_v1/eval.sh 0 \
  --max_episodes 5 --seed 0 --headless True
```

确认环境正常后，将 `--max_episodes` 改为 `100` 进行 random100。启动器调用官方 `scripts/eval_policy.py`，不会修改 evaluator、任务源码、随机配置、动作配置或成功判据。

官方部署接口要求策略提供 `eval.sh`、`deploy_policy.yml` 和 `deploy_policy.py`；本包同时提供入口兼容层和推理计时。默认返回当前 evaluator 使用的四值结果 `(observation, info, truncated, inference_times_s)`。

## 已有评测证据

历史正式 random100（seed 0）结果为 **90/100**：90 个 `*_success.mp4`、10 个 `*_fail.mp4`、0 个临时视频。结果批次：

`eval_result/final_official_random100_20260907/handle_basket/act_observation_only_final_20260907/random/official_random100_seed0/080000/2026-09-07_23-33-30/videos`

该成绩是本地公开 random 配置的历史结果，不应表述为隐藏测试或真实机器人最终成绩。当前打包过程只做静态和完整性验证，未重新启动 GPU random100。

## 官网提交前

本目录是可上传/发布的本地制品，但比赛表单仍可能需要可访问的 GitHub 代码链接、Hugging Face 权重链接、模型方法、数据来源和运行依赖。本次未代替用户上传代码、权重或提交官网表单。

参考：

- https://robosyn-bench.net/
- https://github.com/EDEM-AI/RoboSynChallenge/tree/main
- https://edem-ai.github.io/RoboSynChallenge/html/tutorials/policy/your_own_policy.html
- https://edem-ai.github.io/RoboSynChallenge/html/tutorials/policy/act.html
