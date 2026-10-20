# sample_loading 最终提交包

这是按其他任务格式整理后的 V89 提交候选，顶层只保留评测入口、权重、配置、证据和校验文件；递归历史依赖统一收在 `_runtime/`，不需要人工逐层处理。

## 成绩

- 正式公开观测候选：V89
- random20，master seed `20260929`，`eval_expert_check=true`
- 15/20 = 75%
- 平均动作步数：444.45
- 平均单次推理：0.040254911 秒

历史 V28 的 78/100 使用实时 cube/rack 真值，因此没有作为合规提交策略。V121 的 18/20 是训练专用真值评测，也没有作为正式成绩。

## 目录

```text
sample_loading/
├── __init__.py
├── deploy_policy.py
├── deploy_policy.yml
├── eval.sh
├── checkpoint/          # 4 个运行权重
├── config/              # 抓取校准
├── _runtime/            # 内部策略、网络结构和轨迹库
├── evidence/            # 20 局指标、日志、CSV、视频
├── LICENSE
├── requirements-runtime.txt
├── submission_manifest.json
├── MANIFEST.txt
├── SHA256SUMS.txt
└── PACKAGE_VERIFY.log
```

## 运行

```bash
export ROBOSYNCHALLENGE_ROOT=/absolute/path/to/RoboSynChallenge
bash eval.sh sample_loading random 0
```

附加 evaluator 参数直接放在后面，例如：

```bash
bash eval.sh sample_loading random 0 --max_episodes 1 --eval_video_log false
```

## 官方申报字段

- 输入：joints、gripper、cam_high、cam_left_wrist、cam_right_wrist
- 输出：绝对 joints + gripper，14 维
- 不使用：实时 cube pose、rack pose、深度、触觉、外部 API
- 数据：仿真示范、筛选/自生成仿真轨迹、视觉位姿回归模型

## 提交前唯一需要确认的事项

V89 不读取实时物体真值，但会通过 `env.unwrapped.robot` 使用机器人 FK/IK 模型。由于现有官方书面字段没有明确说明这项访问是否允许，正式 ranked 上传前应让 Help Desk 书面确认。记录成绩所用仓库 commit 为 `7503c7cb000bdf869cdba302e329724b3a223c75`，也应在当前官方 HEAD 上复跑。
