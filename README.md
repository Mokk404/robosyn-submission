# RoboSynChallenge 十任务统一提交候选

更新日期：2026-10-11。当前目录为 NYX 上的完整候选，含十项策略、权重与辅助运行资产。尚未完成官网提交；剩余事项见 [PENDING.md](PENDING.md)。

## 统一运行

复制整个目录到官方 Linux 主机，先按 [ENVIRONMENT.md](ENVIRONMENT.md) 安装环境。从此目录执行：

```bash
export ROBOSYNCHALLENGE_ROOT=/absolute/path/to/official/RoboSynChallenge
export PYTHON_BIN=/absolute/path/to/python
# 如环境需要：export EMBODICHAIN_ROOT=/absolute/path/to/EmbodiChain
bash eval.sh --task table_rearrangement --gpu 0 --episodes 1 --seed 0 --dry-run
bash eval.sh --task table_rearrangement --gpu 0 --episodes 1 --seed 0
bash eval.sh --task table_rearrangement --gpu 0 --episodes 100 --seed 0
```

替换 --task 即可运行其余任务。每项也有 `run_submission.sh --gpu 0 --episodes 1 --seed 0`。未复制权重到官方源码目录的情况下也可运行；路径按包内位置解析。统一入口导入已有策略顶层导出，保留现有计时兼容层，不修改官方 evaluator、任务或随机化设置。

## 十项任务与权重映射

机器可读映射：[TASKS.json](TASKS.json)；完整权重/校准/轨迹资产及哈希：[ASSETS.json](ASSETS.json)。仅上传 ACT 主模型不足以覆盖 hybrid 策略，必须保留表中全部运行资产。

| 任务 | 方案 | 统一说明 |
|---|---|---|
| click_bell | ACT 080000; execution horizon 10 | [click_bell/SUBMISSION.md](click_bell/SUBMISSION.md) |
| drawer_open_place | Placement-weighted ACT 002500; qvel/qf replaced with frozen means | [drawer_open_place/SUBMISSION.md](drawer_open_place/SUBMISSION.md) |
| handle_basket | ACT mixed1020 080000 | [handle_basket/SUBMISSION.md](handle_basket/SUBMISSION.md) |
| item_assembly | ACT 080000 plus fixed bounded qpos residual; replan 20 | [item_assembly/SUBMISSION.md](item_assembly/SUBMISSION.md) |
| items_handover | Dual ACT, RGB translation predictor, fixed gripper schedule and RBF prior v60 | [items_handover/SUBMISSION.md](items_handover/SUBMISSION.md) |
| manipulate_pipette | Pure ACT current1500; execution horizon 50 | [manipulate_pipette/SUBMISSION.md](manipulate_pipette/SUBMISSION.md) |
| mixer_operating | ACT retained baseline 100000 | [mixer_operating/SUBMISSION.md](mixer_operating/SUBMISSION.md) |
| sample_loading | V89 RGB perception and public-observation controller; FK/IK confirmation pending | [sample_loading/SUBMISSION.md](sample_loading/SUBMISSION.md) |
| table_rearrangement | ACT 080000; physical gripper clipping without rescaling | [table_rearrangement/SUBMISSION.md](table_rearrangement/SUBMISSION.md) |
| water_pouring | ACT spatial-translation fine-tune 000200 | [water_pouring/SUBMISSION.md](water_pouring/SUBMISSION.md) |

## 本地检查

```bash
python -B verify_submission.py --integrity
python -B tests/test_runner.py
```

检查不加载模型、不运行模拟器，也不代表 GPU 验收通过。

## 提交材料

- [SUBMISSION_FORM.md](SUBMISSION_FORM.md)、[SUBMISSION_FORM.json](SUBMISSION_FORM.json)：名称、简介、链接及参赛表单字段。
- [PORTAL_FIELDS.json](PORTAL_FIELDS.json)：当前官网完整字段、建议值和逐任务 chunk 协议；六项 50/50 冲突不得静默改值。
- [OFFICIAL_SUBMISSION_CHECKLIST.md](OFFICIAL_SUBMISSION_CHECKLIST.md)：按截止、发布、表单、数据和运行验收整理的最终清单。
- [OFFICIAL_RUNTIME_VALIDATION.json](OFFICIAL_RUNTIME_VALIDATION.json)：官方 Linux/CUDA 验收记录模板。
- [ORGANIZER_CONTACT_DRAFT.md](ORGANIZER_CONTACT_DRAFT.md)：可直接修改后发送给主办方的关键问题邮件草稿。
- [DATA_SOURCES.md](DATA_SOURCES.md)、[DATA_PROVENANCE.json](DATA_PROVENANCE.json)：数据来源及可核实训练元信息。
- [ENVIRONMENT.md](ENVIRONMENT.md)：依赖和运行步骤。
- [VALIDATION_UNIFIED.json](VALIDATION_UNIFIED.json)：本次检查范围与结果。
- `SHA256SUMS_UNIFIED`：当前包文件校验；Linux 使用 `sha256sum -c SHA256SUMS_UNIFIED`。

当前官网评测表单没有任务选择字段，但是否允许一个模型登记包含十套 task-specific controller、以及任务间不同 chunk 的填法仍需主办方确认。提交自己的 GitHub 仓库与 Hugging Face 权重链接，不把官方代码/基线链接当成自己的制品。完整包可保留模型资产；发布代码仓库时，将 ASSETS.json 列出的资产放到 Hugging Face，并在运行前按原相对路径恢复：

```bash
python -m pip install huggingface_hub
python download_assets.py --repo-id ORG/REPO --revision PINNED_REVISION
python -B verify_submission.py --integrity
```

`.gitignore` 已将大型运行资产排除在普通 GitHub 历史之外；`download_assets.py --verify-only` 会按 ASSETS.json 校验恢复后的 396 项运行资产。sample_loading 的 `_runtime` 代码与轨迹资产、items_handover 的全部辅助资产必须保留。发布前还需要由队伍选择根级许可证，见 `LICENSE_REQUIRED.md`。

本次未上传、未发送消息给主办方、未启动GPU测评。历史成功率不是本次统一入口验收成绩；模型权重和已有推理控制逻辑均保留。备份和更改记录位于 _history/20261011_uniform 与 CHANGES_UNIFIED.json。

## 官方参考

- https://robosyn-bench.net/#/faq
- https://edem-ai.github.io/RoboSynChallenge/html/tutorials/policy/your_own_policy.html
- https://github.com/EDEM-AI/RoboSynChallenge/blob/main/scripts/eval_policy.py
