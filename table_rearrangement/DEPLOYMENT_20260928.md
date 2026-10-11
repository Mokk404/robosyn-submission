> 历史记录（2026-09-28）：下文描述更新前状态。2026-10-10 已补齐包内权重；当前运行方法见 README.md。旧 CHECKPOINT [GPU] 调用仍兼容。

# Table 提交策略安装记录

策略名：act_table_physical_v1。原始 080000 权重保持不变，不需要重训。只修改参与者策略，不修改官方评测器、成功判定、随机事件或环境动作配置。

## 官方快照运行（5 局验证）

```bash
cd ~/Robotsynchallenge/RoboSynChallenge_official_clean_20260920
PYTHON_BIN="$HOME/Robotsynchallenge/RoboSynChallenge/policy/act/.venv/bin/python" \
bash policy/act_table_physical_v1/eval.sh \
  "$HOME/Robotsynchallenge/RoboSynChallenge/outputs/act_table_rearrangement_mixed1100_bs8_v0/checkpoints/080000/pretrained_model" \
  0 --max_episodes 5 --seed 51001 --train_config_name table_physical_smoke_20260928
```

须按官方说明安装 RoboSynChallenge 和匹配的 EmbodiChain。上述命令引用此前 ACT 虚拟环境，未验证其满足新版模拟器全部依赖。默认四值返回与已读取的官方快照一致；旧主项目入口运行时增加 `--return_inference_timing False`。

原始 checkpoint 配置使用 robot/qpos、qvel、qf 和三路 RGB；本包仅从 observation 读取它们，不读取物体位姿、深度或私有种子。需确认正式环境提供这些机器人字段。

最终提交目录仅包含代码、文档和 manifest；权重仍引用项目原始 checkpoint，未复制进本包。对外提交需补充 Hugging Face 权重链接。

验证：6 项 CPU 单元测试通过。复制后校验 SHA256；未开展真实模型加载或 GPU 模拟评测。旧入口的 295/300 不作为本包验收成绩。
