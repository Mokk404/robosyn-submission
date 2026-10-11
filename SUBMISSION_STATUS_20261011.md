# 当前统一提交状态

已增加十任务统一入口、任务映射、数据来源、统一说明及校验。进一步补充了官网完整字段草稿 `PORTAL_FIELDS.json`、GitHub/Hugging Face 分离发布规则、资产恢复与校验脚本、官方运行验收模板和逐项提交清单。请从 [README.md](README.md) 开始。

当前本地校验覆盖 10 项任务、642 个发布包哈希和 396 项 Hugging Face 运行资产，均通过。历史评测视频作为可选证据，不进入代码仓或发布包完整性清单。仍待队伍或主办方完成的外部链接、账号、数据声明、六项 50/50 chunk 表单冲突、sample_loading FK/IK 确认、许可证选择和 GPU 验证见 [PENDING.md](PENDING.md) 与 `OFFICIAL_SUBMISSION_CHECKLIST.md`。

本次保留所有原有权重和推理控制逻辑。旧状态记录已保存到 `_history/20261011_uniform/`。
