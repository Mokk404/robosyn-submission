# 官网填写说明

`SUBMISSION_FORM.json` 为统一草稿，尚未注册或提交。Artifact/Experiment 名称是本次整理建议值，可调整；参赛人姓名、单位、邮箱、队伍名称、GitHub、Hugging Face 链接和 leaderboard 选择必须由真实信息填写。当前官网完整字段及任务协议对照见 `PORTAL_FIELDS.json`。

当前官网表单本身没有任务选择字段，但是否允许一个模型条目包含十套 task-specific controller、以及如何填写不同 execution chunk，仍需主办方确认。同一 GitHub 仓库可收录代码；Hugging Face 权重必须按 ASSETS.json 保留相对路径，README 列明对应关系。不要填写官方基线仓库作为自己的提交仓库。

六项 ACT 配置真实使用 `output_chunk_size=50`、`execution_chunk_size=50`，而官网表单拒绝 execution 大于或等于 output。不得为了通过表单擅自改成 49。请先取得主办方填报指示，详见 `OFFICIAL_SUBMISSION_CHECKLIST.md`。

官网模型登记技术上要求 Hugging Face 或 GitHub 至少一个有效链接；本候选包为了可复现性仍计划同时提供代码仓和权重仓。私有仓库的官方访问账号/方式未从当前官网源码得到确认，必须向主办方核实。

检查了本项目相关对话的近期记录，只找到官方仓库/参考权重链接，未确认用户本次提交链接或队名，因此保留空值。

来源：https://robosyn-bench.net/ 以及当前官方站点仓库中的 `templates/evaluation.html`、`templates/models.html` 和 `app.py`。实际提交字段以当时官网为准。
