# 待完成事项（2026-10-11）

详细、可勾选的执行清单见 `OFFICIAL_SUBMISSION_CHECKLIST.md`。目前只剩以下必须由队伍或主办方完成的外部事项：

1. **今天优先：**确认 2026-10-11 预赛截止的准确时间；取得并验证官网邮箱/access token。
2. 填写真实姓名、单位、邮箱、队名、GitHub 代码链接、Hugging Face 权重链接和 leaderboard 选择。没有代为上传或提交。
3. 向主办方确认十项 task-specific controller 应作为一个还是多个模型/评测条目；解决六项 ACT 的真实 `50/50` chunk 与官网强制 `execution < output` 的冲突。
4. sample_loading：获得主办方对 `env.unwrapped.robot` FK/IK 访问的书面确认，并确认非 ACT controller 的 chunk 字段填法。
5. 在官方 Linux/CUDA 环境逐任务运行统一入口，并填写 `OFFICIAL_RUNTIME_VALIDATION.json`。当前仅完成静态、路径、资产哈希和下载恢复校验。
6. 补充 DATA_SOURCES.md 中未证实的数据数量、版本、额外数据采集和授权来源；在官网准确勾选 official real / official simulated / other。
7. 确认官方观测是否提供 qvel/qf；若不提供，需逐任务适配和重新验证，不能静默置零或冒称已兼容。
8. 选择根级代码许可证并核查第三方代码、权重、校准文件和运行轨迹库的再发布权限。当前说明见 `LICENSE_REQUIRED.md`。
9. 旧 README/验证记录属于历史版本，以 README.md、TASKS.json、PORTAL_FIELDS.json 和各任务 SUBMISSION.md 为本次依据。E盘根目录已有的“最终提交.zip”未更新，不代表此目录当前版本。

缺少新的 GPU 记录不等于官方要求必须上传本地 random100；它表示本次尚不能保证真实环境运行成功。正式成绩由主办方隐藏评测产生。
