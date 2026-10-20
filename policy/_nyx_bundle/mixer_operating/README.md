# mixer_operating 提交包（2026-10-10）

包含独立策略、原始权重、配置、启动器和哈希。无需训练目录或旧评测包装脚本。

## 模型与输入输出

源 checkpoint：`outputs/mixer_resume80k_to160k_20260907_160531_lM7Gld/train/checkpoints/100000/pretrained_model`。原始 config.json/model.safetensors 字节保持不变。
输入由 checkpoint 的 input_features 定义：三路 RGB 与机器人状态；适配器支持 qpos/qvel/qf，不读取物体真实位姿或私有随机种子。
输出14维绝对关节动作，夹爪索引6、13。Table仅裁剪物理量；mixer保留原适配器的比例映射，不套用Table修正。

## 运行

须先安装官方环境及LeRobot 0.3.3。本包不会修改官方evaluator、任务、随机事件、动作配置或成功判据。

```bash
ROBOSYNCHALLENGE_ROOT=/path/to/official/RoboSynChallenge \
PYTHON_BIN=/path/to/python \
bash policy/act_mixer_submission_v1/eval.sh 0 --max_episodes 5 --seed 51001
```

在本包根目录执行，或把policy/act_mixer_submission_v1复制进官方仓库。随后将5改为100执行random100。所有命令默认random；结果位置由官方评测器管理，可通过其支持的eval_result_dir覆盖。
默认四值返回，匹配已核对的2026-09-20官方快照。Table可传 `--return_inference_timing False` 兼容旧三值入口；mixer只支持四值入口。
官方环境需要包含checkpoint声明的观测字段；如提供的API不同，先做兼容性核对。

## 证据与状态

100000保留基线。历史463/500及后续88/100等结果使用历史设置（含冻结区间随机事件），不能标为当前官方random成绩；lr2e-6的120000仅单批90/100，未据此替换基线。
完成静态语法、包内依赖路径、权重哈希校验；Table包含6项CPU动作/接口回归测试。没有启动GPU评测，没有上传GitHub/Hugging Face或进行官网提交。

## 官网提交前需填写

提供GitHub代码链接、Hugging Face权重链接、数据来源、运行依赖及模型方法说明。不能用本地路径替代公开或授权可访问的链接。主办方实际表单字段以提交时网站为准。
来源：https://robosyn-bench.net/#/faq ，https://edem-ai.github.io/RoboSynChallenge/html/tutorials/policy/your_own_policy.html
