"""Render the minimal plan; no device or Raw access."""

import csv
from html import escape
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Preserve issue numbers; most findings do not require production changes.
ISSUES = [
    ("01", "XHand tor_max 单位与范围", "确定的接口语义问题；当前 300/360 未越界。",
     "已完成局部修复",
     "配置及驱动已统一使用 tor_max，注明 Mode 3 的 0–400 输出尺度；原有 validate 已修正类型/范围。数值和增益不变，无转换层或旧名兼容层。",
     "默认值、边界/非法值、配置加载及序列化离线检查通过；已在四轮真机运行中使用，数值和增益未改变。接口修复不代表抓取性能已解决。"),
    ("02", "停止原因与任务结果", "停止、SDK 接收都不能代表成功；这是评价口径。",
     "手工记录即可",
     "复用 trial_plan.csv，填成功/失败/不确定、一句原因和 Raw 路径。不加任务 FSM，不改 EpisodeResults 或 Raw 字段。",
     "四轮均手动停止、Raw 发布和关闭正常，但任务均未完成。另一次只归位后退出、没有开始 episode，单独保留 session_result，不冒充第五次抓取。"),
    ("03", "等待行 NaN 与导出拒绝", "新 sync 有 41 行未发送；三轮 async 各仅 2 行启动等待。NaN 与未发送目标一致。",
     "不改代码",
     "撤回新增导出 JSON 汇总。保留 Raw 和完整性检查，接受 evaluation rollout 不一定可回训；不补零、不删帧。",
     "直接用现有日志；以后确需回训此类数据时，再设计明确的数据配方。"),
    ("04", "重复校验和工程复杂度", "准备/检查中位数约 0.34 ms，未被证实是主要滞后来源。",
     "就地简化，取消专项审计",
     "碰到明确重复静态判断或无用分支才合并；不重建检查链、增加验证层或 bypass 开关。保留现有授权、限位、过期与停止边界。",
     "只确认实际修改的行为；没有改动就不补测试，不把完整审计作为实验前置。"),
    ("05", "训练/部署数值与时间语义", "尚无此次单位、顺序、dt 不匹配的证据；主机接收时间也不等于曝光时间。",
     "复用已有核对结果",
     "保持当前 joint 19 维、dt、normalizer 和动作含义。出现具体疑点才抽一个样本对照，不加数值契约、时钟同步系统或标定兼容门禁。",
     "已有配置/样本能回答即结束；不新增设备时间采集准入条件。"),
    ("06", "点云几何、裁剪与遮挡", "新数据抽查六帧，均有 1024 个不同 XYZ 点，玩具与容器轮廓保留；没有发现整物体被裁掉。",
     "保持几何配置，记录光照干扰",
     "强直射光造成明显颜色变化；模型 pc_dim=6，颜色也是输入。现场目前无法遮阳，因此不把不同轮次的抓取差异单独归因于执行配置。",
     "抽查不等于全程几何证明；MP4 RGB 有损，离线重建不是在线 tensor 的逐位回放。没有证据要求重标定或增加常驻几何检查。"),
    ("07", "XHand 告警信息", "已有板级 warning；信息不足不等于需要新的诊断体系。",
     "现有日志为主",
     "撤回 hand_code/Raw 扩展、JSONL 诊断流、板码事件记录和故障分类层。具体 SDK 错误无法定位时，只在相应现有日志带出 code/message；1501035 按匹配资料核实。",
     "A4 首轮 HOME 日志出现 1501070 与 tip[2]=307；不能仅凭同一条异常读取解释 307 的成因。结束后用现有只读入口复查 finger 2：comm/joint/tip 均为 0，12 关节静止电流绝对值 8–11 mA。四段 capture 辅助数据均有限，未新增忽略码或恢复机制。"),
    ("08", "async 名义截止过强", "三轮 async 均未触发 deadline 停止，实际推理最大约 41.9 ms；潜在误停仍未在此次真机复现。",
     "保留现有逻辑",
     "只有实际出现 handoff_model_deadline_exceeded，且时序证明动作仍在本拍有效，才定点调整。不增加 deadline 层、宽限参数或新异常分支。",
     "保留索引、时效与授权。若修改，只核对正常、仍有效的晚结果和真正过期三类；取消整套故障注入前置。"),
    ("09", "闭合大电流、未到位", "两轮 A8 接触时出现约 1.6 A 峰值和指关节短暂停滞；A4 两轮峰值分别约 0.53 A、1.27 A，实际接触不同。",
     "优先处理抓取位置，不盲调输出",
     "只读初检静止电流低；首轮小指约 10.1–10.4 s 受阻后恢复。第四轮能持续夹住并搬运玩具，尚无证据表明需要统一增加输出或增益。",
     "电流是实际 mA，tor_max 是 0–400 输出尺度，两者不能直接比较。不同抓取接触不能用峰值高低判断 A4 优劣；未做强制堵转或保护触发试验。"),
    ("10", "空手触觉、自接触、电流", "非零触觉不等于抓住，网格接触不等于真实碰撞。",
     "与 09 合并观察",
     "看实际接触与卸载后是否回落。有无接触偏置再按现有流程处理；不加接触分类器、碰撞规则或独立实验组。",
     "不在接触中 tare，不重复扣 Raw 偏置；暂时无法区分的现象保留不确定。"),
    ("11", "sync 每八步空一拍", "真机复现：sync 39 次块间间隔中位数 124.8 ms；async A8 为 61.7 ms，A4 约 62.0 ms。",
     "已用现有 async d1 配置消除固定空档",
     "继续使用 async_d1.yaml，不改调度框架。三轮 async 共 166 次块切换没有超过 90 ms 的发令间隔，无 skipped slot。",
     "确认的是调度机制；物体只能近似复位，第四轮布局明显不同、光照也变化，不能据此声称成功率提升。原定其余 sync 重复不再机械执行。"),
    ("12", "A8 闭环更新慢", "async A8 块尾最新输入源年龄中位数 543.2 ms；两轮 A4 均约 286.7 ms。",
     "后续实验采用 async d1 A4",
     "复用现有配置与 --n-action-steps 4，不添加平滑、补偿或重采样。A4 已实测缩短观测到块尾动作的间隔，但四轮均未完成任务。",
     "这是执行基线选择，不是 A4 最优或可靠成功率的结论。保持模型与手部参数，下一步优先检查抓取/放置目标和示教覆盖。"),
    ("13", "块边界大跳变", "反馈变化、采样噪声与跟随误差耦合，未证明必须加跨块算法。",
     "随 11/12 观察，突出时再定位",
     "撤回默认 96 份预测。必要时在 dexmani_policy 选两个关键观测各生成三次，先看同输入差异；不足以回答再扩大。",
     "本次机械臂块边界跳变 p95：sync A8 5.57°、async A8 4.67°、A4 2.60°/2.37°；轨迹不同，只作描述。仍不默认加入平滑、ensemble、RTC 或 PF-DAG。"),
    ("14", "xArm7 跟随滞后", "局部约 240 ms 是相位拟合，不是固定通信延迟；已有 Raw 可分析跟随。",
     "先复用现有分析",
     "已用 arm_ns 对齐此前完成发送的目标，并对第四轮放置阶段作 FK/视频投影；张手附近部分时刻 EEF 误差约 2–3 cm，不能称已消除。",
     "末端目标及指尖投影本身也朝容器外移动；本次失败不能仅解释为落后于正确的放置目标。FK 是主机目标描述，不是控制器内部参考或已识别通信延迟。"),
    ("15", "Mode 1 / Mode 6", "Mode 6 已可部署；Mode 1 需要合适频率与连续细粒度目标。",
     "保持 Mode 6，按需比较",
     "本轮维持 Mode 6；已实测连续执行、抓起和搬运。现有证据尚不足以把放置失败归因于模式，不为此添加高频 owner 或插值系统。",
     "Mode 1 未执行真机对照，不能声称其更差或等价。只有后续确认正确目标被执行响应持续破坏时，再比较同一温和连续轨迹的 Mode 6/16 Hz、Mode 6/F、Mode 1/F。"),
    ("16", "推物、空抓、滑落、空手搬运", "执行节奏正常后仍发生，优先看策略目标、输入与示教覆盖。",
     "回到策略和数据",
     "先看少量失败视频与对应示教，再比较一个已有 checkpoint 或 NFE/数据变量。不加自动重抓、失败 FSM、滤波、fallback；触觉 none/late/query 是独立研究。",
     "前两轮推物空抓；第三轮现场确认夹住边缘后掉落；第四轮确认抓住并搬运、张手时放偏。保持 async A4，针对接近位置、稳定包络和释放位置查看失败观测及相关示教；不把所有失败归为一种原因。"),
]

def main():
    with (HERE / "issues.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("id", "problem", "assessment", "decision", "action", "verification"))
        writer.writerows(ISSUES)
    parts = ['''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>简化后的问题与实验方案</title>
<style>body{max-width:1100px;margin:30px auto;padding:0 20px;font:16px/1.8 system-ui,sans-serif;color:#233746}.lead{font-size:25px;font-weight:650}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #d4dfe5;padding:10px;text-align:left;vertical-align:top}th{background:#eef3f6}a{color:#176c8b}.scroll{overflow:auto}.note{color:#526875}details{border-top:1px solid #d4dfe5;padding:12px 0}summary{font-weight:650;cursor:pointer}</style>
<p class="lead">先用现有能力解决问题；新增代码只服务已经遇到的具体障碍。</p>
<p>2026-10-12 真机后更新。已完成 tor_max 局部修复、四轮抓取（sync A8 一轮、async A8 一轮、async A4 两轮）和必要的只读状态复查。固定空档已通过现有 async 配置消除，A4 缩短了块尾观测年龄；四轮均未完成入筐，抓取/放置性能尚未解决。Raw 保持原样。</p>
<p>撤回新增诊断流、Raw 字段、导出汇总、故障分类层、常驻计时与整套故障注入。现有急停、运动授权、限位、时效和 Raw 真实性边界保留，不另建保障体系。已有离线核对不重复做；潜在问题不自动变成新工程任务。</p>
<p><a href="issues.csv">逐项清单</a> · <a href="../next_experiments/live_results/summary.html">真机证据与结果</a> · <a href="../next_experiments/plan.html">更新后的实验安排</a> · <a href="../next_experiments/trial_plan.csv">实验记录表</a> · <a href="offline_checks.json">修复前离线复现</a></p>
<p><strong>后续只保留有证据支持的工作。</strong></p>
<ol><li>以 Mode 6 + async d1 + A4 作为下一轮实验基线，继续保持现有手部输出和增益；没有新增部署机制。</li>
<li>针对已观察到的边缘抓取、推物空抓、放置位置偏差查看对应观测和示教；模型/数据改进留在 dexmani_policy。</li>
<li>只有正确目标仍被机械臂响应明显破坏，才做小范围 Mode 1/6 对照；本轮没有切换 Mode 1。</li></ol>
<p>四次均使用同一确切 checkpoint、EMA、NFE=4、dt=0.0625 s、seed=0 和手部参数。物体只做近似复位，第四轮更换了初始布局，直射光也变化。因此它们是机制筛选与失败案例，不能组成严格配对的成功率比较。确认机制后暂缓原六次 sync/async 排程中其余四次，没有为了填满表重复实验。</p>
<p>现场交互修正：全局 H/B/S/Q 会捕获桌面其他窗口的字母输入。一次准备 session 只归位后退出，疑与聊天输入冲突；不能从日志确定操作者意图。后续文字反馈全部在程序退出后收集，未新增输入框架。</p>
<p>结果记成功/失败/不确定，加一句原因和 Raw 路径。任务成功仍指规定时间内放入容器、释放撤离并稳定，无人工帮助。技术中断保留，不建立阶段识别器或评分服务。</p>
<div class="scroll"><table><tr><th>问题</th><th>判断</th><th>更新后的处理</th></tr>''']
    for number, problem, assessment, decision, action, verification in ISSUES:
        parts.append(f'<tr><td><a href="#issue-{number}">{number} {escape(problem)}</a></td>'
                     f'<td>{escape(assessment)}</td><td><strong>{escape(decision)}</strong><br>{escape(action)}</td></tr>')
    parts.append('</table></div><p><strong>确认范围缩小到实际改动。</strong>以下是按需人工或一次性核对，不是新运行时规则、测试框架或统一准入步骤。</p>')
    for number, problem, assessment, decision, action, verification in ISSUES:
        parts.append(f'<details id="issue-{number}"><summary>{number} {escape(problem)}</summary><p>{escape(verification)}</p></details>')
    parts.append('''<p><strong>参考项目的作用。</strong>LeFranX、pi-r2-flow、LeRobot、ManiUniCon 的短接口，RIO 的预取思想，DexUMI 对动作时间的重视和厂商插件的 Mode 6 路径，可以指导局部选择；PF-DAG 的生成一致性留作算法研究。不复制整套 middleware、高频控制或诊断体系。源码依据见 <a href="../reference_review.html">首次调研</a>、<a href="../reference_review_supplement.html">补充调研</a>和 <a href="../arm_mode_review.html">模式分析</a>，本轮不重复调研。</p>
<p>Mode 1 如需比较，仍使用连续细粒度目标与相同频率条件。这是控制模式的要求，不能用直接发送稀疏大步目标替代。现阶段维持 Mode 6，实际收益明确才考虑最小接入。</p>
<p class="note">此前离线复现记录的是修复前代码：401/1200 曾可通过旧配置校验，现已修复。没有改动返回码处理、停止边界、标定、控制器模式、输出增益或 Raw schema；此次真机使用现有配置完成。尚未执行 Mode 1、强制堵转、其他 checkpoint/NFE 或训练实验。</p></html>''')
    (HERE / "roadmap.html").write_text("\n".join(parts) + "\n")
    (HERE / "evidence.json").write_text(json.dumps(dict(
        status="four_hardware_trials_completed_task_performance_unresolved",
        basis="User requests simplicity without unnecessary diagnostics, validation or exception-handling mechanisms.",
        issue_count=len(ISSUES), production_code_modified=True, hardware_connected=True,
        hardware_sessions_closed_cleanly=True, completed_task_attempts=4,
        task_successes=0, empty_preparation_sessions=1,
        new_robotics_checks_run=True, prior_evidence="offline_checks.json",
        active_experiment_table="../next_experiments/trial_plan.csv",
        completed_work=["tor_max naming/range correction", "removed unsupported grasp-contact code comment",
                        "sync A8 / async A8 hardware comparison", "two async A4 attempts",
                        "Raw timing, video, sparse cloud and release tracking analysis",
                        "finger 2 read-only check after HOME warning"],
        offline_validation=["defaults/range/types/length", "config load/serialization", "compileall", "git diff --check"],
        immediate_work=["async d1 A4 Mode 6 baseline", "policy/data analysis of observed grasp and release failures"],
        conditional_work=["local async deadline correction if observed",
                          "short arm mode comparison only if needed", "policy/data work"],
        withdrawn=["new diagnostic stream and Raw fields", "export rejection JSON",
                   "new fault-classification and blanket unknown-code stop rule",
                   "permanent device timing and controller telemetry",
                   "full fault-injection matrix as prerequisite",
                   "fixed 96-prediction and 18-trajectory batches",
                   "precommitted 80/100-task schedule"],
    ), ensure_ascii=False, indent=2) + "\n")
    print("Updated roadmap.html, issues.csv and evidence.json; 16 issues retained.")

if __name__ == "__main__":
    main()
