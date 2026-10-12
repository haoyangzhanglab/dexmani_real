"""Render this four-attempt experiment report from offline derived metrics."""
import json
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SELECTOR = 'maniflow/pick_place_toy/2026-10-11_02-00-52-803763_1cbdb2c1_1011'
ATTEMPTS = [
    ('135345', 'sync A8', 'P1', '推转玩具后空抓、空手移向容器；现场确认失败'),
    ('135924', 'async d1 A8', 'P1 近似复位', '推物后空抓；视频确认未完成，现场未报告额外异常'),
    ('140158', 'async d1 A4', 'P1 近似复位', '现场确认夹到边缘，未形成稳定抓取，玩具移动后掉落'),
    ('140823', 'async d1 A4', 'P2 不同布局', '现场确认抓住并搬运到容器附近，张手时放偏'),
]


def main():
    rows, cases, records = [], [], []
    for stamp, condition, layout, observation in ATTEMPTS:
        name = f'session_20261012_{stamp}'
        metric_path = HERE / f'{name}_episode_001.json'
        m = json.loads(metric_path.read_text())
        ep = m['session_result']
        records.append(dict(session=name, condition=condition, layout=layout,
                            outcome='failure', observation=observation, metrics_file=metric_path.name))
        rows.append(
            f'<tr><td>{condition}<br>{stamp} · {layout}</td>'
            f'<td>{ep["duration_s"]:.1f} s<br>{m["rows"]} / {ep["metrics"]["dispatch_count"]}</td>'
            f'<td>{m["gaps_ms"]["between_chunks"]["median"]:.1f} ms</td>'
            f'<td>{m["gaps_over_90ms"]}</td>'
            f'<td>{m["block_last_source_age_ms"]["median"]:.1f} ms</td>'
            f'<td>{observation}</td></tr>')
        raw = ROOT / 'rollouts' / SELECTOR / name
        cases.append(
            f'<details><summary>{condition} · {stamp} · {observation}</summary>'
            f'<p><a href="{raw / "episode_001/rgb.mp4"}">原视频</a> · '
            f'<a href="{raw / "session_result.json"}">session_result</a> · '
            f'<a href="{raw / "run_config.yaml"}">resolved config</a> · '
            f'<a href="{metric_path.name}">离线指标</a></p>'
            f'<img src="{name}_episode_001.jpg" alt="{escape(observation)}"></details>')
    summary = dict(
        date='2026-10-12', attempts=records, successes=0,
        recommendation='Mode 6 + async d1 + A4 as next experimental baseline; task performance unresolved',
        settings=dict(checkpoint='epoch=0566-step=00060000-milestone=60pct.pt', weights='ema',
                      inference_steps=4, seed=0, control_hz=16, hand_parameters='unchanged'),
        empty_preparation_session='session_20261012_140632: two completed HOME requests, no episode; clean shutdown',
        hand_readback=dict(command='examples/xhand_control_example.py --read-only --finger-id 2',
                           sdk_version='1.4.2', hardware_version='1.1.0', inspected_finger=2,
                           comm_err=0, joint_err=0, tip_err=0, rest_current_abs_mA_range=[8, 11]),
        limitations=['Approximate resets and a different fourth layout; daylight changed.',
                     'This is not a paired success-rate experiment or proof of controller latency.',
                     'Mode 1, altered hand gains/output limits and forced-stall protections were not tested.',
                     'CRC/tip warning root cause remains unconfirmed; later zero status does not erase it.'],
        production_changes_in_hardware_turn=[], raw_modified=False,
    )
    (HERE / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>2026-10-12 真机结果</title>
<style>body{max-width:1100px;margin:30px auto;padding:0 20px;font:16px/1.8 system-ui;color:#233746}.lead{font-size:26px;font-weight:650}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #d4dfe5;padding:10px;text-align:left}th{background:#eef3f6}img{max-width:100%}a{color:#176c8b}details{border-top:1px solid #d4dfe5;padding:12px 0}summary{cursor:pointer;font-weight:600}.scroll{overflow:auto}</style>
<p class="lead">四轮真机：调度空档已消除，抓取与放置性能尚未解决。</p>
<p>推荐后续基线为 <strong>Mode 6 + async d1 + A4</strong>。使用现有入口和配置，未增加运行时诊断、验证、补偿或异常处理机制。四轮均在 60 s 预算内手动停止并正常发布 Raw，均未完成“放入容器、释放撤离并稳定至少 2 s”的任务标准。全部程序已正常退出并关闭连接。</p>
<p><a href="../../issue_roadmap/roadmap.html">16 个问题的更新判断</a> · <a href="../plan.html">下一步与可直接运行的命令</a> · <a href="../trial_plan.csv">实验记录</a> · <a href="summary.json">结果摘要</a> · <a href="analyze_trial.py">只读离线指标脚本</a></p>
<p>共同设置：同一 60pct checkpoint（epoch 0566 / step 60000）、EMA、NFE=4、seed=0、16 Hz、Mode 6、原手部数值/增益。最初两轮调度机制已确认，因此暂缓其余四次预排 sync/async，直接试 A4；没有为了填满实验表增加重复。</p>
<div class="scroll"><table><tr><th>条件 / 初始布置</th><th>运行时长<br>Raw 帧 / 发令数</th><th>块间间隔中位数</th><th>间隔 &gt;90 ms 次数</th><th>块尾源年龄中位数</th><th>任务观察（均失败）</th></tr>'''
    html += ''.join(rows) + '''</table></div>
<p>间隔取已完成的 arm+hand 发送时间；90 ms 只是离线统计阈值，没有成为新运行时限制。块尾源年龄使用该预测最新观测中最旧的有效输入源时间，稳态 async 块尾按实际 prediction_index 统计。不同物体接触和轨迹使角度误差/电流峰值不能直接作性能因果比较。</p>
<p><strong>已确认的机制。</strong>sync 39 次约 125 ms 空档在 async 中消失。三轮 async 共 166 次块切换，未出现超过 90 ms 的间隔、跳过控制拍或 deadline 停止。A4 将稳态块尾源年龄从 A8 约 543 ms 降至约 287 ms；这不等于任务成功率已提高。sync 41 行未发令，async 每段各 2 行启动等待，NaN 目标保留原意，不补零或删帧。</p>
<p><strong>抓取与手部。</strong>前两轮推物空抓；第三轮只夹住边缘；第四轮已能夹住搬运，但张手放偏。接触时两轮 A8 的电流峰值约 1.6 A，A4 两轮约 0.53 A / 1.27 A；不据此调高输出。tor_max 的 0–400 SDK 输出尺度不能与反馈电流 mA 直接比较。</p>
<p><strong>告警。</strong>第三轮 HOME 出现 1501070 与 tip[2]=307，具体根因未确认。后续只读复查 finger 2 的 comm/joint/tip 均为 0，12 关节静止电流绝对值 8–11 mA。四段 capture 的关节、触觉和电流均有限。保留原日志，不扩大忽略码、不新增自动恢复，也不把有限数值当作“所有硬件故障已排除”。</p>
<p><strong>Mode 1 / Mode 6。</strong>第四轮张手附近仍有部分时刻约 2–3 cm 的 EEF 跟随误差；但目标轨迹与目标指尖投影本身也朝容器外移动，因此不能把失败单独归为 Mode 6 没跟上正确目标。暂保留 Mode 6。Mode 1 没有实测，本轮不证明两种模式等价，也未证明跟随滞后没有影响。以后只有正确目标被响应持续破坏，才需要同轨迹、同频率的模式对照。</p>
<p><a href="release_tracking.json">放置时序数值</a> · <a href="release_fingertips.json">指尖 FK 数值</a>。下面黄色为反馈关节推算的指尖，红色为前一次完成发送的机械臂目标配合相同手部姿态，用于隔离机械臂位置差。投影使用 Raw 内外参与当前运行的安装变换；这是主机目标描述，不是控制器内部参考、物体质心或已识别的通信延迟。</p>
<img src="release_fk_overlay.jpg" alt="第四轮放置时刻，实际与目标机械臂的指尖投影">
<details><summary>放置时序曲线</summary><img src="release_tracking.png" alt="目标、反馈和手部张开时序"></details>
<p><strong>观测输入。</strong>前两轮各抽查三个时刻，1024 个不同 XYZ 点和玩具/容器轮廓均保留，没有发现整物体被裁掉。强直射光造成明显颜色变化，模型使用 6 维彩色点云；现场无法遮阳，保留为实验干扰。六帧抽查不能排除局部几何偏差，MP4 RGB 有损，也不等于在线 tensor 逐位复现。</p>
<details><summary>六帧点云抽查</summary><p><a href="cloud_spot_check.json">数值</a></p><img src="cloud_spot_check.png" alt="两轮各三帧的点云"></details>
<p><strong>交互和实验边界。</strong>另外保留 session_20261012_140632：两次 HOME 后退出，没有 B 开始、没有 episode，不计为第五次抓取。全局字母快捷键可能捕获聊天输入，但日志不能证明当时操作者意图；后续将文字反馈安排在程序退出后，未新增输入框架。第一、二轮只能近似复位，第四轮布局明显不同，且日照变化，不能据这四次声称严格配对成功率。</p>
<p><strong>下一步。</strong>固定 async A4 / Mode 6，优先对照“边缘夹取”和“张手放偏”两类观测与相关示教，检查抓取中心、接近方向、物体姿态覆盖和释放位置；然后只选择一个已有 checkpoint 或数据变量做定向对照。现有证据不足以要求增加平滑、自动重抓、握持状态机、deadline 宽限或高频控制框架。</p>'''
    html += ''.join(cases) + '</html>\n'
    (HERE / 'summary.html').write_text(html)
    print('Wrote summary.html and summary.json; 4 attempts, 0 task successes.')


if __name__ == '__main__':
    main()
