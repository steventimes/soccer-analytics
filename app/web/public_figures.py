"""Approved original figures for the static research site."""


PUBLIC_FIGURES = (
    ("analysis-upgrades", "压迫、进球概率、空间控制与组织进攻", "模型比较", "各实验使用各自样本与指标，请在图中核对分母和误差范围。", "公开研究实验汇总"),
    ("analysis-upgrades/pressing", "压迫链如何结束", "战术与空间", "供应商标注的压迫链，包含夺回、干扰与未知结果。", "SkillCorner 开放比赛数据"),
    ("analysis-upgrades/goal-probability", "上下文模型能否改善进球概率预测", "模型比较", "历史测试集上的配对比较，区间按比赛重采样。", "Wyscout 历史事件实验"),
    ("analysis-upgrades/space-control", "同一帧下的三种空间控制模型", "战术与空间", "同一公开追踪帧和网格上的模型差异，没有空间控制真值。", "Metrica Sports 开放追踪数据"),
    ("analysis-upgrades/buildup", "加入 xGBuildup 与 xGChain 后的预测比较", "球员与球队", "用历史赛季的参与指标预测下一赛季 xGBuildup/90，区间按球员聚类。", "Understat 历史球员面板实验"),
    ("chance-ablations", "机会模型的特征消融比较", "模型比较", "同一历史测试划分下的模型与特征组合，结果不代表其他赛季。", "StatsBomb 开放事件实验"),
    ("chelsea-psg-final", "切尔西与 PSG 的决赛踢法变化", "球员与球队", "2025 世俱杯决赛与各队此前六场比较，未校准对手强度。", "FIFA Training Centre 官方比赛报告"),
    ("external-load", "跑动负荷与画面可见范围", "跑动与负荷", "公开追踪片段的观测量，画面覆盖限制了可解释范围。", "SkillCorner 开放转播追踪样本"),
    ("football/shot-map", "射门来自哪里", "比赛片段", "单场比赛的射门位置与结果，样本和方向见图中说明。", "StatsBomb 开放事件数据"),
    ("football/passing-network", "传球联系集中在哪些位置", "比赛片段", "指定比赛和时段的完成传球网络，节点位置为事件起点均值。", "StatsBomb 开放事件数据"),
    ("football/visible-space", "持球队员身边的可见空间", "战术与空间", "单个 360 事件画面中的可见球员，未观测球员不能视为空间无人。", "StatsBomb 360 开放数据"),
    ("football/running-position", "观测片段中的跑动与位置", "跑动与负荷", "公开追踪片段，不外推到完整比赛或球员能力。", "Metrica Sports 开放追踪数据"),
    ("player-assessment", "球员代理目标的模型比较", "球员与球队", "历史球员样本上的测试误差，代理指标不等于潜力或实际能力。", "Wyscout 历史球员事件实验"),
    ("representation/Figure_1", "事件表示学习的测试结果", "表示学习", "Wyscout 2017/18，点与横线为种子均值和标准差，标准差不是置信区间。", "Wyscout 历史事件表示学习实验"),
    ("representation/Figure_2", "表示学习模型的训练过程", "表示学习", "目标联赛验证损失，仅画一个随机种子的训练过程。", "Wyscout 历史事件表示学习实验"),
    ("representation-leagues/Figure_1", "相同历史输入下的五联赛模型对照", "表示学习", "MLP与GRU减去各自联赛的树基线，负值误差更低。横线为固定模型的逐项整场重采样范围，不含训练不确定性；同赛季对照不能证明跨赛季效果。", "Wyscout 2017/18，五联赛独立目标训练"),
    ("scouting-observations", "年轻球员观察分组与传球方向", "球员与球队", "历史事件样本中的跟进分组与位置分布，分组不等于潜力排名。", "Wyscout 历史球员与事件统计"),
    ("switch-context", "转移球前后的传球上下文", "战术与空间", "规则识别的转移球与匹配比较，事件上下文不证明因果效果。", "SkillCorner 开放比赛数据"),
    ("team-assessment", "球队预测模型与基线比较", "球员与球队", "历史比赛的时间顺序留出评估，具体指标与基线见图。", "Football-Data 历史赛果与球队统计"),
    ("spatial-style/Figure_1", "空间球风表示的识别稳定性", "模型比较", "Wyscout 2017/18，同俱乐部且两段各至少900分钟的球员。种子范围不是置信区间；识别率不是能力评分，人数见图。", "Wyscout 公开事件与 Player Vectors 启发的本地实验"),
    ("spatial-style/Figure_2", "早段动作拟合的传球空间基底", "球员与球队", "只展示一个固定种子的前两个传球分量，起终点共用色阶。基底不是站位时长或空间控制。", "Wyscout 2017/18 早段聚合传球热图"),
    ("vaep-spadl/Figure_1", "动作历史能否改善进球与失球概率", "模型比较", "英超、法甲时间留出，排除当前进球状态。横线按整场比赛重采样，未覆盖训练不确定性。", "Wyscout 2017/18 与本地 SPADL/VAEP 对照"),
    ("vaep-spadl/Figure_2", "动作增量中的进攻与防守贡献", "球员与球队", "测试集按动作类型汇总，不是球员排名或因果效果。转换器补出的带球会单独计数。", "Wyscout 公开事件与 socceraction 1.5.3 转换"),
    ("outcome-comparison", "比分模型的差值有多稳定", "模型比较", "五联赛固定历史预测，按整天比赛配对重采样。三种平均块长度的百分位范围不等同于显著性或未来赛季优势。", "Football-Data 2022/23–2024/25，固定比分配置与默认 XGBoost"),
    ("paper-followup-20261019/score-counts-pl", "进球频数与完整尾项", "模型比较", "英超 2025/26 的76场时间留出；观察频数与五种点拟合模型的期望频数。没有后验区间或拟合优度检验。", "Football-Data 校验快照，2026年10月19日离线重算"),
    ("paper-followup-20261019/global-contributions", "同一输出下的特征贡献分布", "模型比较", "英超历史留出前32场，固定主胜减客胜的原始分数。树路径参照；不是校准概率或因果影响。", "Football-Data 2025/26，开发期固定 XGBoost"),
    ("paper-followup-20261019/feature-dependence", "赛前特征与原始分数贡献", "模型比较", "同一固定类别对和样本，横轴保留特征原单位。关联散点不等于部分依赖或交互检验。", "Football-Data 2025/26，固定类别 TreeSHAP"),
    ("paper-followup-20261019/feature-interactions", "显式特征交互与主效应", "模型比较", "对角为平均绝对主效应，非对角为完整成对交互的平均绝对值；不是显著性或因果图。", "Football-Data 2025/26，原生 TreeSHAP 交互"),
)



def public_figures(source):
    """Read only approved image paths; missing optional images do not block export."""
    figures, assets = [], {}
    for stem, title, topic, scope, origin in PUBLIC_FIGURES:
        for suffix in ("svg", "png"):
            image = source / f"{stem}.{suffix}"
            if not image.is_file() or not image.resolve().is_relative_to(source.resolve()):
                continue
            identity = stem.replace("/", "-").replace("_", "-").lower()
            filename = f"{identity}.{suffix}"
            assets[filename] = image.read_bytes()
            figures.append({
                "id": identity,
                "title": title,
                "topic": topic,
                "filename": filename,
                "format": suffix,
                "alt": f"{title}。{scope}",
                "scope": scope,
                "source": origin,
            })
            break
    return {"figures": figures}, assets
