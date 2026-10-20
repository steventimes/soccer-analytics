# 足球机构的公开方法

2026-10-14 核对官方方法页、产品发布稿与作者文章。模型类型、可用输入、公开数字和未披露参数分别记录。历史文章只说明当时的版本，不能据此断言 2026 年的完整生产配置。[结构化目录](methods.json)也用于研究页面。

| 机构和用途 | 公开方法与输入 | 参数与复现范围 |
| --- | --- | --- |
| Opta，赛季预测 | 赔率市场与 Power Rankings 形成比赛概率，再模拟剩余赛程 | 2026/27 英超文章明确模拟 10,000 次，预测快照为 2026-08-13。融合权重未公开。[官方预测说明](https://theanalyst.com/articles/premier-league-predictions-2026-27-opta-supercomputer) |
| Opta，球队实力 | 分层 Elo，结合赛果、xG 和联赛、国家、洲的关系 | 有 xG 时，80% 来自比分产生的 Elo 变化，20% 来自 xG 分布产生的变化。完整更新系数未公开。[评级方法](https://theanalyst.com/articles/power-rankings-your-club-ranked) |
| Hudl StatsBomb，xG 与门将 | 梯度提升树，射门几何、门将和防守位置。用二维高斯软化遮挡，并加入单调约束 | 2022 升级稿没有给出树数、深度、学习率或高斯宽度。射后模型另含落点和球速。[模型升级说明](https://www.hudl.com/blog/upgrading-expected-goals) |
| Hudl StatsBomb，OBV | 分别估计本方和对方的进球价值，使用行动几何、身体部位、比赛情境与受压状态 | 2021 发布稿说明采用 xG 目标，并排除控球历史。所查页面未给确切模型架构。[OBV 发布稿](https://blogarchive.statsbomb.com/news/introducing-on-ball-value-obv/) |
| Thom Lawrence，XSpace | 梯度提升树，以行动、位置、当前 OBV 和对手身份预测空间。再与 360 中最近防守人距离比较 | 2022 作者原型未单独控制己方创造空间的能力，不能视为已确认的当前商业模型。[空间与压力分析](https://blogarchive.statsbomb.com/articles/soccer/revisiting-pressure-with-statsbomb-360/) |
| Understat，xG | 官网说明采用神经网络，训练样本超过十万次射门，每次超过十项参数 | 网络层数、宽度、学习率与完整训练特征未公开。下载字段不等同于训练字段。[官网说明](https://understat.com/) |
| CIES，转会估值 | 历史交易统计模型，输入含年龄、合同、职业经历、表现、球队竞技和财务水平 | 2020 报告使用 1,790 笔交易，按随机 80%/20% 验证。完整系数和当前版本未公开。[Monthly Report 53](https://football-observatory.com/IMG/sites/mr/mr53/en/) |
| FiveThirtyEight，SPI 历史接口 | 官方 README 列出赛前评级、攻防参数、比赛概率和预计进球，也列出赛后 xG 等字段 | 方法文章本轮跳转到其他页面，未确认具体算法或权重。README 的 latest 不证明仍更新。[官方数据仓库](https://github.com/fivethirtyeight/data/tree/master/soccer-spi) |

Opta 的 80/20 混合对象是两种评级变化，不是原始进球与 xG，也不是胜率。展示的 0–100 实力等级同样不是胜率。蒙特卡洛是利用概率反复模拟结果的方法，不能只因名称是 supercomputer 就归类为深度学习。

球员分析需要先区分目标。转会价格受到合同与买方财力影响，不能直接作为潜力或战术适配标签。OBV 的发布稿不直接给接球者记价值，单靠有球事件也难以估计无球跑动。360 的空间快照提供了更多对手情境，仍缺少连续跑动时序。

10 月 18 日已重新读完 xG、OBV 和 XSpace 三篇文章的正文，并查看 OBV 的四张、XSpace 的六张静态配图；xG 的嵌入图像和动画、OBV 视频及客户白皮书不在本次已读范围。XSpace 的最近防守人距离是监督目标，由 360 提供观测，不能再列为同一预测的输入；输入、目标和 SAX 比值已在[方法表](methods.json)及本地页面数据中分开说明。三篇都属于历史公开版本，尚未确认 2026 年生产模型是否沿用这些细节。

这些资料给本项目的启发是把赛前实力作为对手难度参照，将有球行动的收益与风险分开，并在空间模型中考虑对手。具体实现还受本地覆盖限制，官方算法和商业数据不能被公共样本完整替代。ClubElo 的官方方法页读取超时，因此没有补写常见的 K 值、主场修正或净胜球公式。

表示学习可以减少任务特征的手工设计，但观察单位、坐标方向、时间切分和标签仍要定义清楚。本轮 GRU、共享任务和遮挡重建的实际实验见[实验报告](REPRESENTATION_LEARNING.md)，机构资料没有被当成这些实验优于基线的证据。
