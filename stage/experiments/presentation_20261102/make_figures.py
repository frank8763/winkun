"""PPT figures from accepted Stage results; no native simulation or training."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from matplotlib.font_manager import FontProperties
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'stage/docs/presentation_20261102'
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
FP=FontProperties(fname=FONT);plt.rcParams['font.family']=FP.get_name();plt.rcParams['svg.fonttype']='path';plt.rcParams['axes.unicode_minus']=False
NAVY='#112338';BLUE='#1368a0';CYAN='#eaf5fc';GREEN='#167254';RED='#b64b39';MUTED='#62758b'

def text(ax,x,y,s,size=18,color=NAVY,weight='normal',ha='left',va='center'):
    ax.text(x,y,s,fontproperties=FP,fontsize=size,color=color,ha=ha,va=va,weight=weight,transform=ax.transAxes,linespacing=1.5)

def box(ax,x,y,w,h,title,body,color=BLUE,fill=CYAN):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012,rounding_size=0.012',facecolor=fill,edgecolor=color,lw=1.4,transform=ax.transAxes))
    text(ax,x+w/2,y+h*.77,title,19,color,'bold','center');text(ax,x+w/2,y+h*.36,body,15,NAVY,ha='center')

def arrow(ax,x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=20,color=MUTED,lw=2,transform=ax.transAxes))

def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=120,facecolor='white');fig.savefig(OUT/(name+'.svg'),facecolor='white');plt.close(fig)

fig=plt.figure(figsize=(16,9));ax=fig.add_axes([0,0,1,1]);ax.set_axis_off()
text(ax,.045,.93,'Stage：无地图端到端车辆导航',30,NAVY,'bold')
text(ax,.045,.87,'单一神经策略部署 · 真实执行器与 MuJoCo 动力学闭环',18,MUTED)
box(ax,.045,.57,.22,.21,'120维因果观测','局部激光、相对目标\n本体状态、显式历史')
box(ax,.31,.57,.28,.21,'单一深度 Actor','MLP主体 + 单步门控残差\n选中模型每次清零隐藏状态')
box(ax,.635,.57,.14,.21,'连续命令','驱动 + 转向')
box(ax,.82,.57,.135,.21,'原生动力学','2 ms物理步\n200 ms决策')
for x1,x2 in [(.27,.30),(.595,.625),(.78,.81)]:arrow(ax,x1,.675,x2,.675)
arrow(ax,.885,.56,.885,.515);arrow(ax,.885,.515,.155,.515);arrow(ax,.155,.515,.155,.56)
text(ax,.52,.49,'新的局部观测反馈',14,MUTED,ha='center')
box(ax,.045,.265,.21,.15,'真实闭环采样','时间相关 AR 探索')
box(ax,.29,.265,.21,.15,'多任务策略梯度','完整回报 / 裁剪代理 / PCGrad')
box(ax,.535,.265,.19,.15,'旧能力保持','保持误差 / 接受或回退')
box(ax,.76,.265,.195,.15,'冻结分级验收','开发 → 留出 → 独立重载')
for x1,x2 in [(.26,.28),(.505,.525),(.73,.75)]:arrow(ax,x1,.34,x2,.34)
text(ax,.045,.17,'已验证：3–5 m 原场景局部基线',20,GREEN,'bold');text(ax,.54,.17,'待解决：6 m 及更远距离',20,RED,'bold')
text(ax,.045,.075,'训练历史包含自身成功经验监督整合，再进行真实强化学习微调；图中组件不等于已证明的原创算法。',13,MUTED)
save(fig,'stage_method')

fig=plt.figure(figsize=(16,9));ax=fig.add_axes([.09,.20,.84,.58]);ax.set_ylim(0,116);ax.spines[['top','right']].set_visible(False)
labels=['3 m','4 m','5 m','6 m'];counts=[(108,108),(49,49),(54,54),(0,54)];heights=[100,100,100,0]
ax.bar(labels,heights,width=.55,color=[GREEN,GREEN,GREEN,RED],zorder=3)
ax.scatter([3],[1.5],s=160,color=RED,zorder=4)
ax.set_yticks([0,25,50,75,100]);ax.set_yticklabels(['0%','25%','50%','75%','100%'],fontsize=16);ax.tick_params(axis='x',labelsize=22,pad=12);ax.grid(axis='y',alpha=.18,zorder=0)
for i,(a,b) in enumerate(counts):ax.text(i,heights[i]+4,f'{a}/{b}',ha='center',fontsize=23,color=RED if i==3 else GREEN,fontweight='bold')
fig.text(.06,.91,'当前验收：3–5 m通过，6 m未通过',fontproperties=FP,fontsize=30,color=NAVY)
fig.text(.06,.845,'同一冻结Actor，原场景固定任务族与局部扰动；不是随机地图总体成功率',fontproperties=FP,fontsize=17,color=MUTED)
fig.text(.10,.075,'六米左侧：27接触    |    六米右侧：27超时    |    16例留出及78例复验未执行',fontproperties=FP,fontsize=17,color=RED)
save(fig,'stage_distance_results')
print('Saved two 1920x1080 PNG figures and text SVG sources')
