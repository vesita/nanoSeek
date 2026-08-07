# 09 · matplotlib 中文字体（Glyph missing）

## 现象
画 loss 曲线时刷屏：
```
UserWarning: Glyph 36845 (\N{CJK UNIFIED IDEOGRAPH-8FED}) missing from font(s) DejaVu Sans.
```
图上中文标题/坐标轴变成方块 □。

## 根因
matplotlib 默认字体 **DejaVu Sans 没有中文字形**。系统里装了
Noto Sans CJK，但 matplotlib 默认不用它。

## 修复
画图前按优先级找 CJK 字体，找不到回退默认（图能出，只是中文变方块，不崩）：

```python
from matplotlib import font_manager
for _font in ('Noto Sans CJK SC', 'Source Han Sans CN', 'WenQuanYi Zen Hei',
              'Microsoft YaHei', 'SimHei'):
    try:
        font_manager.findfont(_font, fallback_to_default=False)
        plt.rcParams['font.family'] = _font
        break
    except Exception:
        continue
plt.rcParams['axes.unicode_minus'] = False  # 负号用 ASCII，避免变方块
```

`findfont(..., fallback_to_default=False)` 找不到会抛异常 → 用 try/except 优雅回退，
这样脚本在没装中文字体的机器上也能跑。

## 怎么避免
中文 matplotlib 图的**标配三件套**：
1. `font_manager.findfont` 找 CJK 字体
2. 设 `plt.rcParams['font.family']`
3. `plt.rcParams['axes.unicode_minus'] = False`

验证是否修好：画图时把 UserWarning 当错误抛，有 glyph 警告就失败：

```python
import warnings
with warnings.catch_warnings():
    warnings.simplefilter('error', UserWarning)
    plt.savefig('test.png', dpi=120)   # 有任何 glyph 警告直接报错
```
