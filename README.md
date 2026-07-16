# JCE Professional Scanner V4

V4采用两层结构：

## 第一层：Entry Filter

股票满足以下任一途径才进入候选：

### 均线压缩路线

- MA5、MA8、MA13均位于MA60上方或最多低0.2%
- 四线最大宽度不超过2%
- 最新收盘价距离MA60不超过5%
- 最近5日涨幅不超过20%
- 当前价格不处于半年区间90%以上

### 连跌首阳路线

- 最多检查15个交易日
- 中间允许最多4个单日涨幅不超过3%的小涨日
- 整个阶段净跌至少8%
- 最新交易日收阳且收盘高于前一日
- 当前价格距离三个月最低价不超过15%

## 第二层：Quality Score（100分）

- Compression：25
- Entry：20
- Trend与相对强度：15
- Reversal：15
- 三个月低位：10
- Dry-up Volume及放量确认：10
- VCP波动收缩：5

网页同时显示推荐原因和未通过原因。

## GitHub更新

将本包全部文件覆盖到SourceTree本地仓库：

暂存全部 → Commit → Push

GitHub Actions将自动运行并更新GitHub Pages。
