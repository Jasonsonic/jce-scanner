# JCE Scanner GitHub Pages 最新版

## 部署

1. 将本文件夹内全部文件提交并推送到 GitHub 仓库 `main` 分支。
2. 确认存在：
   `.github/workflows/jce-pages.yml`
3. 在 GitHub 仓库进入：
   `Settings → Pages`
4. 将 Source 设置为：
   `GitHub Actions`
5. 进入：
   `Actions → JCE Daily Scan → Run workflow`
6. 首次运行成功后，在 `Settings → Pages` 查看网页地址。

## 自动运行时间

默认每个工作日 22:30 UTC 运行，对应新加坡时间次日 06:30。

## 本版内容

- JCE V2 五维评分
- Yahoo Finance限流缓解
- 本地及GitHub Actions缓存
- 自动生成HTML排行榜
- 网页下载Excel与CSV
- 145只股票自选列表

## 注意

GitHub Actions共享出口IP仍可能被Yahoo限流。失败时可稍后在Actions中重新运行。
