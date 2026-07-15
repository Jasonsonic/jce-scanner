# GitHub Actions + GitHub Pages 部署说明

这个版本不需要 Azure CLI。

1. 在 GitHub 新建仓库，例如 `jce-scanner`。
2. 将解压后的全部文件上传到仓库，必须保留 `.github/workflows/jce-pages.yml`。
3. 打开仓库 `Settings → Pages`，将 Source 设为 `GitHub Actions`。
4. 打开 `Actions → JCE Daily Scan → Run workflow`，手动运行第一次。
5. 完成后在 `Settings → Pages` 查看网页地址。

默认每个工作日 22:30 UTC 自动执行，对应新加坡时间次日06:30。

建议使用 GitHub Desktop 上传：
- File → Add local repository
- 选择解压文件夹
- Publish repository

注意：
- 公开仓库会公开自选股列表、代码和网页。
- 私人仓库能否使用 Pages 取决于 GitHub 套餐。
- GitHub Actions共享IP也可能被Yahoo限流；失败时稍后在Actions页面重新运行。
