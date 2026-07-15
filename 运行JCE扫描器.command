#!/bin/bash
cd "$(dirname "$0")"
if [ ! -d ".venv" ]; then
  python3 -m venv .venv || exit 1
fi
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 jce_scan.py
echo
read -n 1 -s -r -p "扫描完成，按任意键关闭窗口……"
