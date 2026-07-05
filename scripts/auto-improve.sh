#!/bin/bash
cd $(dirname $0)/..
git config user.email "manikantapotla3@gmail.com"
git config user.name "MANI8148"
while true; do
  opencode serve --port 4096 &
  sleep 5
  opencode run --auto --dir . \
    "Read ROADMAP.md. Implement next feature. Commit and push."
  sleep 21600
done
