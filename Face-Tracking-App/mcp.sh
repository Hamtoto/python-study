#!/bin/bash

# 가상 환경 활성화 및 Serena MCP 실행 스크립트
source .venv/bin/activate

# 2. Serena 디렉토리로 이동
cd serena || {
  echo "❌ 'serena' 디렉토리를 찾을 수 없습니다."
  exit 1
}

# 3. Serena MCP 실행
echo "🚀 Serena MCP 서버 실행 시작..."
uv run --active serena-mcp-server
