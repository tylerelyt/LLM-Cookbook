#!/usr/bin/env python3
"""
测试DashScope API连接
"""

import os
import requests
import json

def test_dashscope_api():
    """测试DashScope API连接"""
    
    # 读取配置
    LLM_MODEL = os.environ.get("LLM_MODEL", "qwen-max")
    DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
    
    if not DASHSCOPE_API_KEY:
        print("❌ 请设置DASHSCOPE_API_KEY环境变量")
        return
    
    print(f"模型: {LLM_MODEL}")
    print(f"API密钥: {DASHSCOPE_API_KEY[:20]}...")
    
    # 测试请求
    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "user", "content": "你好，请简单回复一下。"}
        ],
        "max_tokens": 100
    }
    
    try:
        print("正在测试API连接...")
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        print(f"状态码: {response.status_code}")
        print(f"响应头: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API连接成功!")
            print(f"回复: {result.get('choices', [{}])[0].get('message', {}).get('content', '')}")
        else:
            print("❌ API连接失败!")
            print(f"错误响应: {response.text}")
            
    except Exception as e:
        print(f"❌ 请求出错: {e}")

if __name__ == "__main__":
    test_dashscope_api() 