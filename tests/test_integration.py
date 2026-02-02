"""
Run integration tests for the Flask wrapper with fake SGLang server.
"""

import subprocess
import time
import json
import sys
import urllib.request
import urllib.error


def http_request(url, method="GET", data=None):
    """Simple HTTP request helper."""
    if method == "POST":
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode('utf-8') if data else None,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
    else:
        req = urllib.request.Request(url)

    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode('utf-8'))
    except urllib.error.URLError as e:
        if hasattr(e, 'read'):
            error_data = e.read().decode('utf-8')
            return {"error": str(e), "details": error_data}
        return {"error": str(e)}


def start_fake_sglang():
    """Start the fake SGLang server."""
    print("🚀 Starting fake SGLang server on port 30000...")
    proc = subprocess.Popen(
        [sys.executable, "fake_sglang_server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    time.sleep(2)
    return proc


def start_flask_wrapper():
    """Start the Flask wrapper."""
    print("🚀 Starting Flask wrapper on port 5000...")
    import os
    os.environ["SGLANG_BASE_URL"] = "http://localhost:30000"
    proc = subprocess.Popen(
        [sys.executable, "flask_wrapper.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    time.sleep(2)
    return proc


def test_health_check():
    """Test health check endpoint."""
    print("\n📋 Test 1: Health Check")
    try:
        data = http_request("http://localhost:5000/health")
        print(f"  ✅ Status: {data.get('status')}")
        print(f"     SGLang URL: {data.get('sglang_base_url')}")
        print(f"     Port: {data.get('flask_port')}")
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_tool_call_detection():
    """Test tool call detection and conversion."""
    print("\n📋 Test 2: Tool Call Detection")

    payload = {
        "model": "MiroThinker",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant with tool access."},
            {"role": "user", "content": "What's the weather like in London?"}
        ]
    }

    try:
        data = http_request("http://localhost:5000/v1/chat/completions", "POST", payload)
        message = data["choices"][0]["message"]

        print(f"  ✅ Response received")

        if "tool_calls" in message:
            print(f"  🔧 Tool calls detected: {len(message['tool_calls'])}")
            for tc in message["tool_calls"]:
                print(f"     - Tool: {tc['function']['name']}")
                print(f"       Arguments: {tc['function']['arguments']}")
            return True
        elif message.get("content"):
            print(f"  📝 Text response: {message['content'][:50]}...")
            return True
        else:
            print(f"  ⚠️  No content or tool calls")
            return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_multiple_requests():
    """Test multiple requests to cycle through mock responses."""
    print("\n📋 Test 3: Multiple Requests (Cycling Mock Responses)")

    results = {"tool_calls": 0, "text_only": 0}

    for i in range(5):
        payload = {
            "model": "MiroThinker",
            "messages": [
                {"role": "user", "content": f"Test request {i+1}"}
            ]
        }

        try:
            data = http_request("http://localhost:5000/v1/chat/completions", "POST", payload)
            message = data["choices"][0]["message"]

            if "tool_calls" in message:
                tool_names = [tc["function"]["name"] for tc in message["tool_calls"]]
                print(f"  Request {i+1}: 🔧 Tool calls: {', '.join(tool_names)}")
                results["tool_calls"] += 1
            else:
                content = message.get("content", "")[:40]
                print(f"  Request {i+1}: 📝 Text: {content}...")
                results["text_only"] += 1

        except Exception as e:
            print(f"  Request {i+1}: ❌ Error: {e}")

    print(f"\n  Summary:")
    print(f"    - Tool call responses: {results['tool_calls']}")
    print(f"    - Text-only responses: {results['text_only']}")
    return True


def test_follow_up_request():
    """Test follow-up request after tool result."""
    print("\n📋 Test 4: Follow-up Request (After Tool Result)")

    payload = {
        "model": "MiroThinker",
        "messages": [
            {"role": "user", "content": "What's the weather in London?"},
            {"role": "assistant", "content": "I'll check for you.", "tool_calls": [
                {"id": "call_0", "type": "function", "function": {"name": "get_weather", "arguments": '{"location": "London"}'}}
            ]},
            {"role": "tool", "tool_call_id": "call_0", "content": '{"temperature": 15, "condition": "sunny"}'}
        ]
    }

    try:
        data = http_request("http://localhost:5000/v1/chat/completions", "POST", payload)
        message = data["choices"][0]["message"]

        if message.get("content"):
            print(f"  ✅ Final response received:")
            print(f"     {message['content'][:100]}...")
            return True
        else:
            print(f"  ⚠️  No content in response")
            return False

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_models_endpoint():
    """Test models list endpoint."""
    print("\n📋 Test 5: Models List Endpoint")

    try:
        data = http_request("http://localhost:5000/v1/models")

        if data.get("object") == "list":
            models = [m["id"] for m in data.get("data", [])]
            print(f"  ✅ Available models: {', '.join(models)}")
            return True
        else:
            print(f"  ⚠️  Unexpected response format")
            return False

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def main():
    """Run all tests."""
    print(f"\n{'='*60}")
    print("Flask Wrapper Integration Tests")
    print(f"{'='*60}\n")

    # Start servers
    fake_server = start_fake_sglang()
    flask_wrapper = start_flask_wrapper()

    try:
        # Run tests
        tests = [
            test_health_check,
            test_tool_call_detection,
            test_multiple_requests,
            test_follow_up_request,
            test_models_endpoint
        ]

        passed = 0
        for test in tests:
            if test():
                passed += 1

        print(f"\n{'='*60}")
        print(f"Test Results: {passed}/{len(tests)} passed")
        print(f"{'='*60}\n")

    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        fake_server.terminate()
        flask_wrapper.terminate()
        fake_server.wait()
        flask_wrapper.wait()
        print("✅ All servers stopped")


if __name__ == "__main__":
    main()
