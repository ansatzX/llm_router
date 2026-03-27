"""Tests for thread safety of global state initialization."""

import threading
import time
from unittest.mock import patch

from llm_router.llm_client import _get_client, get_request_timeout


class TestThreadSafety:
    """Test thread-safe initialization of global state."""

    def test_get_request_timeout_thread_safety(self, reset_llm_client_state):
        """Test that timeout is initialized exactly once under concurrent access."""
        import llm_router.llm_client as client_module

        init_count = 0

        # Patch the actual initialization to count how many times it runs
        original_init = client_module.httpx.Timeout
        def counting_init(*args, **kwargs):
            nonlocal init_count
            init_count += 1
            time.sleep(0.001)  # Increase race condition window
            return original_init(*args, **kwargs)

        with patch.object(client_module.httpx, 'Timeout', counting_init):
            threads = []
            for _ in range(10):
                t = threading.Thread(target=get_request_timeout)
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

        # Should initialize exactly once despite concurrent access
        assert init_count == 1, f"Timeout initialized {init_count} times (expected 1)"

    def test_get_client_thread_safety(self, reset_llm_client_state):
        """Test that client is created exactly once per cache key under concurrent access."""
        import llm_router.llm_client as client_module

        create_count = 0

        # Patch OpenAI client creation to count how many times it runs
        original_openai = client_module.OpenAI
        def counting_openai(*args, **kwargs):
            nonlocal create_count
            create_count += 1
            time.sleep(0.001)  # Increase race condition window
            return original_openai(*args, **kwargs)

        with patch.object(client_module, 'OpenAI', counting_openai):
            threads = []
            for _ in range(10):
                t = threading.Thread(
                    target=_get_client,
                    args=('http://localhost:8000', None)
                )
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

        # Should create exactly one client despite concurrent access
        assert create_count == 1, f"Client created {create_count} times (expected 1)"

    def test_multiple_clients_different_keys(self, reset_llm_client_state):
        """Test that different cache keys create different clients."""
        client1 = _get_client('http://localhost:8000', 'key1')
        client2 = _get_client('http://localhost:8000', 'key2')
        client3 = _get_client('http://localhost:9000', None)

        # Should create 3 different clients
        assert client1 is not client2
        assert client1 is not client3
        assert client2 is not client3

    def test_client_reuse_same_key(self, reset_llm_client_state):
        """Test that same cache key reuses existing client."""
        client1 = _get_client('http://localhost:8000', None)
        client2 = _get_client('http://localhost:8000', None)
        client3 = _get_client('http://localhost:8000', None)

        # Should reuse the same client
        assert client1 is client2
        assert client2 is client3

    def test_concurrent_different_clients(self, reset_llm_client_state):
        """Test concurrent creation of clients with different keys."""
        import llm_router.llm_client as client_module

        create_counts = {'client1': 0, 'client2': 0}

        # Patch OpenAI client creation
        original_openai = client_module.OpenAI
        def counting_openai(*args, **kwargs):
            # Detect which client is being created based on base_url
            base_url = kwargs.get('base_url', '')
            if '8000' in base_url:
                create_counts['client1'] += 1
            elif '9000' in base_url:
                create_counts['client2'] += 1
            time.sleep(0.001)
            return original_openai(*args, **kwargs)

        with patch.object(client_module, 'OpenAI', counting_openai):
            threads = []
            # Create 10 threads for each client
            for _ in range(10):
                t1 = threading.Thread(
                    target=_get_client,
                    args=('http://localhost:8000', None)
                )
                t2 = threading.Thread(
                    target=_get_client,
                    args=('http://localhost:9000', None)
                )
                threads.extend([t1, t2])
                t1.start()
                t2.start()

            for t in threads:
                t.join()

        # Each client should be created exactly once
        assert create_counts['client1'] == 1, f"Client1 created {create_counts['client1']} times"
        assert create_counts['client2'] == 1, f"Client2 created {create_counts['client2']} times"
