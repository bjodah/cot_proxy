import pytest
import os
import tempfile
import json
import logging
import responses
from unittest.mock import patch, MagicMock
import re

from cot_proxy import VariantConfig, ThinkingConfig, PseudoModel


@responses.activate
def test_system_prompt_injection_new_messages(client, mocker, caplog):
    """Test system prompt injection when messages list is empty."""
    caplog.set_level(logging.DEBUG)
    mocked_target_base = "http://fake-target-system/"
    mocker.patch('cot_proxy.TARGET_BASE_URL', mocked_target_base)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("You are a helpful assistant.")
        system_prompt_file = f.name

    try:
        variant = VariantConfig(
            name='test-variant',
            label='test',
            model_regex='test-model',
            prepend_system_prompt_from_file=system_prompt_file,
            inject_at_end='',
            weak_defaults={},
            thinking=ThinkingConfig(do_strip=False, tags=('think', '/think')),
            weak_logit_bias=[]
        )

        pseudo = PseudoModel(upstream_model_name='test-model', variant=variant)

        with patch('cot_proxy.resolve_variant', return_value=pseudo):
            request_body = {"model": "test-model@test", "stream": False}

            payload_to_target = None
            def request_callback(request):
                nonlocal payload_to_target
                payload_to_target = json.loads(request.body)
                return (200, {}, json.dumps({"choices": [{"message": {"content": "Hello"}}]}))

            responses.add_callback(
                responses.POST,
                f"{mocked_target_base}v1/chat/completions",
                callback=request_callback,
                content_type="application/json"
            )

            proxy_response = client.post("/v1/chat/completions", json=request_body)

            assert proxy_response.status_code == 200
            assert payload_to_target is not None
            assert len(payload_to_target['messages']) == 2
            assert payload_to_target['messages'][0]['role'] == 'system'
            assert payload_to_target['messages'][0]['content'] == "You are a helpful assistant."
            assert payload_to_target['messages'][1]['role'] == 'user'
            assert "Prepended system prompt to messages" in caplog.text
    finally:
        os.unlink(system_prompt_file)


@responses.activate
def test_system_prompt_injection_existing_system_message(client, mocker, caplog):
    """Test system prompt injection when a system message already exists."""
    caplog.set_level(logging.DEBUG)
    mocked_target_base = "http://fake-target-system/"
    mocker.patch('cot_proxy.TARGET_BASE_URL', mocked_target_base)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("New system prompt.")
        system_prompt_file = f.name

    try:
        variant = VariantConfig(
            name='test-variant',
            label='test',
            model_regex='test-model',
            prepend_system_prompt_from_file=system_prompt_file,
            inject_at_end='',
            weak_defaults={},
            thinking=ThinkingConfig(do_strip=False, tags=('think', '/think')),
            weak_logit_bias=[]
        )

        pseudo = PseudoModel(upstream_model_name='test-model', variant=variant)

        with patch('cot_proxy.resolve_variant', return_value=pseudo):
            request_body = {
                "model": "test-model@test",
                "stream": False,
                "messages": [{"role": "system", "content": "Existing system message."}]
            }

            payload_to_target = None
            def request_callback(request):
                nonlocal payload_to_target
                payload_to_target = json.loads(request.body)
                return (200, {}, json.dumps({"choices": [{"message": {"content": "Hello"}}]}))

            responses.add_callback(
                responses.POST,
                f"{mocked_target_base}v1/chat/completions",
                callback=request_callback,
                content_type="application/json"
            )

            proxy_response = client.post("/v1/chat/completions", json=request_body)

            assert proxy_response.status_code == 200
            assert payload_to_target is not None
            assert len(payload_to_target['messages']) == 2
            assert payload_to_target['messages'][0]['role'] == 'system'
            assert payload_to_target['messages'][0]['content'].startswith("New system prompt.\n\nExisting system message.")
            assert payload_to_target['messages'][1]['role'] == 'user'
            assert "Prepended system prompt to existing system message" in caplog.text
    finally:
        os.unlink(system_prompt_file)


@responses.activate
def test_system_prompt_injection_no_system_prompt_configured(client, mocker, caplog):
    """Test that no system prompt is injected when prepend_system_prompt_from_file is not set."""
    caplog.set_level(logging.DEBUG)
    mocked_target_base = "http://fake-target-system/"
    mocker.patch('cot_proxy.TARGET_BASE_URL', mocked_target_base)

    variant = VariantConfig(
        name='test-variant',
        label='test',
        model_regex='test-model',
        prepend_system_prompt_from_file='',
        inject_at_end='',
        weak_defaults={},
        thinking=ThinkingConfig(do_strip=False, tags=('think', '/think')),
        weak_logit_bias=[]
    )

    pseudo = PseudoModel(upstream_model_name='test-model', variant=variant)

    with patch('cot_proxy.resolve_variant', return_value=pseudo):
        request_body = {
            "model": "test-model@test",
            "stream": False,
            "messages": [{"role": "user", "content": "Hello"}]
        }

        payload_to_target = None
        def request_callback(request):
            nonlocal payload_to_target
            payload_to_target = json.loads(request.body)
            return (200, {}, json.dumps({"choices": [{"message": {"content": "Hi"}}]}))

        responses.add_callback(
            responses.POST,
            f"{mocked_target_base}v1/chat/completions",
            callback=request_callback,
            content_type="application/json"
        )

        proxy_response = client.post("/v1/chat/completions", json=request_body)

        assert proxy_response.status_code == 200
        assert payload_to_target is not None
        assert len(payload_to_target['messages']) == 1
        assert payload_to_target['messages'][0]['role'] == 'user'
        assert payload_to_target['messages'][0]['content'] == "Hello"


@responses.activate
def test_system_prompt_injection_with_user_messages(client, mocker, caplog):
    """Test system prompt injection when user messages exist."""
    caplog.set_level(logging.DEBUG)
    mocked_target_base = "http://fake-target-system/"
    mocker.patch('cot_proxy.TARGET_BASE_URL', mocked_target_base)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("System instruction.")
        system_prompt_file = f.name

    try:
        variant = VariantConfig(
            name='test-variant',
            label='test',
            model_regex='test-model',
            prepend_system_prompt_from_file=system_prompt_file,
            inject_at_end='',
            weak_defaults={},
            thinking=ThinkingConfig(do_strip=False, tags=('think', '/think')),
            weak_logit_bias=[]
        )

        pseudo = PseudoModel(upstream_model_name='test-model', variant=variant)

        with patch('cot_proxy.resolve_variant', return_value=pseudo):
            request_body = {
                "model": "test-model@test",
                "stream": False,
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                    {"role": "user", "content": "How are you?"}
                ]
            }

            payload_to_target = None
            def request_callback(request):
                nonlocal payload_to_target
                payload_to_target = json.loads(request.body)
                return (200, {}, json.dumps({"choices": [{"message": {"content": "I'm good"}}]}))

            responses.add_callback(
                responses.POST,
                f"{mocked_target_base}v1/chat/completions",
                callback=request_callback,
                content_type="application/json"
            )

            proxy_response = client.post("/v1/chat/completions", json=request_body)

            assert proxy_response.status_code == 200
            assert payload_to_target is not None
            assert len(payload_to_target['messages']) == 4
            assert payload_to_target['messages'][0]['role'] == 'system'
            assert payload_to_target['messages'][0]['content'] == "System instruction."
            assert "Prepended system prompt to messages" in caplog.text
    finally:
        os.unlink(system_prompt_file)
