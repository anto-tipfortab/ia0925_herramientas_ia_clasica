"""Shared pytest fixtures/config.

A dummy OPENAI_API_KEY is set before the app modules import so that any
accidental real-client construction fails loudly rather than reaching out to a
real service. All external APIs (OpenAI, Chroma, Polly, Dialogflow) are mocked
in the individual tests — the suite never touches the network.
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used")
os.environ.setdefault("AWS_DEFAULT_REGION", "eu-west-1")
