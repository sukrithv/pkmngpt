"""Thin, swappable wrapper around the two generation backends: local Ollama
or the Claude API. Everything else in this project (RAG pipeline, DSPy
metrics) talks to `generate()` and doesn't care which backend is active.
"""

import config


def generate(prompt, system=None):
    if config.LLM_BACKEND == "ollama":
        return _generate_ollama(prompt, system)
    if config.LLM_BACKEND == "claude":
        return _generate_claude(prompt, system)
    raise ValueError(
        f"Unknown LLM_BACKEND: {config.LLM_BACKEND!r} (use 'ollama' or 'claude')"
    )


def _generate_ollama(prompt, system):
    import ollama

    client = ollama.Client(host=config.OLLAMA_BASE_URL)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = client.chat(model=config.OLLAMA_MODEL, messages=messages)
    return response["message"]["content"]


def _generate_claude(prompt, system):
    import anthropic

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=1024,
        system=system or "",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text
