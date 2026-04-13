VICUNA_CHAT_TEMPLATE = """{% if messages[0]['role'] == 'system' -%}
{% set system_message = messages[0]['content'] | trim ~ ' ' -%}
{% set messages = messages[1:] -%}
{% else -%}
{% set system_message = '' -%}
{% endif -%}

{{- system_message -}}
{% for message in messages -%}
{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}
{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') -}}
{% endif -%}

{% if message['role'] == 'user' -%}
{{- 'USER: ' ~ message['content'] | trim ~ ' ' -}}
{% elif message['role'] == 'assistant' -%}
{{- 'ASSISTANT: ' ~ message['content'] | trim ~ eos_token -}}
{% endif -%}
{% endfor -%}

{% if add_generation_prompt -%}
{{- 'ASSISTANT:' -}}
{% endif -%}"""
