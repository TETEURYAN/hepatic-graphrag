"""
Hotfix para `graphrag query` (v2.7+): os passos "map" da busca global e
"primer" da busca drift forçam response_format={"type": "json_object"} na
chamada ao LLM, ignorando a config `model_supports_json` do settings.yaml.
Isso quebra contra roteamentos do OpenRouter (ex.: provider Novita) cujo
modelo só aceita response_format=json_schema:

    litellm...OpenAIError: Model 'tencent/hy3' does not support 'json_object'
    response format. Supported formats: json_schema.

Os prompts do GraphRAG já pedem JSON em texto e o parsing já é tolerante
(graphrag.query.llm.text_utils.try_parse_json_object), então remover o
response_format forçado tem o mesmo efeito pretendido por
`model_supports_json: false` — só que aplicado de fato.
"""

from graphrag.language_model.providers.litellm.chat_model import LitellmChatModel

_original_get_kwargs = LitellmChatModel._get_kwargs


def _get_kwargs_sem_json_mode(self, **kwargs):
    kwargs = dict(kwargs)
    kwargs.pop("json", None)
    kwargs.pop("response_format", None)
    kwargs.pop("json_model", None)
    return _original_get_kwargs(self, **kwargs)


LitellmChatModel._get_kwargs = _get_kwargs_sem_json_mode
