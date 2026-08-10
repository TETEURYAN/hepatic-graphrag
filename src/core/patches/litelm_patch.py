import litellm

# Desativa o registro de cleanup async que causa o warning
def noop(*args, **kwargs):
    pass

try:
    litellm.register_async_client_cleanup = noop
except Exception:
    pass
