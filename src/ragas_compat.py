import sys
import types

_stub = types.ModuleType("langchain_community.chat_models.vertexai")


class ChatVertexAI:
    pass


_stub.ChatVertexAI = ChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = _stub
