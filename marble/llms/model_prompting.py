import litellm
from beartype import beartype
from beartype.typing import Any, Dict, List, Optional
from litellm.types.utils import Message

from marble.llms.error_handler import api_calling_error_exponential_backoff


@beartype
@api_calling_error_exponential_backoff(retries=5, base_wait_time=1)
def model_prompting(
    llm_model: str,
    messages: List[Dict[str, str]],
    return_num: Optional[int] = 1,
    max_token_num: Optional[int] = 512,
    temperature: Optional[float] = 0.0,
    top_p: Optional[float] = None,
    stream: Optional[bool] = None,
    mode: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[str] = None,
) -> List[Message]:
    """
    Select model via router in LiteLLM with support for function calling.
    """
    # litellm.set_verbose=True
    extra_body = None
    if llm_model.startswith("openai/Qwen"):
        base_url = "http://localhost:9999/v1"
        # Disable Qwen3 <think> so vLLM returns clean content (JSON for the
        # planner/judge); thinking left empty content and broke json parsing.
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
    elif "together_ai/TA" in llm_model:
        base_url = "https://api.ohmygpt.com/v1"
    else:
        # Catch-all: any other model name (e.g., "gpt-3.5-turbo" injected by
        # coding-tool arguments, or any leftover config model) is forced onto
        # the local Qwen3-8B vLLM, so everything runs on the local model.
        llm_model = "openai/Qwen3-8B"
        base_url = "http://localhost:9999/v1"
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
    completion = litellm.completion(
        model=llm_model,
        messages=messages,
        max_tokens=max_token_num,
        n=return_num,
        top_p=top_p,
        temperature=temperature,
        stream=stream,
        tools=tools,
        tool_choice=tool_choice,
        base_url=base_url,
        extra_body=extra_body,
    )
    message_0: Message = completion.choices[0].message
    assert message_0 is not None
    assert isinstance(message_0, Message)
    return [message_0]
