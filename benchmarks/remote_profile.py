import asyncio

from backend_request_func import (
    ASYNC_REQUEST_FUNCS,
    RequestFuncInput,
    RequestFuncOutput,
)
from openai import OpenAI

request_func = ASYNC_REQUEST_FUNCS["vllm"]

profile_input = RequestFuncInput(
    model=0,
    prompt="start profile",
    api_url="http://localhost:8000" + "/start_profile",
    prompt_len=2,
    output_len=1,
    logprobs=0.9,
    best_of=1,
)
profile_output = asyncio.run(request_func(request_func_input=profile_input))
if profile_output.success:
    print("Profiler started")


client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",
)

models = client.models.list()
model = models.data[0].id

# Single-image input inference
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
image_url = "https://images.pexels.com/photos/1525041/pexels-photo-1525041.jpeg?cs=srgb&dl=pexels-francesco-ungaro-1525041.jpg&fm=jpg"
# image_url = "https://t3.ftcdn.net/jpg/05/71/06/76/360_F_571067620_JS5T5TkDtu3gf8Wqm78KoJRF1vobPvo6.jpg"

## Use image url in the payload
chat_completion_from_url = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
            ],
        }
    ],
    model=model,
    max_tokens=3,
)

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",
)

models = client.models.list()
model = models.data[0].id

# Single-image input inference
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
image_url = "https://images.pexels.com/photos/1525041/pexels-photo-1525041.jpeg?cs=srgb&dl=pexels-francesco-ungaro-1525041.jpg&fm=jpg"
# image_url = "https://t3.ftcdn.net/jpg/05/71/06/76/360_F_571067620_JS5T5TkDtu3gf8Wqm78KoJRF1vobPvo6.jpg"

## Use image url in the payload
chat_completion_from_url = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
            ],
        }
    ],
    model=model,
    max_tokens=3,
)


profile_input = RequestFuncInput(
    model=0,
    prompt="end profile",
    api_url="http://localhost:8000" + "/stop_profile",
    prompt_len=2,
    output_len=1,
    logprobs=0.9,
    best_of=1,
)
profile_output = asyncio.run(request_func(request_func_input=profile_input))
if profile_output.success:
    print("Profiler ended")

print(chat_completion_from_url)
