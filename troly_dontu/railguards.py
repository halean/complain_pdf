import os
import asyncio
from openai import AsyncOpenAI

openai_token = os.getenv("OPENAI_API_KEY")
open_ai_client = AsyncOpenAI(api_key=openai_token)  # uses env by default if None

llm_fns = {
    "open_ai_gpt_4o": lambda user_request, system_prompt: get_openai_chat_response(
        user_request, system_prompt, model="gpt-4o"
    ),
    "open_ai_gpt_4o-mini": lambda user_request, system_prompt: get_openai_chat_response(
        user_request, system_prompt, model="gpt-4o-mini"
    ),
    "open_ai_gpt_4o-mini-fixed": lambda user_request, system_prompt: get_openai_chat_response(
        user_request, system_prompt, model="gpt-4o-mini", temperature=0
    ),
}


async def get_openai_chat_response(user_request, system_prompt, model="gpt-4o-mini", temperature=0.5):
    print("Getting LLM response")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_request},
    ]
    response = await open_ai_client.chat.completions.create(
        model=model, messages=messages, temperature=temperature
    )
    print("Got LLM response")

    return response.choices[0].message.content




async def get_chat_response(user_request, system_prompt, llm_fn):
    return await llm_fn(user_request, system_prompt)
    


async def topical_guardrail(user_request, topic):
    print("Checking topical guardrail")
    messages = [
        {
            "role": "system",
            "content": "Vai trò của bạn là đánh giá xem câu hỏi của người dùng có được phép hay không. Chủ đề được phép là {topic}. Nếu chủ đề được cho phép, hãy nói 'được phép' nếu không hãy nói 'không được phép'. không nói thêm gì khác",
        },
        {"role": "user", "content": user_request},
    ]
    response = await open_ai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0,
        max_tokens=300,
    )
    print("Got guardrail response")
    print(response.choices[0].message.content)
    return response.choices[0].message.content


async def execute_chat_with_guardrail(
    user_request, topic, llm_fn, inspiration="", system_prompt=""
):
    topical_guardrail_task = asyncio.create_task(
        topical_guardrail(user_request, topic)
    )
    chat_task = asyncio.create_task(
        get_chat_response(user_request, system_prompt, llm_fn)
    )

    while True:
        done, _ = await asyncio.wait(
            [topical_guardrail_task, chat_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        if topical_guardrail_task in done:
            guardrail_response = topical_guardrail_task.result()
            if guardrail_response == "không được phép":
                chat_task.cancel()
                print("Topical guardrail triggered")
                return f"Tôi chỉ có thể hỗ trợ {topic}, {inspiration}"
            elif chat_task in done:
                chat_response = chat_task.result()
                return chat_response
        else:
            await asyncio.sleep(
                0.1
            )  # sleep for a bit before checking the tasks again
