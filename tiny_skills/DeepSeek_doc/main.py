import apikey
from openai import OpenAI


def first_call():
    client = OpenAI(
        api_key=apikey.DS_API_key,
        base_url="https://api.deepseek.com",
    )

    response = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=[
            {"role": "system", "content": "你是一个强大的AI助手"},
            {"role": "user", "content": "Hello"},
        ],
        stream=False,
        reasoning_effort="high",  # 思考程度
        extra_body={"thinking": {"type": "enabled"}},
    )

    print(response.choices[0].message.content)


def no_thought_chain():
    client = OpenAI(
        api_key=apikey.DS_API_key,
        base_url="https://api.deepseek.com",
    )

    messages = [{"role": "user", "content": "9.11 and 9.8, which is greater?"}]

    # 第一轮对话
    response = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=messages,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}},
    )

    reasoning_content = response.choices[0].message.reasoning_content
    content = response.choices[0].message.content

    print("Turn 1: (reasoning)", reasoning_content)
    print("Turn 1: (content)", content)

    # 需要把本次的回答也装入message
    messages.append({"role": "assistant", "content": content})
    # 如果想要加入思考内容也很简单
    """
    messages.append({   
        "role": "assistant", 
        "reasoning_content": reasoning_content, 
        "content": content
    })
    """

    # 第二轮对话, 思考内容被忽略, 注意只需要加入输出内容,没有思考内容!
    messages.append(
        {"role": "user", "content": "How many Rs are there in the word 'strawberry'?"}
    )
    response = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=messages,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}},
    )

    reasoning_content = response.choices[0].message.reasoning_content
    content = response.choices[0].message.content

    print("Turn 2: (reasoning)", reasoning_content)
    print("Turn 2: (content)", content)
    # 后续轮次...

    print("message: ", messages)


def json_output():
    import json

    client = OpenAI(
        api_key=apikey.DS_API_key,
        base_url="https://api.deepseek.com",
    )

    system_prompt = """
    The user will provide some exam text. Please parse the "question" and "answer" and output them in JSON format. 

    EXAMPLE INPUT: 
    Which is the highest mountain in the world? Mount Everest.

    EXAMPLE JSON OUTPUT:
    {
        "question": "Which is the highest mountain in the world?",
        "answer": "Mount Everest"
    }
    """

    user_prompt = "Which is the longest river in the world? The Nile River."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    response = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=messages,
        response_format={"type": "json_object"},  # 注意这里指定了格式!
    )

    print(json.loads(response.choices[0].message.content))


def use_tool():
    from openai import OpenAI

    def send_messages(messages):
        response = client.chat.completions.create(
            model="deepseek-v4-pro", messages=messages, tools=tools
        )
        return response.choices[0].message

    client = OpenAI(
        api_key=apikey.DS_API_key,
        base_url="https://api.deepseek.com",
    )

    # 手动写好工具列表
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather of a location, the user should supply a location first.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        }
                    },
                    "required": ["location"],
                },
            },
        },
    ]

    messages = [{"role": "user", "content": "How's the weather in Hangzhou, Zhejiang?"}]
    message = send_messages(messages)
    print(f"User>\t {messages[0]['content']}")

    tool = message.tool_calls[0]
    messages.append(message)

    messages.append({"role": "tool", "tool_call_id": tool.id, "content": "24℃"})
    message = send_messages(messages)
    print(f"Model>\t {message.content}")


if __name__ == "__main__":
    # first_call()
    # no_thought_chain()
    # json_output()
    use_tool()
    pass
