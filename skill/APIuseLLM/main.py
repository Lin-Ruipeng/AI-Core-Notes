import apikey  # 密钥


def test():
    # uv venv # 先创建环境
    # uv pip install zai-sdk # 安装依赖
    # . ./.venv/bin/activate # 激活环境
    # 记得可以选择vscode的解释器路径为 ./.venv/bin/python
    # uv run main.py # 运行
    import zai

    print(zai.__version__)


# 示例1 一般用法
def example_chat():
    from zai import ZhipuAiClient

    # 初始化客户端
    client = ZhipuAiClient(api_key=apikey.zhipuAPIkey)  # 填入密钥

    # 创建聊天
    response = client.chat.completions.create(
        model="glm-5.1",
        messages=[
            {
                "role": "system",
                "content": "你是用户的前女友，是他甩了你",
            },
            {
                "role": "user",
                "content": "你怎么又来找我了？",
            },
        ],
        temperature=1.0,
    )

    # 获取回复
    print(response.choices[0].message.content)


# 示例2 流式输出(加思考模式)
def example_chat_stream():
    from zai import ZhipuAiClient

    client = ZhipuAiClient(api_key=apikey.zhipuAPIkey)

    completion = client.chat.completions.create(
        model="glm-5.1",
        messages=[
            {"role": "user", "content": "你谁啊？"},
        ],
        # 开启思考模式
        extra_body={"enable_thinking": True},
        # 开启流式输出
        stream=True,
        temperature=1.0,
    )

    for chunk in completion:
        # 先收集思考过程
        if (
            hasattr(chunk.choices[0].delta, "reasoning_content")
            and chunk.choices[0].delta.reasoning_content
        ):
            print(chunk.choices[0].delta.reasoning_content, end="", flush=True)
        # 再收集正式回复
        if (
            hasattr(chunk.choices[0].delta, "content")
            and chunk.choices[0].delta.content
        ):
            print(chunk.choices[0].delta.content, end="", flush=True)


def example_chat_multi_modal():
    import zai
    import base64  # 用于对图片文件进行编码

    # 读取图片文件到内存
    with open("avatar.png", "rb") as image_file:
        # 编码为base64格式 然后解码为utf-8的文本以便于传输
        encoding_string = base64.b64encode(image_file.read()).decode("utf-8")

    client = zai.ZhipuAiClient(api_key=apikey.zhipuAPIkey)

    response = client.chat.completions.create(
        model="glm-4.6v",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "这张照片是什么？",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoding_string}",
                        },
                    },
                ],
            }
        ],
        # stream=True,
        temperature=1.0,
    )

    print(response.choices[0].message.content)


if __name__ == "__main__":
    # test()
    # example_chat()
    # example_chat_stream()
    example_chat_multi_modal()
    pass
