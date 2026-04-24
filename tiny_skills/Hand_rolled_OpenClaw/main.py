import apikey
from zai import ZhipuAiClient
import os
import time


# 示例1 循环聊天
def example_chat():
    client = ZhipuAiClient(api_key=apikey.zhipuAPIkey)  # 填入密钥

    while True:
        # 1 获取输入
        user_input = input("\n你: ")

        # 2 创建聊天
        response = client.chat.completions.create(
            model="glm-4.7-flash",  # 响应速度快的模型
            messages=[
                # {
                #     "role": "system",
                #     "content": "你是xxx",
                # },
                {
                    "role": "user",
                    "content": user_input,
                },
            ],
            temperature=1.0,
        )

        # 3 获取回复
        print(response.choices[0].message.content)

        # 需要手动按 Ctrl+C 退出


def chat_has_memory():
    client = ZhipuAiClient(api_key=apikey.zhipuAPIkey)

    messages = []  # 存取记忆
    while True:
        # 存用户输入
        user_input = input("\n你: ")
        messages.append({"role": "user", "content": user_input})
        # 调用API请求
        response = client.chat.completions.create(
            model="glm-4.7-flash",  # 响应速度快的模型
            messages=messages,
        )
        # 存模型输出
        reply = response.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})

        print("AI: ", reply)


def chat_agent():
    client = ZhipuAiClient(api_key=apikey.zhipuAPIkey)
    # 提前注入 agent 提示词
    messages = [
        {
            "role": "system",
            "content": """
你的目标是完成用户的任务, 你必须选择下面的其中一种格式进行回复:
1. 如果你认为需要执行命令, 则输出: '命令: XXX', 这里的 XXX 是命令本身, 不要有任何的格式, 不要解释;
2. 如果你认为不需要执行命令, 则输出: '完成: XXX', 这里的 XXX 是你的总结信息.
""",
        }
    ]

    while True:
        # 存用户输入
        user_input = input("\n你: ")
        messages.append({"role": "user", "content": user_input})

        print("\n--- Agent 循环开始! ---")

        while True:  # Agent 循环! 对用户的单次输入, 会执行运行多次! 直到完成满足要求!
            # 调用API请求
            response = client.chat.completions.create(
                model="glm-5",
                messages=messages,
            )

            # 存模型输出
            reply = response.choices[0].message.content
            messages.append({"role": "assistant", "content": reply})

            print("[AI](过程消息):", reply)

            # 如果回复了'完成'那么就显示给用户
            if reply.strip().startswith("完成:"):
                print("\n---agent 循环结束 最终成果报告---")
                print(f"AI:{reply.strip().split('完成:')[1].strip()}")
                break

            # 如果没有回复完成, 还需要继续执行
            command = reply.strip().split("命令:")[1].strip()
            command_result = os.popen(command).read()  # 用系统运行命令!

            content = f"执行完毕(操作系统的终端输出是): {command_result}"

            print(f"[Agent]: {content}")
            messages.append({"role": "user", "content": content})
            # 把系统终端的输出继续作为用户输入, 提交给模型, 这样就可以多轮运行直到任务完成了

            print("[SYSTEM]: 防止请求过快, 睡5s")
            time.sleep(5)


def chat_agent_with_skill():
    # API创建客户端
    client = ZhipuAiClient(api_key=apikey.zhipuAPIkey)
    # 读取同目录下的 智能体md 和 技能md 注入到系统提示词里
    agent_md = open("Agent.md", "r").read()
    skill_md = open("Skill.md", "r").read()
    messages = [{"role": "system", "content": agent_md + skill_md}]

    while True:
        # 存用户输入
        user_input = input("\n你: ")
        messages.append({"role": "user", "content": user_input})

        print("\n---------- Agent 循环开始! ----------")

        while True:  # Agent 循环! 对用户的单次输入, 会执行运行多次! 直到完成满足要求!
            # 调用API请求
            response = client.chat.completions.create(
                model="glm-5.1",
                messages=messages,
            )

            # 存模型输出
            reply = response.choices[0].message.content
            messages.append({"role": "assistant", "content": reply})

            print("[AI](过程消息):", reply)

            # 如果回复了'完成'那么就显示给用户
            if reply.strip().startswith("完成:"):
                print("\n------agent 循环结束 最终成果报告------")
                print(f"AI:{reply.strip().split('完成:')[1].strip()}")
                break

            # 如果没有回复完成, 还需要继续执行
            command = reply.strip().split("命令:")[1].strip()
            command_result = os.popen(command).read()  # 用系统运行命令!

            content = f"执行完毕(操作系统的终端输出是): {command_result}"

            print(f"[Agent]: {content}")
            messages.append({"role": "user", "content": content})
            # 把系统终端的输出继续作为用户输入, 提交给模型, 这样就可以多轮运行直到任务完成了

            print("[SYSTEM]: 防止请求过快, 睡5s")
            time.sleep(5)


if __name__ == "__main__":
    # example_chat()
    # chat_has_memory()
    # chat_agent()
    chat_agent_with_skill()
