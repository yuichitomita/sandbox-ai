from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,  # システムメッセージ
    HumanMessage,  # 人間の質問
    AIMessage  # ChatGPTの返答
)

message = "「名探偵コナン」の映画風タイトルを一つ考えて"
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=message)
]
for temperature in [0, 0.5, 1.0, 2.0]:
    print(f'==== temp: {temperature}')
    llm = ChatOpenAI(model_name="gpt-4",temperature=temperature)

    # llm = ChatOpenAI(temperature=temperature)
    for i in range(5):
        print(llm(messages).content)
       