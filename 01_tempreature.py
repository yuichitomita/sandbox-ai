from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,  # システムメッセージ
    HumanMessage  # 人間の質問
)

message = "「名探偵コナン」の映画風タイトルを一つ考えて"
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=message)
]
for temperature in [0, 1, 2]:
    print(f'==== temp: {temperature}')
    llm = ChatOpenAI(model_name="gpt-4",temperature=temperature)

    for i in range(5):
        print(llm(messages).content)
       