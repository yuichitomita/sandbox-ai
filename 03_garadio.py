import os

from os.path import join, dirname
from dotenv import load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import AsyncCallbackManager

import gradio as gr

# APIキーの設定
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

os.environ["OPENAI_API_KEY"] = os.environ.get("API_KEY")

# 設定プロンプト
character_setting = "YourFavoriteSystemPrompt"

# チャットプロンプトテンプレート
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(character_setting),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

# チャットモデル
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo", 
    max_tokens=512,
    temperature=0.2,
    streaming=True, 
    callback_manager=AsyncCallbackManager([StreamingStdOutCallbackHandler()])
)

# メモリ
memory = ConversationBufferWindowMemory(k=3, return_messages=True)

# 会話チェーン
conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm, verbose=True)


# フロントエンド
css = """
.message.user{
    background: #06c755 !important;
}"""

with gr.Blocks(css=css) as demo:
    # コンポーネント
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    def user(message, history):
        return "", history + [[message, None]]

    def chat(history):
        message = history[-1][0]
        response = conversation.predict(input=message)
        history[-1][1] = response
        return history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        chat, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch()
