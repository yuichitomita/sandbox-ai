import os
from os.path import join, dirname
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.evaluation.qa import QAEvalChain

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

os.environ["OPENAI_API_KEY"] = os.environ.get("API_KEY")

# 質問応答のプロンプトテンプレートの準備
prompt = PromptTemplate(
    template="Question: {question}\nAnswer:", 
    input_variables=["question"]
)

# 質問応答のLLMチェーンの準備
chain = LLMChain(
    llm=OpenAI(temperature=0), 
    prompt=prompt
)

# 評価データセットの準備
examples = [
    {
        "question": "テニスボールが2ケースあります。1ケースにテニスボールが3個入っています。今何個のテニスボールを持っていますか？",
        "answer": "6"
    },
    {
        "question": '次の文はもっともらしいですか？もっともらしくないですか？ "カレーは飲み物"',
        "answer": "もっともらしくない"
    },
     {
        "question": '次の文はもっともらしいですか？もっともらしくないですか？ "カレーは飲み物"',
        "answer": "もっともらしくない"
    }
]

# 予測
predictions = chain.apply(examples)
print(predictions)

# 言語モデルによる評価
eval_chain = QAEvalChain.from_llm(
    llm=OpenAI(temperature=0)
)
graded_outputs = eval_chain.evaluate(
    examples, 
    predictions, 
    question_key="question", 
    prediction_key="text"
)
print(graded_outputs)

# 結果を整形して出力
for i, eg in enumerate(examples):
    print(f"Example {i}:")
    print("Question: " + eg['question'])
    print("Real Answer: " + eg['answer'])
    print("Predicted Answer: " + predictions[i]['text'])
    print("Predicted Grade: " + graded_outputs[i]['results'])
    print()