import os
# OpenAIのChatGPTを使用する方はこちら（APIキーと呼ばれる、サービスを利用するための鍵を取る必要があります）
os.environ["OPENAI_API_KEY"]=""

"""## LLMと埋め込みモデルの準備
"""

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# LLMの準備
Settings.llm = OpenAI(
    model="gpt-3.5-turbo",
    # model="gpt-4-turbo", # 最新モデルはこちら
    temperature=0
    )

# 埋め込みモデルの準備
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-large",  # 日本語に弱いが今回はこのモデル（サイトで紹介されていたモデル）を使用します
)

"""## dataディレクトリを作成し、ドキュメントを追加
左のアイコンでファイルの一覧を表示し、右クリック「新しいフォルダ」でdataフォルダを作成し、フォルダ内にドキュメントを追加します。

## 6行のコードを実行
先生から提供して頂いたコードを実行します。
"""

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("data").load_data()

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

response_1 = query_engine.query("桃太郎は何を連れて旅をしましたか？") #ここに追加したドキュメントに対しての質問を追加
#response_2 = query_engine.query("")

print(response_1)
#print(response_2)

"""## 参考サイト
リンクをクリックするとリダイレクトの警告文がでますが、表示されたリンクをクリックしてサイトにアクセスしてください。

[LlamaIndexのGemini統合を試す](https://note.com/npaka/n/n68bd11eac933)
"""