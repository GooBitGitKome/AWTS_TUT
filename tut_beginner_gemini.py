import os
# GoogleのGeminiを使用する方はこちら（APIキーと呼ばれる、サービスを利用するための鍵を取る必要があります）
os.environ["GOOGLE_API_KEY"] = ""

"""## LLMと埋め込みモデルの準備
"""

from llama_index.core import Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

# LLMの準備
Settings.llm = Gemini(
    model_name="models/gemini-pro",
)

# 埋め込みモデルの準備
Settings.embed_model = GeminiEmbedding(
    model_name="models/embedding-001",  # 日本語に弱いが今回はこのモデル（サイトで紹介されていたモデル）を使用します
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