# %%
# pip install -qU langchain-qdrant

# %%
import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = "YOUR KEY"

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# %%
vector_embed = embeddings.embed_query("What is the capital of France?")

# %%
len(vector_embed)

# %%
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams


# %%

client = QdrantClient(url="http://localhost:6333", )


# %%

client.create_collection(
    collection_name="chatbot_collection",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)


# %%

vector_store = QdrantVectorStore(
    client=client,
    collection_name="chatbot_collection",
    embedding=embeddings,
)

# %%
from uuid import uuid4

from langchain_core.documents import Document

document_1 = Document(
    page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
    metadata={"source": "tweet", 'user': '111'},
)

document_2 = Document(
    page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees Fahrenheit.",
    metadata={"source": "news"},
)

documents = [
    document_1,
    document_2
]
uuids = [str(uuid4()) for _ in range(len(documents))]

# %%
vector_store.add_documents(documents=documents, ids=uuids)

# %%
results = vector_store.similarity_search(
    "LangChain provides abstractions to make working with LLMs easy", k=1
)
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")

# %%
results = vector_store.similarity_search_with_score(
    "LangChain provides abstractions to make working with LLMs easy", k=1
)

# %%
results

# %%
text1 = """

Tổng Bí thư: Sáp nhập tỉnh giàu với tỉnh nghèo đòi hỏi lãnh đạo phải công tâm
Tổng Bí thư Tô Lâm lưu ý việc sáp nhập tỉnh miền núi với đồng bằng hay tỉnh giàu với nghèo đòi hỏi lãnh đạo phải công tâm, có tầm nhìn nhằm cân đối nguồn lực, hài hòa lợi ích phát triển.

Trong bài viết "Sức mạnh của đoàn kết" ngày 29/6, Tổng Bí thư Tô Lâm nhấn mạnh trong giai đoạn cả nước đang "sắp xếp lại giang sơn", tinh thần đoàn kết càng phải được phát huy mạnh mẽ hơn bao giờ hết.

Đoàn kết là truyền thống quý báu trong lịch sử dựng nước và giữ nước của dân tộc Việt Nam. Nguyễn Trãi tổng kết "đẩy thuyền là dân, lật thuyền cũng là dân" hay "thuyền bị lật mới tin rằng dân như nước". Cha ông từng đúc kết "dễ trăm lần không dân cũng chịu, khó vạn lần dân liệu cũng xong".

Chủ tịch Hồ Chí Minh là người kế thừa xuất sắc tinh thần "nước lấy dân làm gốc" của dân tộc. "Chúng ta có được cơ đồ, tiềm lực, vị thế và uy tín quốc tế như ngày hôm nay phần nhiều do sức mạnh của khối đại đoàn kết toàn dân tộc bồi đắp nên", Tổng Bí thư viết.

Lịch sử cũng không thiếu những bài học về mất đoàn kết, như các cuộc đấu tranh chống thực dân cuối thế kỷ 19 của nhân dân thất bại có nguyên nhân sâu xa là cả nước không đoàn kết thành khối thống nhất. Nhiều cuộc khởi nghĩa anh dũng bị dập tắt vì thiếu sự phối hợp, đồng lòng giữa các lực lượng và lãnh tụ đương thời.

Sắp xếp bộ máy cần hy sinh lợi ích cá nhân

Hiện nay, Việt Nam quyết liệt cải cách mạnh mẽ tổ chức bộ máy của hệ thống chính trị, sắp xếp lại đơn vị hành chính các cấp và vận hành chính quyền địa phương hai cấp. Mục tiêu là tinh giản bộ máy, nâng cao hiệu lực, hiệu quả quản lý, quản trị nhà nước, đồng thời phân cấp, phân quyền mạnh mẽ hơn cho địa phương.

Mô hình này không chỉ cắt bỏ cấp trung gian không cần thiết, mà quan trọng hơn là tổ chức lại không gian phát triển bền vững, để chính quyền gần dân, sát dân, vì dân, phục vụ nhân dân được tốt hơn. Trung ương cũng phân định rõ thẩm quyền và trao quyền chủ động nhiều hơn cho địa phương để mỗi nơi năng động, sáng tạo, phát triển phù hợp thực tiễn.

Tuy nhiên, việc tinh gọn tổ chức bộ máy và sắp xếp lại đơn vị hành chính cũng tác động, ảnh hưởng đến một bộ phận cán bộ, đảng viên, công chức. Điều đó đòi hỏi sự công minh, đồng thuận và quyết tâm chính trị rất cao và đặc biệt là sự hy sinh lợi ích cá nhân, theo Tổng Bí thư.
"""

# %%
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
texts = text_splitter.split_text(text1)

# %%
texts

# %%
for text in texts:
    Document(
        page_content=text,
        metadata={"source": "news", 'user': '111'}
    )
    # insert vector to vector store
    vector_store.add_documents(
        documents=[Document(page_content=text, metadata={"page": "1"})],
        ids=[str(uuid4())]
    )

# %%


# %%
from langchain_text_splitters import CharacterTextSplitter
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=400, chunk_overlap=50
)
texts = text_splitter.split_text(text1)

# %%
texts

# %%
