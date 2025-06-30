Quy trình xây dựng RAG Q&A Chatbot
🧱 1. Indexing (Tiền xử lý dữ liệu)
Tải dữ liệu
– Sử dụng Document Loader để tải nội dung từ website, file, hay cơ sở dữ liệu.
– Ví dụ: tải nội dung từ một bài blog.

Chia nhỏ văn bản
– Dùng Text Splitter (như RecursiveCharacterTextSplitter) để chia văn bản dài thành các đoạn nhỏ (~1000 ký tự).
– Giúp dễ tìm kiếm và phù hợp với giới hạn context của LLM.

Tạo vector embedding
– Sử dụng mô hình Embedding (như text-embedding-3-large) để chuyển từng đoạn văn thành vector số.

Lưu vào vector store
– Dùng VectorStore (như InMemoryVectorStore, FAISS, Pinecone...) để lưu các embedding và hỗ trợ tìm kiếm theo ngữ nghĩa.

🧠 2. Retrieval + Generation (Truy xuất và tạo câu trả lời)
Phân tích truy vấn người dùng (tùy chọn)
– Dùng LLM để phân tích hoặc viết lại câu hỏi thành truy vấn tối ưu.
– Có thể thêm bộ lọc (metadata filter) để giới hạn phạm vi tìm kiếm.

Tìm kiếm ngữ nghĩa
– Truy vấn vector store để lấy các đoạn văn liên quan đến câu hỏi.

Tạo prompt và sinh câu trả lời
– Kết hợp câu hỏi và nội dung truy xuất vào một prompt định sẵn.
– Dùng LLM (như Gemini hoặc GPT-4) để sinh ra câu trả lời cuối cùng.



Cách đọc data từ file pdf:
```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("/Users/sonnguyen/Documents/llm-rag/loadpdf/2506.21538v1.pdf")
pages = []
async for page in loader.alazy_load():
    pages.append(page)
```
