import os
import asyncio
from uuid import uuid4
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_core.documents import Document

# Cấu hình OpenAI API Key
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "YOUR KEY"
   
class RAGIndexer:
    def __init__(self, collection_name="chatbot_collection"):
        self.collection_name = collection_name
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.client = QdrantClient(url="http://localhost:6333")
        self.vector_store = None
        
    def setup_vector_store(self):
        """Thiết lập vector store và tạo collection nếu chưa có"""
        try:
            # Kiểm tra xem collection đã tồn tại chưa
            collections = self.client.get_collections()
            collection_exists = any(col.name == self.collection_name for col in collections.collections)
            
            if not collection_exists:
                print(f"Tạo collection mới: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
                )
            else:
                print(f"Collection {self.collection_name} đã tồn tại")
                
            self.vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embeddings,
            )
            return True
        except Exception as e:
            print(f"Lỗi khi thiết lập vector store: {e}")
            return False
    
    async def load_pdf(self, pdf_path):
        """Đọc nội dung từ file PDF"""
        try:
            print(f"Đang đọc PDF: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            pages = []
            async for page in loader.alazy_load():
                pages.append(page)
            print(f"Đã đọc {len(pages)} trang từ PDF")
            return pages
        except Exception as e:
            print(f"Lỗi khi đọc PDF: {e}")
            return []
    
    def split_documents(self, documents, chunk_size=1000, chunk_overlap=200):
        """Chia nhỏ documents thành các chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        all_chunks = []
        for doc in documents:
            chunks = text_splitter.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                chunk_doc = Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_id": i,
                        "source": doc.metadata.get("source", "unknown"),
                        "page": doc.metadata.get("page", 0)
                    }
                )
                all_chunks.append(chunk_doc)
        
        print(f"Đã chia thành {len(all_chunks)} chunks")
        return all_chunks
    
    def add_documents_to_vector_store(self, documents):
        """Thêm documents vào vector store"""
        try:
            if not self.vector_store:
                print("Vector store chưa được thiết lập")
                return False
                
            # Tạo unique IDs cho mỗi document
            uuids = [str(uuid4()) for _ in range(len(documents))]
            
            print(f"Đang thêm {len(documents)} documents vào vector store...")
            self.vector_store.add_documents(documents=documents, ids=uuids)
            print("Đã thêm documents thành công!")
            return True
        except Exception as e:
            print(f"Lỗi khi thêm documents: {e}")
            return False
    
    async def index_pdf(self, pdf_path, chunk_size=1000, chunk_overlap=200):
        """Quy trình hoàn chỉnh: đọc PDF, chia nhỏ và index vào vector store"""
        print("=== BẮT ĐẦU QUY TRÌNH INDEXING ===")
        
        # 1. Thiết lập vector store
        if not self.setup_vector_store():
            return False
        
        # 2. Đọc PDF
        pages = await self.load_pdf(pdf_path)
        if not pages:
            return False
        
        # 3. Chia nhỏ documents
        chunks = self.split_documents(pages, chunk_size, chunk_overlap)
        
        # 4. Thêm vào vector store
        success = self.add_documents_to_vector_store(chunks)
        
        if success:
            print("=== HOÀN THÀNH INDEXING ===")
            print(f"Đã index {len(chunks)} chunks từ {len(pages)} trang PDF")
        
        return success
    
    def search_similar(self, query, k=5):
        """Tìm kiếm documents tương tự"""
        if not self.vector_store:
            print("Vector store chưa được thiết lập")
            return []
        
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            return results
        except Exception as e:
            print(f"Lỗi khi tìm kiếm: {e}")
            return []

# Hàm main để chạy indexing
async def main():
    # Đường dẫn đến file PDF
    pdf_path = "/Users/sonnguyen/Documents/llm-rag/RAG/2506.21538v1.pdf"
    
    # Tạo indexer
    indexer = RAGIndexer()
    
    # Chạy indexing
    success = await indexer.index_pdf(pdf_path, chunk_size=400, chunk_overlap=80)
    
    if success:
        print("\n=== TEST TÌM KIẾM ===")
        # Test tìm kiếm
        query = "What is this paper about?"
        results = indexer.search_similar(query, k=3)
        
        print(f"Kết quả tìm kiếm cho: '{query}'")
        for i, (doc, score) in enumerate(results):
            print(f"\n{i+1}. Score: {score:.4f}")
            print(f"Content: {doc.page_content[:200]}...")
            print(f"Metadata: {doc.metadata}")

if __name__ == "__main__":
    asyncio.run(main())
