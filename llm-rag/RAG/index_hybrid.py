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
import json

# Cấu hình OpenAI API Key
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "YOUR KEY"
   
class HybridRAGIndexer:
    def __init__(self, collection_name="hybrid_rag_collection"):
        self.collection_name = collection_name
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.client = QdrantClient(url="http://localhost:6333")
        self.vector_store = None
        self.document_store = {}  # Store full documents by source
        
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
    
    def store_full_documents(self, documents):
        """Lưu trữ từng page riêng biệt theo source và page number để sử dụng trong hybrid retrieval"""
        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            page_num = doc.metadata.get("page", 0)
            
            # Tạo key unique cho mỗi page
            page_key = f"{source}#page_{page_num}"
            
            if source not in self.document_store:
                self.document_store[source] = {}
            
            # Lưu từng page riêng biệt
            self.document_store[source][page_num] = {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
                "page_key": page_key
            }
        
        total_pages = sum(len(pages) for pages in self.document_store.values())
        print(f"Đã lưu trữ {total_pages} pages từ {len(self.document_store)} sources")
    
    def split_documents(self, documents, chunk_size=1000, chunk_overlap=200):
        """Chia nhỏ documents thành các chunks với metadata đầy đủ cho hybrid retrieval"""
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
                        "page": doc.metadata.get("page", 0),
                        "total_chunks": len(chunks),
                        "document_type": "chunk",  # Đánh dấu đây là chunk
                        "original_content_length": len(doc.page_content)
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
    
    def save_document_store(self, file_path="document_store.json"):
        """Lưu document store vào file để sử dụng trong retrieval"""
        try:
            # Document store đã ở format có thể serialize (page-based structure)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.document_store, f, ensure_ascii=False, indent=2)
            
            print(f"Đã lưu document store vào {file_path}")
            return True
        except Exception as e:
            print(f"Lỗi khi lưu document store: {e}")
            return False
    
    async def index_pdf_hybrid(self, pdf_path, chunk_size=1000, chunk_overlap=200):
        """Quy trình hoàn chỉnh cho hybrid indexing: lưu cả chunks và full documents"""
        print("=== BẮT ĐẦU QUY TRÌNH HYBRID INDEXING ===")
        
        # 1. Thiết lập vector store
        if not self.setup_vector_store():
            return False
        
        # 2. Đọc PDF
        pages = await self.load_pdf(pdf_path)
        if not pages:
            return False
        
        # 3. Lưu trữ full documents cho hybrid retrieval
        self.store_full_documents(pages)
        
        # 4. Chia nhỏ documents thành chunks
        chunks = self.split_documents(pages, chunk_size, chunk_overlap)
        
        # 5. Thêm chunks vào vector store
        success = self.add_documents_to_vector_store(chunks)
        
        # 6. Lưu document store
        if success:
            self.save_document_store()
        
        if success:
            print("=== HOÀN THÀNH HYBRID INDEXING ===")
            print(f"Đã index {len(chunks)} chunks từ {len(pages)} trang PDF")
            print(f"Đã lưu trữ {len(self.document_store)} full documents")
        
        return success
    
    def search_similar_chunks(self, query, k=5):
        """Tìm kiếm chunks tương tự (bước đầu của hybrid retrieval)"""
        if not self.vector_store:
            print("Vector store chưa được thiết lập")
            return []
        
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            return results
        except Exception as e:
            print(f"Lỗi khi tìm kiếm: {e}")
            return []
    
    def get_source_documents_from_chunks(self, chunk_results):
        """Lấy danh sách các source documents từ chunks tìm được"""
        sources = set()
        for doc, score in chunk_results:
            source = doc.metadata.get("source", "unknown")
            sources.add(source)
        
        return list(sources)

# Hàm main để chạy hybrid indexing
async def main():
    # Đường dẫn đến file PDF
    pdf_path = "/Users/sonnguyen/Documents/llm-rag/RAG/2506.21538v1.pdf"
    
    # Tạo hybrid indexer
    indexer = HybridRAGIndexer()
    
    # Chạy hybrid indexing
    success = await indexer.index_pdf_hybrid(pdf_path, chunk_size=400, chunk_overlap=80)
    
    if success:
        print("\n=== TEST HYBRID SEARCH ===")
        # Test tìm kiếm chunks
        query = "What is this paper about?"
        chunk_results = indexer.search_similar_chunks(query, k=3)
        
        print(f"Kết quả tìm kiếm chunks cho: '{query}'")
        for i, (doc, score) in enumerate(chunk_results):
            print(f"\n{i+1}. Score: {score:.4f}")
            print(f"Content: {doc.page_content[:200]}...")
            print(f"Metadata: {doc.metadata}")
        
        # Test lấy source documents
        sources = indexer.get_source_documents_from_chunks(chunk_results)
        print(f"\nSource documents được xác định: {sources}")
        print(f"Số lượng full documents có sẵn: {len(indexer.document_store)}")

if __name__ == "__main__":
    asyncio.run(main())
