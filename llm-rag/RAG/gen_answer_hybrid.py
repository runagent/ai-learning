import os
import json
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

# Cấu hình OpenAI API Key
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "YOUR KEY"
    
class HybridRAGChatbot:
    def __init__(self, collection_name="hybrid_rag_collection", document_store_path="document_store.json"):
        self.collection_name = collection_name
        self.document_store_path = document_store_path
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.client = QdrantClient(url="http://localhost:6333")
        self.llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.1)
        self.vector_store = None
        self.retriever = None
        self.document_store = {}
        self.rag_chain = None
        self.setup_hybrid_rag_chain()
    
    def load_document_store(self):
        """Tải document store từ file (page-based structure)"""
        try:
            with open(self.document_store_path, 'r', encoding='utf-8') as f:
                self.document_store = json.load(f)
            
            # Document store đã ở format page-based: {source: {page_num: {page_content, metadata, page_key}}}
            total_pages = sum(len(pages) for pages in self.document_store.values())
            print(f"Đã tải document store với {len(self.document_store)} sources, {total_pages} pages")
            return True
        except FileNotFoundError:
            print(f"Không tìm thấy file {self.document_store_path}")
            print("Hãy chạy index_hybrid.py trước để tạo document store")
            return False
        except Exception as e:
            print(f"Lỗi khi tải document store: {e}")
            return False
    
    def setup_hybrid_rag_chain(self):
        """Thiết lập Hybrid RAG chain"""
        try:
            # Tải document store
            if not self.load_document_store():
                return False
            
            # Thiết lập vector store
            self.vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embeddings,
            )
            
            # Tạo retriever cho chunks
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            # Tạo prompt template cho hybrid approach với citations
            template = """Bạn là một trợ lý AI thông minh và hữu ích. Hãy trả lời câu hỏi dựa trên thông tin được cung cấp từ tài liệu.

Thông tin từ tài liệu (đã được lọc và mở rộng từ các chunks liên quan):
{context}

Câu hỏi: {question}

Hướng dẫn:
1. Sử dụng thông tin từ tài liệu được cung cấp để trả lời câu hỏi
2. Nếu không tìm thấy thông tin liên quan, hãy nói rằng bạn không tìm thấy thông tin đó trong tài liệu
3. Trả lời bằng tiếng Việt một cách rõ ràng và chi tiết
4. QUAN TRỌNG: Khi trích dẫn thông tin, hãy bao gồm citation theo format [tên_file, trang X] để người đọc có thể tham khảo nguồn gốc
5. Khi trả lời, hãy tận dụng toàn bộ ngữ cảnh từ tài liệu gốc, không chỉ giới hạn trong các đoạn nhỏ
6. Cuối câu trả lời, hãy liệt kê các nguồn tham khảo đã sử dụng

Ví dụ format trả lời:
"Theo tài liệu, [thông tin chính]... [tên_file, trang X]. Ngoài ra, [thông tin bổ sung]... [tên_file, trang Y].

Nguồn tham khảo:
- [tên_file, trang X]
- [tên_file, trang Y]"

Câu trả lời:"""

            self.prompt = ChatPromptTemplate.from_template(template)
            
            # Tạo Hybrid RAG chain với custom retrieval logic
            self.rag_chain = (
                {"context": RunnablePassthrough() | self._hybrid_retrieve_and_format, "question": RunnablePassthrough()}
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
            
            print("Hybrid RAG chain đã được thiết lập thành công!")
            return True
            
        except Exception as e:
            print(f"Lỗi khi thiết lập Hybrid RAG chain: {e}")
            return False
    
    def _hybrid_retrieve_and_format(self, query):
        """
        Hybrid retrieval strategy với page-based approach:
        1. Tìm kiếm chunks tương tự
        2. Xác định pages từ chunks metadata
        3. Tải full text của các pages liên quan
        4. Kết hợp thông tin để tạo context với citations
        """
        try:
            # Bước 1: Vector search trên chunks
            chunk_results = self.vector_store.similarity_search_with_score(query, k=5)
            
            if not chunk_results:
                return "Không tìm thấy thông tin liên quan trong tài liệu."
            
            print(f"Tìm thấy {len(chunk_results)} chunks liên quan")
            
            # Bước 2: Xác định pages từ chunks metadata
            pages_with_scores = {}
            for doc, score in chunk_results:
                source = doc.metadata.get("source", "unknown")
                page_num = doc.metadata.get("page", 0)
                page_key = f"{source}#page_{page_num}"
                
                if page_key not in pages_with_scores:
                    pages_with_scores[page_key] = {
                        "source": source,
                        "page_num": page_num,
                        "scores": [],
                        "chunks": []
                    }
                
                pages_with_scores[page_key]["scores"].append(score)
                pages_with_scores[page_key]["chunks"].append(doc.page_content)
            
            print(f"Xác định được {len(pages_with_scores)} pages liên quan")
            
            # Bước 3: Tải full text của các pages và tạo context
            context_parts = []
            citation_info = []
            
            for page_key, page_info in pages_with_scores.items():
                source = page_info["source"]
                page_num = page_info["page_num"]
                avg_score = sum(page_info["scores"]) / len(page_info["scores"])
                
                # Lấy full text của page từ document store
                if source in self.document_store and str(page_num) in self.document_store[source]:
                    page_data = self.document_store[source][str(page_num)]
                    full_page_content = page_data["page_content"]
                    
                    # Tạo citation info
                    doc_name = source.split("/")[-1] if "/" in source else source
                    citation = f"[{doc_name}, trang {page_num}]"
                    citation_info.append(citation)
                    
                    # Tạo context cho page này
                    page_context = f"\n--- {citation} (Độ liên quan: {avg_score:.3f}) ---\n"
                    page_context += full_page_content
                    
                    # Thêm thông tin về chunks liên quan từ page này
                    page_context += f"\n\n[Các đoạn liên quan nhất từ trang này:]\n"
                    for i, chunk_content in enumerate(page_info["chunks"][:2]):
                        page_context += f"- Đoạn {i+1}: {chunk_content[:150]}...\n"
                    
                    context_parts.append((avg_score, page_context))
            
            # Sắp xếp theo độ liên quan và kết hợp
            context_parts.sort(key=lambda x: x[0], reverse=True)
            final_context = "\n".join([context for _, context in context_parts])
            
            # Thêm thông tin citations vào cuối context
            if citation_info:
                final_context += f"\n\n=== NGUỒN THAM KHẢO ===\n"
                for i, citation in enumerate(set(citation_info), 1):
                    final_context += f"{i}. {citation}\n"
            
            return final_context
            
        except Exception as e:
            print(f"Lỗi trong hybrid retrieval: {e}")
            return f"Lỗi khi xử lý truy vấn: {str(e)}"
    
    def get_relevant_chunks(self, query, k=5):
        """Lấy các chunks liên quan đến query (để debug)"""
        try:
            if not self.vector_store:
                print("Vector store chưa được thiết lập")
                return []
            
            results = self.vector_store.similarity_search_with_score(query, k=k)
            return results
        except Exception as e:
            print(f"Lỗi khi tìm kiếm chunks: {e}")
            return []
    
    def get_source_documents(self, sources):
        """Lấy full documents từ danh sách sources"""
        result_docs = []
        for source in sources:
            if source in self.document_store:
                result_docs.extend(self.document_store[source])
        return result_docs
    
    def generate_answer(self, question):
        """Tạo câu trả lời sử dụng hybrid RAG strategy"""
        try:
            if not self.rag_chain:
                return "Lỗi: Hybrid RAG chain chưa được thiết lập"
            
            print(f"Đang xử lý câu hỏi bằng Hybrid RAG: {question}")
            
            # Debug: Hiển thị quá trình hybrid retrieval
            print("\n=== HYBRID RETRIEVAL PROCESS ===")
            
            # Bước 1: Tìm chunks liên quan
            relevant_chunks = self.get_relevant_chunks(question, k=3)
            print(f"Bước 1 - Tìm thấy {len(relevant_chunks)} chunks liên quan:")
            for i, (doc, score) in enumerate(relevant_chunks):
                print(f"  {i+1}. Score: {score:.4f} - {doc.page_content[:100]}...")
            
            # Bước 2: Xác định sources
            sources = list(set(doc.metadata.get("source", "unknown") for doc, _ in relevant_chunks))
            print(f"Bước 2 - Xác định {len(sources)} source documents: {sources}")
            
            # Bước 3: Lấy full documents
            full_docs = self.get_source_documents(sources)
            print(f"Bước 3 - Tải {len(full_docs)} full documents")
            
            print("=== GENERATING ANSWER ===")
            
            # Tạo câu trả lời
            answer = self.rag_chain.invoke(question)
            return answer
            
        except Exception as e:
            print(f"Lỗi khi tạo câu trả lời: {e}")
            return f"Xin lỗi, đã có lỗi xảy ra: {str(e)}"
    
    def chat_loop(self):
        """Vòng lặp chat với người dùng"""
        print("=== HYBRID RAG CHATBOT ===")
        print("Chiến lược: Tìm chunks → Xác định documents → Tải full context")
        print("Hãy đặt câu hỏi về tài liệu. Gõ 'quit' để thoát.\n")
        
        while True:
            try:
                question = input("Bạn: ").strip()
                
                if question.lower() in ['quit', 'exit', 'thoát']:
                    print("Tạm biệt!")
                    break
                
                if not question:
                    continue
                
                print("\nHybrid RAG đang xử lý...")
                answer = self.generate_answer(question)
                print(f"\nBot: {answer}\n")
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\nTạm biệt!")
                break
            except Exception as e:
                print(f"Lỗi: {e}")

def main():
    """Hàm main để chạy hybrid chatbot"""
    print("Đang khởi tạo Hybrid RAG Chatbot...")
    
    # Tạo chatbot
    chatbot = HybridRAGChatbot()
    
    # Kiểm tra xem có dữ liệu trong vector store không
    try:
        test_results = chatbot.get_relevant_chunks("test", k=1)
        if not test_results:
            print("⚠️  Cảnh báo: Vector store có vẻ trống. Hãy chạy index_hybrid.py trước để thêm dữ liệu.")
            return
        
        if not chatbot.document_store:
            print("⚠️  Cảnh báo: Document store trống. Hãy chạy index_hybrid.py để tạo document store.")
            return
            
    except Exception as e:
        print(f"⚠️  Lỗi khi kiểm tra hệ thống: {e}")
        return
    
    print(f"✅ Hệ thống sẵn sàng với {len(chatbot.document_store)} source documents")
    
    # Bắt đầu chat
    chatbot.chat_loop()

if __name__ == "__main__":
    main()
