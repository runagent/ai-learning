import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Cấu hình OpenAI API Key
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "YOUR KEY"
   
class RAGChatbot:
    def __init__(self, collection_name="chatbot_collection"):
        self.collection_name = collection_name
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.client = QdrantClient(url="http://localhost:6333")
        self.llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.1)
        self.vector_store = None
        self.retriever = None
        self.rag_chain = None
        self.setup_rag_chain()
    
    def setup_rag_chain(self):
        """Thiết lập RAG chain"""
        try:
            # Thiết lập vector store
            self.vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embeddings,
            )
            
            # Tạo retriever
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            # Tạo prompt template
            template = """Bạn là một trợ lý AI thông minh và hữu ích. Hãy trả lời câu hỏi dựa trên thông tin được cung cấp từ tài liệu.

Thông tin từ tài liệu:
{context}

Câu hỏi: {question}

Hướng dẫn:
1. Chỉ trả lời dựa trên thông tin có trong tài liệu được cung cấp
2. Nếu không tìm thấy thông tin liên quan, hãy nói rằng bạn không tìm thấy thông tin đó trong tài liệu
3. Trả lời bằng tiếng Việt một cách rõ ràng và chi tiết
4. Nếu có thể, hãy trích dẫn các phần quan trọng từ tài liệu

Câu trả lời:"""

            self.prompt = ChatPromptTemplate.from_template(template)
            
            # Tạo RAG chain
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
            
            self.rag_chain = (
                {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
            
            print("RAG chain đã được thiết lập thành công!")
            return True
            
        except Exception as e:
            print(f"Lỗi khi thiết lập RAG chain: {e}")
            return False
    
    def get_relevant_documents(self, query, k=5):
        """Lấy các documents liên quan đến query"""
        try:
            if not self.vector_store:
                print("Vector store chưa được thiết lập")
                return []
            
            results = self.vector_store.similarity_search_with_score(query, k=k)
            return results
        except Exception as e:
            print(f"Lỗi khi tìm kiếm documents: {e}")
            return []
    
    def generate_answer(self, question):
        """Tạo câu trả lời cho câu hỏi"""
        try:
            if not self.rag_chain:
                return "Lỗi: RAG chain chưa được thiết lập"
            
            print(f"Đang xử lý câu hỏi: {question}")
            
            # Lấy documents liên quan để debug
            relevant_docs = self.get_relevant_documents(question, k=3)
            print(f"\nTìm thấy {len(relevant_docs)} documents liên quan:")
            for i, (doc, score) in enumerate(relevant_docs):
                print(f"{i+1}. Score: {score:.4f} - {doc.page_content[:100]}...")
            
            # Tạo câu trả lời
            answer = self.rag_chain.invoke(question)
            return answer
            
        except Exception as e:
            print(f"Lỗi khi tạo câu trả lời: {e}")
            return f"Xin lỗi, đã có lỗi xảy ra: {str(e)}"
    
    def chat_loop(self):
        """Vòng lặp chat với người dùng"""
        print("=== RAG CHATBOT ===")
        print("Hãy đặt câu hỏi về tài liệu. Gõ 'quit' để thoát.\n")
        
        while True:
            try:
                question = input("Bạn: ").strip()
                
                if question.lower() in ['quit', 'exit', 'thoát']:
                    print("Tạm biệt!")
                    break
                
                if not question:
                    continue
                
                print("\nBot đang suy nghĩ...")
                answer = self.generate_answer(question)
                print(f"\nBot: {answer}\n")
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\nTạm biệt!")
                break
            except Exception as e:
                print(f"Lỗi: {e}")

def main():
    """Hàm main để chạy chatbot"""
    print("Đang khởi tạo RAG Chatbot...")
    
    # Tạo chatbot
    chatbot = RAGChatbot()
    
    # Kiểm tra xem có dữ liệu trong vector store không
    try:
        test_results = chatbot.get_relevant_documents("test", k=1)
        if not test_results:
            print("⚠️  Cảnh báo: Vector store có vẻ trống. Hãy chạy index_vector.py trước để thêm dữ liệu.")
            return
    except Exception as e:
        print(f"⚠️  Lỗi khi kiểm tra vector store: {e}")
        return
    
    # Bắt đầu chat
    chatbot.chat_loop()

if __name__ == "__main__":
    main()
