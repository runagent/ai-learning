import base64
import os
from openai import OpenAI

# Set up OpenAI API key from the configuration
OPENAI_KEY = "YOUR KEY"
os.environ["OPENAI_API_KEY"] = OPENAI_KEY

# Initialize OpenAI client
client = OpenAI()

def encode_image_to_base64(image_path):
    """
    Encode an image file to base64 string
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error encoding image: {str(e)}")
        return None

def perform_ocr_with_openai(image_path, detail_level="high"):
    """
    Perform OCR on an image using OpenAI's GPT-4.1-mini vision model
    
    Args:
        image_path (str): Path to the image file
        detail_level (str): Detail level for image processing ("low", "high", or "auto")
    
    Returns:
        str: Extracted text from the image
    """
    
    # Encode image to base64
    base64_image = encode_image_to_base64(image_path)
    if not base64_image:
        return None
    
    try:
        # Create the request using Chat Completions API
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please extract all text from this image. Provide the text exactly as it appears, maintaining the original formatting and structure as much as possible. If there are tables, lists, or structured content, please preserve that structure in your response."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": detail_level
                            }
                        }
                    ]
                }
            ],
            max_tokens=4000,
            temperature=0.1
        )
        
        # Extract the text response
        extracted_text = response.choices[0].message.content
        return extracted_text
        
    except Exception as e:
        print(f"Error during OCR processing: {str(e)}")
        return None

def save_extracted_text(text, output_path):
    """
    Save extracted text to a file
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Extracted text saved to: {output_path}")
    except Exception as e:
        print(f"Error saving text: {str(e)}")

def main():
    # Path to the image file
    image_path = "/Users/sonnguyen/Documents/llm-rag/loadpdf/baocaohpg.png"
    
    print("Starting OCR process...")
    print(f"Processing image: {image_path}")
    
    # Perform OCR
    extracted_text = perform_ocr_with_openai(image_path, detail_level="high")
    
    if extracted_text:
        print("\n" + "="*50)
        print("EXTRACTED TEXT:")
        print("="*50)
        print(extracted_text)
        print("="*50)
        
        # Save to file
        output_path = "extracted_text.txt"
        save_extracted_text(extracted_text, output_path)
        
    else:
        print("Failed to extract text from the image.")

if __name__ == "__main__":
    main()
