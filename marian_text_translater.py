from transformers import MarianMTModel, MarianTokenizer

# Function to perform translation
def translate(text, model, tokenizer):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    
    # Generate translation
    outputs = model.generate(**inputs)
    
    # Decode the translated text
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

def main():
    # Choose the model and tokenizer
    model_name = "Helsinki-NLP/opus-mt-tr-en"  # Turkish to English model
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    
    # Take user input for translation
    print("Enter text to translate or type 'exit' to quit: ")
    while True:
        text_to_translate = input("› ")
        if text_to_translate.lower() == "exit":
            print("Exiting...")
            break
        
        # Perform translation
        translated_text = translate(text_to_translate, model, tokenizer)
        print("<", translated_text)

if __name__ == "__main__":
    main()
