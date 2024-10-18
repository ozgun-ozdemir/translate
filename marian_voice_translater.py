from transformers import MarianMTModel, MarianTokenizer
import speech_recognition as sr

# Function to perform translation
def translate(text, model, tokenizer):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    
    # Generate translation
    outputs = model.generate(**inputs)
    
    # Decode the translated text
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return translated_text

# Function to recognize speech
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print("You said:", text)
            return text
        except sr.UnknownValueError:
            print("Sorry, I did not understand that.")
            return None
        except sr.RequestError:
            print("Sorry, there was an issue.")
            return None
        
def main():
    # Choose the model and tokenizer
    model_name = "Helsinki-NLP/opus-mt-tr-en"  # Turkish to English model
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    
    print("Enter 'voice' to use speech or 'text' to use text or type 'exit' to quit: ")
    
    while True:
        mode = input("> ").strip().lower()
        
        if mode == 'exit':
            print("Exiting...")
            break
        
        elif mode == 'voice':
            text_to_translate = recognize_speech()
            if text_to_translate:
                # Perform translation
                translated_text = translate(text_to_translate, model, tokenizer)
                print("Translated text:", translated_text)
        
        elif mode == 'text':
            print("Enter text to translate:")
            text_to_translate = input("› ")
            # Perform translation
            translated_text = translate(text_to_translate, model, tokenizer)
            print("Translated text:", translated_text)
        
        else:
            print("Please enter 'voice' or 'text'.")

if __name__ == "__main__":
    main()
