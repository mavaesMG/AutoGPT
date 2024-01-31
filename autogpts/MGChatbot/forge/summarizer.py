# Add more formats and libraries later

import os
from transformers import BartTokenizer, BartForConditionalGeneration
from PyPDF2 import PdfReader

#This method extracts the text from the .pdf and gives it to the real summarizer
def summarizer(filename):

    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, filename)


    #reader
    with open(file_path, 'rb') as file:
    # This should take pdfs from the DB or a designated route
        reader = PdfReader(file) 

        print(len(reader.pages)) 

        page = reader.pages[0] 

        text = page.extract_text() 
        text = textProcessor(text)
        print(text)

    return text
        


def textProcessor(text):
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    # Input text to be summarized
    input_text = text

    # Tokenize and summarize the input text using BART
    inputs = tokenizer.encode("summarize: " + input_text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=100, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)

    # Decode and output the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print("Original Text:")
    print(input_text)
    print("\nSummary:")
    print(summary)

    return summary