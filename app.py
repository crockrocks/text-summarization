import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration
from rouge_score import rouge_scorer

# Load BART model and tokenizer
model_name = "facebook/bart-large-cnn"
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

# Load ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

st.title("Text Summarization")
input_text = st.text_area("Enter an article to summarize:")

if st.button("Generate Summary"):
    if input_text.strip() != "":
        # Generate summary using the model
        inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Display generated summary
        st.subheader("Generated Summary:")
        st.write(generated_summary)
