

import streamlit as st
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from deep_translator import GoogleTranslator
from gtts import gTTS
import langid
import os
from moviepy.editor import VideoFileClip

# Page config
st.set_page_config(page_title="VOICE-LINGUA", page_icon=":microphone:")

# Style
st.markdown("""
    <style>
    .reportview-container {
        background-color: #FFFF00;
    }
    h1, .stFileUploader, .stHeader {
        color: #0072C6;
    }
    [data-testid="stSidebar"] {
        min-width: 300px;
        max-width: 300px;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("logo.png")
    st.title("VOICE-LINGUA")
    if "option" not in st.session_state:
        st.session_state.option = "Speech Recognition"
    option = st.radio("Select an option:", (
        "Speech Recognition",
        "Translation",
        "Speech Generation",
        "Audio Extraction",
        "Summarization"
    ), key="option")

# Language mappings
lang_code_mapping = {
    "en": "eng_Latn", "hi": "hin_Deva", "fr": "fra_Latn", "de": "deu_Latn", "es": "spa_Latn",
    "it": "ita_Latn", "pt": "por_Latn", "ru": "rus_Cyrl", "ja": "jpn_Jpan", "ko": "kor_Hang",
    "zh": "chi_Hans", "ar": "ara_Arab", "tr": "tur_Latn", "nl": "nld_Latn", "pl": "pol_Latn",
    "uk": "ukr_Cyrl", "vi": "vie_Latn", "th": "tha_Thai", "id": "ind_Latn", "ms": "mal_Mlym",
    "ta": "tam_Taml", "te": "tel_Telu", "mr": "mar_Deva", "bn": "ben_Beng", "gu": "guj_Gujr",
    "kn": "kan_Knda", "pa": "pan_Guru", "ur": "urd_Arab", "si": "sin_Sinh", "mt": "mlt_Latn",
    "fi": "fin_Latn", "sv": "swe_Latn", "da": "dan_Latn", "no": "nor_Latn", "hu": "hun_Latn",
    "he": "heb_Hebr", "el": "ell_Grek", "ro": "rom_Latn", "bg": "bul_Cyrl", "sr": "srp_Cyrl",
    "cs": "ces_Latn", "sk": "slk_Latn", "hr": "hrv_Latn", "fa": "pes_Arab", "lt": "lit_Latn",
    "lv": "lav_Latn", "et": "est_Latn", "sw": "swa_Latn", "sl": "slv_Latn"
}

lang_code_mapping2 = {k: k for k in lang_code_mapping}

# Cached resources
@st.cache_resource
def load_whisper_model():
    model_id = "openai/whisper-large-v3"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=dtype, use_safetensors=True).to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor, device, dtype

@st.cache_resource
def load_nllb_model():
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
    return tokenizer, model

@st.cache_resource
def load_summarizer():
    return pipeline('summarization', model='sshleifer/distilbart-cnn-12-6')

# Utility functions
def detect_language_nllb(text):
    lang_code, _ = langid.classify(text)
    return lang_code_mapping.get(lang_code, "eng_Latn")

def translate_and_generate_audio(text, target_lang, filename):
    translated_text = GoogleTranslator(source='auto', target=target_lang).translate(text)
    gTTS(text=translated_text, lang=target_lang).save(filename)

# Speech Recognition
if option == "Speech Recognition":
    st.title("Speech Recognition")
    st.subheader("Upload an audio file to transcribe:")

    uploaded_file = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a", "mpeg"], key="speech_to_text")
    if uploaded_file:
        audio_path = f"temp_{uploaded_file.name}"
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.audio(audio_path, format="audio/wav")

        model, processor, device, dtype = load_whisper_model()
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=15,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=dtype,
            device=device,
        )
        result = pipe(audio_path)
        transcript = result["text"]
        lang_code = detect_language_nllb(transcript)

        st.header("Transcription:")
        st.write(transcript)
        st.subheader("Detected Language Code:")
        st.write(lang_code)

# Translation
elif option == "Translation":
    st.title("Translation")
    st.subheader("Translate text from one language to another:")

    input_text = st.text_area("Enter text to translate:", key="translation_input")
    src_lang = st.selectbox("Select source language:", list(lang_code_mapping.keys()), key="translation_src_lang")
    target_lang = st.selectbox("Select target language:", list(lang_code_mapping.keys()), key="translation_target_lang")

    if st.button("Translate"):
        src_code = lang_code_mapping[src_lang]
        tgt_code = lang_code_mapping[target_lang]
        tokenizer, model = load_nllb_model()
        translator = pipeline('translation', model=model, tokenizer=tokenizer, src_lang=src_code, tgt_lang=tgt_code)
        translated = translator(input_text)[0]['translation_text']

        st.header("Translated Text:")
        st.write(translated)

# Speech Generation
elif option == "Speech Generation":
    st.title("Speech Generation")
    st.subheader("Translate text and generate audio in the target language:")

    input_text = st.text_area("Enter text to convert:", key="speak_input")
    target_lang = st.selectbox("Select target language:", list(lang_code_mapping2.keys()), key="speak_target_lang")

    if st.button("Generate Speech"):
        filename = "generated_output.mp3"
        translate_and_generate_audio(input_text, lang_code_mapping2[target_lang], filename)
        st.success("Audio file generated successfully!")
        st.audio(filename, format="audio/mp3")

# Audio Extraction
elif option == "Audio Extraction":
    st.title("Audio Extraction")
    st.subheader("Extract audio from a video file:")

    video_path = st.text_input("Enter path to video file:")
    if st.button("Extract Audio"):
        if not os.path.isfile(video_path):
            st.error("File path is invalid.")
        else:
            video = VideoFileClip(video_path)
            audio = video.audio
            audio_path = "extracted_audio.mp3"
            audio.write_audiofile(audio_path)
            st.success("Audio extracted successfully!")
            with open(audio_path, "rb") as file:
                st.download_button("Download Audio", file.read(), file_name=audio_path, mime="audio/mpeg")

# Summarization
elif option == "Summarization":
    st.title("Text Summarizer")

    input_text = st.text_area("Enter the text to summarize:", height=200)
    word_count = len(input_text.split())

    if word_count > 0:
        st.subheader("Word Count:")
        st.write(word_count)

        min_len = max(10, word_count // 3)
        max_len = word_count

        min_summary = st.slider("Minimum summary length", min_value=min_len, max_value=word_count // 2, value=min_len, step=5)
        max_summary = st.slider("Maximum summary length", min_value=word_count // 2, max_value=max_len, value=max_len, step=10)

        if st.button("Summarize"):
            summarizer = load_summarizer()
            summary = summarizer(input_text, max_length=max_summary, min_length=min_summary, do_sample=False)[0]["summary_text"]
            st.subheader("Summary:")
            st.write(summary)
