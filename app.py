import streamlit as st
import librosa
from transformers import pipeline

st.set_page_config(page_title="AI Sloka Tutor", layout="centered")

st.title("🧘‍♀️ AI Sloka Tutor (Voice to Meaning)")
st.markdown("Upload a Sanskrit sloka audio file (.wav) to get a simplified English explanation.")

# Load models
@st.cache_resource
def load_models():
    stt_model = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")
    text_model = pipeline("text-generation", model="sshleifer/tiny-gpt2")
    return stt_model, text_model

stt, generator = load_models()

# File uploader
audio_file = st.file_uploader("🎙️ Upload your sloka audio (.wav only)", type=["wav"])

if audio_file:
    st.audio(audio_file, format="audio/wav")
    with st.spinner("Transcribing and analyzing..."):
        try:
            y, sr = librosa.load(audio_file, sr=16000)
            result = stt(y)
            sloka_text = result.get("text", "").strip()

            if not sloka_text:
                st.warning("No recognizable speech found.")
            else:
                st.subheader("📜 Sloka Transcript")
                st.success(sloka_text)

                prompt = f"Explain this Sanskrit sloka in simple English: {sloka_text}"
                output = generator(prompt, max_length=60, do_sample=True)[0]["generated_text"]
                meaning = output.replace(prompt, "").strip()

                st.subheader("🧠 AI-Generated Meaning")
                st.info(meaning or "No meaning generated.")

        except Exception as e:
            st.error(f"❌ Error: {e}")
