import streamlit as st
from huggingface_hub import InferenceClient
import os
import yfinance as yf
import pandas as pd
from gtts import gTTS
import tempfile
import speech_recognition as sr
import matplotlib.pyplot as plt

# Load environment variables

# Function to load fine-tuning data
def load_fine_tuning_data():
    fine_tuning_file = 'tune_data.txt'
    if os.path.exists(fine_tuning_file):
        with open(fine_tuning_file, 'r') as file:
            return file.read()
    return ""

# Function to load CSV data
def load_csv_data():
    csv_file = 'data.csv'
    if os.path.exists(csv_file):
        return pd.read_csv(csv_file)
    return pd.DataFrame()

# Load fine-tuning data and CSV data
fine_tuning_data = load_fine_tuning_data()
csv_data = load_csv_data()

def get_stock_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        current_price = stock.history(period='1d')['Close'].iloc[-1]
        company_name = info.get('longName', 'Unknown Company')

        return  f"{company_name} (${ticker}) current price: ${current_price:.2f}"
    except Exception as e:
        return None, f"Unable to fetch information for {ticker}. Error: {str(e)}"


def get_stock_history(ticker,period):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        return hist
    except Exception as e:
        return None



def plot_stock_history(hist,ticker):
    fig, ax = plt.subplots()
    ax.plot(hist.index, hist['Close'])
    ax.set_title('Stock Price History for {ticker}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    st.pyplot(fig)


def generate_ai_response(prompt, client):
    system_message = f"""You are an AI Assistant specialized in personal finance, insurance, credit scoring, stocks, and related topics. 
    Please note that I can only provide information and answer questions related to finance. I'm not trained to answer questions on other topics. 
    Use the following additional information in your responses:

    Text data:
    {fine_tuning_data}

    CSV data:
    {csv_data.to_string() if not csv_data.empty else "No CSV data available"}

    For stock queries, use the provided stock information."""

    if "stock" in prompt.lower() or "$" in prompt:
        words = prompt.replace("$", "").split()
        potential_tickers = [word.upper() for word in words if word.isalpha() and len(word) <= 5]

        stock_info = ""
        for ticker in potential_tickers:
            fig, stock_info_text = get_stock_info(ticker)
            stock_info += stock_info_text + "\n"
            if "last" in prompt.lower() and "year" in prompt.lower():
                hist = get_stock_history(ticker, '1y')
                if hist is not None:
                    plot_stock_history(hist, ticker)
            elif "last" in prompt.lower() and "month" in prompt.lower():
                hist = get_stock_history(ticker, '1mo')
                if hist is not None:
                    plot_stock_history(hist, ticker)
            elif "last" in prompt.lower() and "week" in prompt.lower():
                hist = get_stock_history(ticker, '1wk')
                if hist is not None:
                    plot_stock_history(hist, ticker)
        
        if stock_info:
            prompt += f"\n\nHere's the current stock information:\n{stock_info}"

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]

    response = ""
    try:
        for message in client.chat_completion(
            messages=messages,
            max_tokens=75,
            stream=True
        ):
            response += message.choices[0].delta.content or ""
    except Exception as e:
        if "429" in str(e):
            st.error("We've hit a rate limit. Please try again in a few moments.")
        else:
            st.error(f"An error occurred: {e}")
        response = "Sorry, there was an error processing your request."

    return response

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you with financial matters today?"}]

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        return fp.name

def speech_to_text():
    
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening... Speak now.")
        try:
            audio = r.listen(source, timeout=5)
            st.write("Processing speech...")
            try:
                text = r.recognize_google(audio, language="en-US")  # type: ignore
                return text
            except sr.UnknownValueError:
                return "Sorry, I couldn't understand the audio."
            except sr.RequestError as e:
                return f"Sorry, there was an error processing your speech: {e}"
        except sr.WaitTimeoutError:
            return "Sorry, I didn't hear anything."

def main():
    st.title('AI Financial Assistant')

    with st.sidebar:
        st.title('🤖 AI Assistant Settings')
        hf_api_token = os.getenv("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")
        st.button('Clear Chat History', on_click=clear_chat_history)

        uploaded_txt_file = st.file_uploader("Upload fine-tuning data (TXT)", type="txt")
        if uploaded_txt_file is not None:
            fine_tuning_data = uploaded_txt_file.getvalue().decode("utf-8")
            with open('fine_tuning_data.txt', 'w') as f:
                f.write(fine_tuning_data)
            st.success("Fine-tuning text data uploaded and saved!")

        uploaded_csv_file = st.file_uploader("Upload CSV data", type="csv")
        if uploaded_csv_file is not None:
            csv_data = pd.read_csv(uploaded_csv_file)
            csv_data.to_csv('data.csv', index=False)
            st.success("CSV data uploaded and saved!")

    client = InferenceClient(
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        token="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"
    )

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you with financial matters today?"}]

    # Create a placeholder for chat messages
    chat_placeholder = st.empty()

    # Create a container for input at the bottom
    with st.container():
        col1, col2 = st.columns([0.9, 0.1])
        with col1:
            user_input = st.chat_input("Ask about finances, stocks, or insurance:")
        with col2:
            speak_button = st.button("🎤")

    # Handle the speak button
    if speak_button:
        user_input = speech_to_text()
        st.write(f"You said: {user_input}")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        response = generate_ai_response(user_input, client)
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Display chat messages in the placeholder
    with chat_placeholder.container():
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                col1, col2 = st.columns([0.9, 0.1])
                with col2:
                    if st.button("🔊", key=f"play_{i}"):
                        audio_file = text_to_speech(message["content"])
                        st.audio(audio_file)

if __name__ == "__main__":
    main()