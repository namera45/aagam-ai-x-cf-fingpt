import os
import streamlit as st
from huggingface_hub import InferenceClient
from gtts import gTTS
import speech_recognition as sr
import tempfile
import yfinance as yf  # Import yfinance

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Function to generate AI response
def generate_ai_response(prompt, client):
    system_message = "You are a finance expert AI assistant. Your purpose is to provide concise and accurate responses to user queries related to finance only. You should not engage in conversations or provide information on topics outside of finance. Your responses should be brief, clear, and to the point. If a user asks a question that is not related to finance, respond with 'Not related to finance.' If you are unsure or do not know the answer to a question, respond with 'I'm not sure.' Your goal is to provide helpful and informative responses to users' finance-related questions."
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]

    response = ""
    for message in client.chat_completion(
        messages=messages,
        max_tokens=120,
        stream=True
    ):
        response += message.choices[0].delta.content or ""
     

    # Check if user requested financial data using yfinance
    if "stock performance of" in prompt.lower():
        try:
            # Extract ticker symbol from user input (assuming it's in a specific format)
            # Example: "Show me the stock performance of HDFC of last year"
            words = prompt.lower().split()
            index_of_symbol = words.index("of") + 1
            ticker_symbol = words[index_of_symbol].upper()  # Extract ticker symbol
            
            # Fetch data for the past year
            ticker = yf.Ticker(ticker_symbol)
            data = ticker.history(period="1y")
            
            # Format the data into a response
            response += f"\n\nHere is the stock performance for {ticker_symbol} over the past year:\n"
            
            # Iterate through rows of the DataFrame and format each row
            for index, row in data.iterrows():
                response += f"Date: {index}\n"
                response += f"Open: {row['Open']}, High: {row['High']}, Low: {row['Low']}, Close: {row['Close']}, Adj Close: {row['Adj Close']}, Volume: {row['Volume']}\n\n"
        
        except Exception as e:
            response += f"\n\nCould not retrieve stock performance data for {ticker_symbol}. Error: {str(e)}"
    return response
    # Convert AI response to speech and append to messages
    text_to_speech(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    

# Function to clear chat history
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Function to handle file upload
def handle_file_upload(f):
    with open(os.path.join("uploads", f.name), "wb") as file:
        file.write(f.getbuffer())
    st.success(f"Saved file: {f.name}")

# Function for text-to-speech conversion
def text_to_speech(text):
    with tempfile.NamedTemporaryFile(delete=False) as fp:
        tts = gTTS(text=text, lang='en')
        tts.save(fp.name)
        st.audio(fp.name, format='audio/mp3')

# Function for speech-to-text conversion
def speech_to_text():
    r = sr.Recognizer()
    st.info("Speak now...")
    with sr.Microphone() as source:
        audio_data = r.listen(source)
    
    try:
        text = r.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        st.warning("Could not understand audio")
    except sr.RequestError as e:
        st.error(f"Could not request results; {e}")

def main():
    st.title('AI Assistant')

    # Sidebar for settings and file upload
    st.sidebar.title('🤖 AI Assistant Settings')
    hf_api_token = os.getenv("your API token here ")

    client = InferenceClient(token="your API  token here")
    
    # File upload section in sidebar
    uploaded_file = st.sidebar.file_uploader("Upload Files", type=['png', 'jpg', 'jpeg', 'gif', 'csv'])
    if uploaded_file is not None:
        handle_file_upload(uploaded_file)

    # Display chat history
    for message in st.session_state.messages:
        with st.expander(f"{message['role']} says:"):
            st.write(message["content"])
            if message['role'] == 'assistant':
                text_to_speech(message["content"])

    # Speech button to input text
    if st.button("Speech"):
        user_input = speech_to_text()
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            response = generate_ai_response(user_input, client)

    # User input
    user_input = st.text_input("Type your message here:")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        response = generate_ai_response(user_input, client)

    # Clear chat history button
    if st.sidebar.button('Clear Chat History'):
        clear_chat_history()

if __name__ == "__main__":
    main()
