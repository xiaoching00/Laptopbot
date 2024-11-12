import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import base64
import logging

# Load environment variables from .env file
load_dotenv()
logging.basicConfig(level=logging.INFO)  # Set logging level to INFO
NUMBER_OF_MESSAGES_TO_DISPLAY = 20  # Number of chat messages to display

# Function to load system instructions from a text file
def load_system_instructions():
    with open('laptop_db.txt', 'r') as file:
        return file.read()  # Return the contents of the file as a string

# Function to parse laptop data from the text file
def load_laptop_data():
    laptops = []
    try:
        with open("laptop_db.txt", "r") as file:
            for line in file:
                # Split line into key-value pairs (e.g., LaptopName: ASUS TUF Gaming F15)
                attributes = line.strip().split(", ")
                laptop = {}
                for attribute in attributes:
                    key, value = attribute.split(": ")
                    laptop[key.strip()] = value.strip()
                laptops.append(laptop)
    except Exception as e:
        logging.error(f"Error loading laptop data: {str(e)}")
    return laptops

# Function to set up the generative model with specific configurations
def setup_model(system_instruction):
    # Set model configuration parameters
    generation_config = {
        "temperature": 1,  # Controls randomness of the response
        "top_p": 0.95,  # Top-p sampling for diversity
        "top_k": 40,  # Limits the number of tokens considered during sampling
        "max_output_tokens": 8192,  # Max tokens for the response
        "response_mime_type": "text/plain",  # Response format
    }

    # Get the API key from the environment variables
    genai.api_key = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

    # Create and return the generative model instance
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",  # Set model name
        generation_config=generation_config,  # Pass the configuration
        system_instruction=system_instruction,  # Provide the system instruction
    )
    return model

# Function to start the chat with the model and send a user message
def start_chat(model, user_message):
    chat_session = model.start_chat(
        history=[  # Initialize the chat history with the user's message
            {
                "role": "user",
                "parts": [user_message],
            },
        ]
    )
    # Send the user's message to the model and get the response
    return chat_session.send_message(user_message).text

# Function to convert an image file to a base64 string (useful for embedding in the app)
def img_to_base64(image_path):
    """Convert image to base64."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()  # Return base64 encoded string
    except Exception as e:
        logging.error(f"Error converting image to base64: {str(e)}")  # Log any error that occurs
        return None

# Function to initialize session state variables (persistent storage during the session)
def initialize_session_state():
    """Initialize session state variables."""
    if "history" not in st.session_state:
        st.session_state.history = []  # Store user-assistant conversation history
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []  # Store past conversations

# Main function to run the Streamlit app
def main():
    initialize_session_state()  # Ensure session state variables are initialized
    st.set_page_config(
        page_title="Laptopbot - AI Assistant",  # Set the page title
        page_icon="streamly_logo.png",  # Set the favicon
        layout="wide",  # Page layout style
        initial_sidebar_state="auto",  # Sidebar state
        menu_items={  # Optional menu items for the Streamlit app
        }
    )

    # Display the main title of the app
    st.title("Laptopbot 💻")
    system_instruction = load_system_instructions()  # Load system instructions

    # Custom CSS for styling the app
    st.markdown(
        """
        <style>
        .cover-glow {
            width: 100%;
            height: auto;
            padding: 3px;
            position: relative;
            z-index: -1;
            border-radius: 30px;
        }
        </style>
        """,
        unsafe_allow_html=True,  # Allow custom HTML and CSS in Streamlit
    )

    # Convert logo image to base64 for embedding in the sidebar
    img_path = "image\streamly_logo.png"
    img_base64 = img_to_base64(img_path)
    if img_base64:
        # Embed the image in the sidebar
        st.sidebar.markdown(
            f'<img src="data:image/png;base64,{img_base64}" class="cover-glow">',
            unsafe_allow_html=True,
        )

    # Sidebar content with options to switch between modes (Info or Chat with Laptopbot)
    st.sidebar.markdown("---")
    mode = st.sidebar.radio("Select Mode:", options=["Info", "Chat with Laptopbot"], index=0)

    # Basic interaction options in the sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Basic Interactions")
    st.sidebar.markdown("- **Ask About Laptops**: Type your questions about laptops or requirements.")
    st.sidebar.markdown("- **Requirements**: Ask for recommendations based on specific requirements.")
    st.sidebar.markdown("- **Helps**: If you have any problem, please contact support team - 012-345 6789")
    st.sidebar.markdown("---")

    # Initialize the model using the system instructions
    model = setup_model(system_instruction)


    # Chat Mode
    if mode == "Chat with Laptopbot":
        st.subheader("Welcome to chat with our AI Agent! 🤖")
        user_input = st.text_input("Ask about laptops or requirements:", placeholder="e.g. suggest a laptop below 2000 and is used for video editing")

        if user_input:  # If the user enters a message
            # Get the response from the AI model
            response = start_chat(model, user_input)
            # Save both user and assistant messages to session state history
            st.session_state.history.append({"role": "user", "content": user_input})
            st.session_state.history.append({"role": "assistant", "content": response})
            # Display the AI's response
            st.markdown(f"{response}")
        
        # Use markdown to create a separator line
        st.write("---") 

        st.subheader("Chat History 💭")
        

        # Display the conversation history
        for message in st.session_state.history[-NUMBER_OF_MESSAGES_TO_DISPLAY:]:
            role = message["role"]
            avatar_image = "image/chat_lego.png" if role == "assistant" else "image/user_logo.png"
            # Display the message with appropriate avatar image
            with st.chat_message(role, avatar=avatar_image):
                st.write(message["content"])

    # Info Mode
    else:
        st.subheader("Laptopbot Information")
        st.markdown("""
            Laptopbot can help you choose the best laptop based on your needs. You can ask about:
            - Specific laptop models
            - Performance, specifications, and other requirements
            - Price comparison, and much more!
        """)

        # Optionally add more details or examples for interactions
        st.markdown("### Example Queries:")
        st.markdown("1. **Which laptop is best for gaming?**")
        st.markdown("2. **I need a laptop with 16GB RAM and Intel i7 processor.**")
        st.markdown("3. **Find laptops under $1000 with 8GB RAM.**")

# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()
