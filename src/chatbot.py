from gpt import GPT
import gradio as gr

chat = GPT()
vectorDB =  chat.load_vectorDB() 

def chatbot(input_text):
    response = chat.query(input_text)
    return response

def chatbot_interface():
    return gr.Interface(
        fn=chatbot,
        inputs=gr.inputs.Textbox(lines=7, label="Enter your text"),
        outputs="text",
        title="CXLab Chatbot"
    )

chatbot_interface().launch(share=True)

