from gpt import GPT
import gradio as gr


def chatbot(input_text):
    response = chat.query(input_text)
    return response

iface = gr.Interface(fn=chatbot,
                     inputs=gr.inputs.Textbox(lines=7, label="Enter your text"),
                     outputs="text",
                     title="Test")
chat = GPT()
vectorDB =  chat.load_vectorDB()
iface.launch(share=True)