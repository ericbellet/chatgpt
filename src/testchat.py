import gradio as gr
import json
import time
from gpt import GPT


class ChatBot():

    def __init__(self):
      self.chat = GPT()
      self.chat.load_vectorDB()
      self.theme_conf()
      with open('configusers.json') as f:
        self.users = json.load(f)
        
    def query(self, input_text):
        response = self.chat.query(input_text)
        return response
      
    def theme_conf(self):
        self.theme = gr.themes.Base(
            primary_hue=gr.themes.Color(c100="#bfdbfe", c200="#bfdbfe", c300="#93c5fd", c400="#60a5fa", c50="#327474", c500="#3b82f6", c600="#2563eb", c700="#1d4ed8", c800="#1e40af", c900="#1e3a8a", c950="#1d3660"),
        )

    def login_validation(self, username, password):
        if username in self.users and self.users[username] == password:
            return True
        else:
            return False

    def launch(self):

        with gr.Blocks(theme=self.theme) as demo:
            chatbot = gr.Chatbot(label='The Chat')
            msg = gr.Textbox(show_label=False)
            clear = gr.Button("Clear chat", variant="secondary")
            send = gr.Button("Send message", variant="primary")

            def user(user_message, history):
                return "", history + [[user_message, None]]

            def bot(history):
                bot_message = self.query(history[-1][0])
                history[-1][1] = bot_message
                time.sleep(1)
                return history

            msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                bot, chatbot, chatbot
            )
            clear.click(lambda: None, None, chatbot, queue=False)
            send.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                bot, chatbot, chatbot
            )
        demo.launch(auth=self.login_validation)

if __name__ == "__main__":
    ChatBot().launch()
