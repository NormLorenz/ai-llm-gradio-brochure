import os
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
import google.generativeai
import gradio as gr
from bs4 import BeautifulSoup
import requests
from typing import List


# Load environment variables in a file called .env
load_dotenv(override=True)

openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')


# Connect to OpenAI, Anthropic and Google
openai = OpenAI()
claude = anthropic.Anthropic()
google.generativeai.configure()


# A generic system message
system_message = "You are an assistant that analyzes the contents of a company website landing page \
and creates a short brochure about the company for prospective customers, investors and recruits. Respond in markdown."


class Website:
    url: str
    title: str
    text: str

    def __init__(self, url):
        self.url = url
        response = requests.get(url)
        self.body = response.content
        soup = BeautifulSoup(self.body, 'html.parser')
        self.title = soup.title.string if soup.title else "No title found"
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        self.text = soup.body.get_text(separator="\n", strip=True)

    def get_contents(self):
        return f"Webpage Title:\n{self.title}\nWebpage Contents:\n{self.text}\n\n"


def main() -> None:
    ### High level entry point ###
    view = gr.Interface(
        fn=select_model,
        inputs=[
            gr.Textbox(label="Company name:"),
            gr.Textbox(label="Landing page URL including http:// or https://"),
            gr.Dropdown(["gpt-4o-mini",
                         "claude-3-5-haiku-latest",
                         "gemini-1.5-flash",
                         "gemini-2.5-flash-lite"], label="Select model", value="gpt-4o-mini")],
        outputs=[
            gr.Markdown(label="Brochure:")],
        flagging_mode="never"
    )
    view.launch(inbrowser=True)


# def stream_brochure(company_name, url, model):
#     yield ""
#     prompt = f"Please generate a company brochure for {company_name}. Here is their landing page:\n"
#     prompt += Website(url).get_contents()
#     if model == "GPT":
#         result = stream_gpt(prompt)
#     elif model == "Claude":
#         result = stream_claude(prompt)
#     else:
#         raise ValueError("Unknown model")
#     yield from result


def select_model(company_name: str, url: str, model: str):
    # yield
    prompt = f"Please generate a company brochure for {company_name}. Here is their landing page:\n"
    prompt += Website(url).get_contents()
    if model == "gpt-4o-mini":
        result = call_gpt(prompt, model)
    elif model == "claude-3-5-haiku-latest":
        result = call_claude(prompt, model)
    elif model == "gemini-1.5-flash":
        result = call_gemini(prompt, model)
    elif model == "gemini-2.5-flash-lite":
        result = call_gemini(prompt, model)
    else:
        raise ValueError("Unknown model")
    # yield from result


def call_gpt(prompt: str, model: str):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content


def call_claude(prompt: str, model: str):
    messages = [{"role": "user", "content": prompt}]
    response = claude.messages.create(
        model=model,
        system=system_message,
        messages=messages,
        max_tokens=500
    )
    return response.content[0].text


def call_gemini(prompt: str, model: str):
    gemini = google.generativeai.GenerativeModel(
        model_name=model,
        system_instruction=system_message
    )
    response = gemini.generate_content(prompt)
    return response.text


if __name__ == "__main__":
    main()
