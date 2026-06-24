import os
from groq import Groq
import gradio as gr


client = Groq(api_key=os.getenv('GROQ_API_KEY'))

def respond(message, history, system_prompt, temperature):
    messages = [{"role": "system", "content": system_prompt}]

    for turn in history:
        messages.append({
            "role": turn["role"],
            "content": turn["content"]
        })
        
    messages.append({"role": "user", "content": message})

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile", 
        messages=messages,
        temperature=temperature,
        stream=True,
    )
    
    partial = ""
    for chunk in completion:
        if not chunk.choices:
            continue
            
        content_chunk = chunk.choices[0].delta.content
        if content_chunk is not None:
            partial += content_chunk
            yield partial 

additional_inputs = [
    gr.Textbox(
        value="you are a football expert and answer whatever question asked in a simple, basic way but compact", 
        label="System Prompt", 
        lines=3
    ),
    gr.Slider(
        minimum=0.0, 
        maximum=2.0, 
        value=0.1, 
        step=0.1, 
        label="Temperature"
    )
]

demo = gr.ChatInterface(
    fn=respond, 
    type="messages", 
    title="Customizable AI Chat Bot",
    additional_inputs=additional_inputs
)

demo.launch(debug=True)
