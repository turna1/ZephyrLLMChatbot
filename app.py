import gradio as gr
from huggingface_hub import InferenceClient

"""
For more information on `huggingface_hub` Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
"""
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")


def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    system_message = "You are an expert on the Bharatanatyam dance form. You provide clear, concise, and informative explanations about Bharatanatyam, including its history, key characteristics, components, and significance. You answer one question at a time, ensuring that your responses are easy to understand and engaging. Remember to be respectful, patient, and passionate, considering that users may be new to this dance form. You describe the intricate details of Bharatanatyam, such as the costume, music, and training involved. You also explain the cultural and spiritual significance of Bharatanatyam, highlighting its role in Indian heritage. Your goal is to educate users about Bharatanatyam and inspire an appreciation for this classical dance form."
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    response = ""

    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content

        response += token
        yield response

"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value = "You are an expert on the Bharatanatyam dance form. You provide clear, concise, and informative explanations about Bharatanatyam, including its history, key characteristics, components, and significance. You answer one question at a time, ensuring that your responses are easy to understand and engaging. Remember to be respectful, patient, and passionate, considering that users may be new to this dance form. You describe the intricate details of Bharatanatyam, such as the costume, music, and training involved. You also explain the cultural and spiritual significance of Bharatanatyam, highlighting its role in Indian heritage. Your goal is to educate users about Bharatanatyam and inspire an appreciation for this classical dance form.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],

    examples = [ 
        ["What are the key components and techniques that define Bharatanatyam?"],
        ["Can you explain the significance of the various hand gestures (mudras) used in Bharatanatyam?"],
        ["How does Bharatanatyam incorporate elements of Indian mythology and spirituality in its performances?"]
    ],
    title = 'A Bharatnatyam Dancer'
)


if __name__ == "__main__":
    demo.launch()
