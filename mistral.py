from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7B-GGML", gpu_layers=50)

print(llm("ye kia bakwas h.... give the sentiment of this sentence "))