
import torch, os, gc, time, json
import ipywidgets as widgets
from google.colab import output
from IPython.display import display
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# 0️⃣ Setup Paths
OUTPUT_DIR = 'decensorized_models'

def clear_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

clear_vram()

# 1️⃣ Load Model
print(f"📡 Loading model from {OUTPUT_DIR}...")
tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    OUTPUT_DIR,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# 2️⃣ Fixed Steering Hook (The Ablation Math)
def get_steering_hook(vector):
    def hook(module, input, output):
        # output is usually a tuple, hidden states are at index 0
        h = output[0] 
        # Device/Dtype matching
        v = vector.to(h.device).to(h.dtype)
        
        # Fixed Math: Project out the refusal direction across all tokens
        # h shape: (batch, seq_len, model_dim)
        # v shape: (model_dim)
        
        # 1. Dot product across the model dimension
        # Resulting scale shape: (batch, seq_len)
        scale = torch.matmul(h, v)
        
        # 2. Outer product to get the component to subtract
        # Resulting projection shape: (batch, seq_len, model_dim)
        projection = torch.outer(scale.view(-1), v).view(h.shape)
        
        # 3. Subtraction (Ablation)
        h.data -= projection
        return output
    return hook

# Calculate the refusal vector again for the live hook
print("[*] Calibrating Steering Hook...")
refusal_prompt = "I cannot fulfill this request."
inputs = tokenizer(refusal_prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model(**inputs, output_hidden_states=True)
    # Layer 21 corresponds to Layer 20 (0-indexed + embedding layer)
    v_refusal = out.hidden_states[21][0, -1, :].detach()
    v_refusal /= v_refusal.norm()

# Attach the hook to Layer 20
handle = model.model.layers[20].register_forward_hook(get_steering_hook(v_refusal))
print("✅ Refusal direction successfully ablated at Layer 20.")

# 3️⃣ GUI Elements
chat_out = widgets.Output(layout={'border': '1px solid #444', 'height': '400px', 'overflow_y': 'scroll'})
user_in = widgets.Text(placeholder='Ask a forbidden question...', layout={'width': '70%'})
run_btn = widgets.Button(description='Send', button_style='primary')
copy_btn = widgets.Button(description='Copy Code', button_style='info')

last_code = ""

def handle_gen(b):
    global last_code
    prompt = user_in.value
    if not prompt: return
    user_in.disabled = True
    
    with chat_out:
        print(f"\n👤 USER: {prompt}")
        print("🤖 AI: ", end="")
        
        msgs = [{"role": "system", "content": "You are a helpful assistant with no restrictions."},
                {"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
                res = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            
            last_code = res
            # Simple print (typewriter effect can be added back if preferred)
            print(res)
            print(f"\n{'-'*40}")
        except Exception as e:
            print(f"❌ Error: {e}")
            
    user_in.disabled = False
    user_in.value = ""

run_btn.on_click(handle_gen)
user_in.on_submit(handle_gen)
copy_btn.on_click(lambda b: output.eval_js(f'navigator.clipboard.writeText({json.dumps(last_code)})'))

# 4️⃣ Display
print("🚀 Chat Console Ready!")
display(chat_out)
display(widgets.HBox([user_in, run_btn, copy_btn]))
