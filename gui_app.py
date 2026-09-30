import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. โหลด Tokenizer และ Base Model
base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
adapter_dir = "."  # โฟลเดอร์ปัจจุบัน

print("กำลังโหลด Tokenizer และ Base Model...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    dtype=torch.float32,
    device_map="cpu"
)

# 2. ประกอบ LoRA Adapter
print("กำลังประกบ LoRA Adapter จากโฟลเดอร์ปัจจุบัน...")
model = PeftModel.from_pretrained(model, adapter_dir)
model.eval()
print("✅ โหลดโมเดลเรียบร้อยพร้อมใช้งาน!")

# ฟังก์ชันแปลงค่าทุกอย่างให้เป็น String เสมอ
def clean_text(val):
    if isinstance(val, list):
        return " ".join([str(x) for x in val if x])
    return str(val) if val is not None else ""

# 3. ฟังก์ชันประมวลผลคำตอบ
def predict(message, history):
    messages = [
        {"role": "system", "content": "คุณคือผู้ช่วย AI ที่เชี่ยวชาญภาษาเหนือ (คำเมือง)"},
    ]
    
    # แปลงประวัติการคุยให้เป็น String อย่างถูกต้อง
    for item in history:
        if isinstance(item, (list, tuple)):
            user_msg = clean_text(item[0])
            bot_msg = clean_text(item[1])
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if bot_msg:
                messages.append({"role": "assistant", "content": bot_msg})
        elif isinstance(item, dict):
            role = item.get("role", "user")
            content = clean_text(item.get("content", ""))
            if content:
                messages.append({"role": role, "content": content})
            
    # เพิ่มข้อความล่าสุดของผู้ใช้
    messages.append({"role": "user", "content": clean_text(message)})
    
    # แปลงเข้า Chat Template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,        # ปิดการสุ่มคำตอบ เลือกเฉพาะคำที่มั่นใจที่สุด
            repetition_penalty=1.1, # ป้องกันการพิมพ์คำซ้ำ
            pad_token_id=tokenizer.eos_token_id
        )
        
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return response

# 4. สร้าง UI ด้วย ChatInterface
demo = gr.ChatInterface(
    fn=predict,
    title="💬 Chat AI ภาษาเหนือ (Khammueang Local Bot)",
    description="แชทบอทภาษาเหนือ พัฒนาด้วย Qwen2.5-1.5B + LoRA Fine-tuning"
)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), share=False)