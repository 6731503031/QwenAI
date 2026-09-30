import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# 1. เช็กระบบการประมวลผล
print("="*50)
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print(f"✅ ใช้งาน GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ ไม่พบ GPU! ระบบจะสลับไปรันผ่าน CPU (อาจจะตอบช้าลงนิดหน่อย)")
print("="*50)

base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"

# 2. แก้ปัญหา Path: ตรวจสอบโฟลเดอร์ปัจจุบัน
current_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(current_dir, "adapter_config.json")):
    # ถ้า app.py วางอยู่ในโฟลเดอร์ adapter โดยตรง
    adapter_path = current_dir
elif os.path.exists(os.path.join(current_dir, "qwen2.5-1.5b-khammueang-adapter", "adapter_config.json")):
    # ถ้า app.py วางอยู่นอกโฟลเดอร์ adapter
    adapter_path = os.path.join(current_dir, "qwen2.5-1.5b-khammueang-adapter")
else:
    adapter_path = current_dir

print(f"\nกำลังโหลด Base Model ({base_model_id})...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# ตั้งค่าการโหลดโมเดลตามฮาร์ดแวร์ที่มี
if device == "cuda":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
else:
    # สำหรับ CPU (ไม่ใช้ bitsandbytes quantization)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float32,
        device_map="cpu"
    )

print(f"กำลังประกบ LoRA Adapter จาก: {adapter_path}")
model = PeftModel.from_pretrained(base_model, adapter_path)
print("✅ โหลดโมเดลเรียบร้อยพร้อมใช้งาน!\n")

def chat(prompt):
    messages = [
        {"role": "system", "content": "คุณคือผู้ช่วย AI ที่เชี่ยวชาญภาษาเหนือ (คำเมือง)"},
        {"role": "user", "content": prompt}
    ]
    
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=128, 
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    print("=== Chat AI ภาษาเหนือ (Khammueang Local Bot) ===")
    print("พิมพ์คำถามเป็นภาษาไทยหรือภาษาเหนือได้เลย (พิมพ์ 'exit' เพื่อออกจากโปรแกรม)")
    print("-" * 50)
    
    while True:
        user_input = input("\nคุณ: ")
        if user_input.lower() in ["exit", "quit", "ออก"]:
            print("ปิดการทำงาน...")
            break
            
        reply = chat(user_input)
        print(f"AI: {reply}")