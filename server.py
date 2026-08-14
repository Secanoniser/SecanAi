import os
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

app = FastAPI(title="Local LLM Architect UI")

# Use SmolLM2-135M-Instruct directly for pristine, flawless conversational fluency
MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
print(f"[*] Loading pre-trained instruction model: {MODEL_ID}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    device_map="auto"
)
model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=128,
    temperature=0.7,
    do_sample=True,
    repetition_penalty=1.1,
    top_k=50,
    top_p=0.9
)

class ChatRequest(BaseModel):
    prompt: str

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Use SmolLM2 official chat template format
        messages = [{"role": "user", "content": request.prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        outputs = pipe(formatted_prompt)
        full_text = outputs[0]["generated_text"]
        response = full_text[len(formatted_prompt):].strip()
        
        clean_response = response
        if not clean_response:
            clean_response = "I understand! How else can I assist you?"
            
        return {"response": clean_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_index():
    return FileResponse("index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
