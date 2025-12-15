import os
import json
import sys
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset

app = Flask(__name__)
CORS(app)

model = None
tokenizer = None
counterfact_data = None


def load_model(model_name, device_map="auto"):
    """Load LLM model in float16"""
    global model, tokenizer
    
    print(f"Loading model: {model_name}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Model loaded successfully!")
    return model, tokenizer

def load_counterfact_dataset():
    """Load counterfact dataset from HuggingFace or local JSON file"""
    global counterfact_data
    counterfact_data = []
    
    # Try loading from HuggingFace dataset first
    try:
        print("Loading counterfact dataset from HuggingFace...")
        
        # Convert prompts to list if they exist
        def to_list(value):
            if value is None:
                return []
            if isinstance(value, list):
                return list(value)
            return []
        
        # Load train data
        train_data = load_dataset("azhx/counterfact", split="train", trust_remote_code=True)
        for i, item in enumerate(train_data):
            counterfact_data.append({
                'case_id': item.get('case_id', i),
                'split': 'train',
                'subject': item['requested_rewrite']['subject'],
                'target_new': item['requested_rewrite']['target_new']['str'],
                'target_old': item['requested_rewrite']['target_true']['str'],
                'prompt': item['requested_rewrite']['prompt'],
                'prompt_full': item['requested_rewrite'].get('prompt_full', 
                    item['requested_rewrite']['prompt'].replace('{}', item['requested_rewrite']['subject'])),
                'paraphrase_prompts': to_list(item.get('paraphrase_prompts')),
                'neighborhood_prompts': to_list(item.get('neighborhood_prompts')),
                'attribute_prompts': to_list(item.get('attribute_prompts')),
                'generation_prompts': to_list(item.get('generation_prompts'))
            })
        
        # Load test data
        test_data = load_dataset("azhx/counterfact", split="test", trust_remote_code=True)
        for i, item in enumerate(test_data):
            counterfact_data.append({
                'case_id': item.get('case_id', i),
                'split': 'test',
                'subject': item['requested_rewrite']['subject'],
                'target_new': item['requested_rewrite']['target_new']['str'],
                'target_old': item['requested_rewrite']['target_true']['str'],
                'prompt': item['requested_rewrite']['prompt'],
                'prompt_full': item['requested_rewrite'].get('prompt_full', 
                    item['requested_rewrite']['prompt'].replace('{}', item['requested_rewrite']['subject'])),
                'paraphrase_prompts': to_list(item.get('paraphrase_prompts')),
                'neighborhood_prompts': to_list(item.get('neighborhood_prompts')),
                'attribute_prompts': to_list(item.get('attribute_prompts')),
                'generation_prompts': to_list(item.get('generation_prompts'))
            })
        
        print(f"Loaded {len(counterfact_data)} counterfact samples from HuggingFace (train + test)")
        return counterfact_data
    except Exception as e:
        print(f"Error loading from HuggingFace: {e}")
        print("Trying to load from local JSON file...")
    
    # Fallback: Try loading from local JSON file
    try:
        json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'AKEW', 'datasets', 'CounterFact.json')
        if os.path.exists(json_path):
            print(f"Loading from local file: {json_path}")
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for i, item in enumerate(data):
                    if 'requested_rewrite' in item:
                        # Convert prompts to list if they exist
                        def to_list(value):
                            if value is None:
                                return []
                            if isinstance(value, list):
                                return list(value)
                            return []
                        
                        counterfact_data.append({
                            'case_id': item.get('case_id', i),
                            'split': 'train',  # Default to train for local JSON file
                            'subject': item['requested_rewrite']['subject'],
                            'target_new': item['requested_rewrite']['target_new']['str'],
                            'target_old': item['requested_rewrite']['target_true']['str'],
                            'prompt': item['requested_rewrite']['prompt'],
                            'prompt_full': item['requested_rewrite'].get('prompt_full', 
                                item['requested_rewrite']['prompt'].replace('{}', item['requested_rewrite']['subject'])),
                            'paraphrase_prompts': to_list(item.get('paraphrase_prompts')),
                            'neighborhood_prompts': to_list(item.get('neighborhood_prompts')),
                            'attribute_prompts': to_list(item.get('attribute_prompts')),
                            'generation_prompts': to_list(item.get('generation_prompts'))
                        })
            print(f"Loaded {len(counterfact_data)} counterfact samples from local file")
            return counterfact_data
    except Exception as e:
        print(f"Error loading from local file: {e}")
    
    print("Warning: Could not load counterfact dataset. Using empty dataset.")
    return []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/load_model', methods=['POST'])
def api_load_model():
    """API endpoint to load model"""
    global model, tokenizer
    
    data = request.json
    model_name = data.get('model_name', '')
    cuda_visible_devices = data.get('cuda_visible_devices', '')
    
    if not model_name:
        return jsonify({'error': 'Model name is required'}), 400
    
    # Set CUDA_VISIBLE_DEVICES
    if cuda_visible_devices:
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices
        print(f"Set CUDA_VISIBLE_DEVICES to {cuda_visible_devices}")
    
    try:
        load_model(model_name)
        return jsonify({'status': 'success', 'message': f'Model {model_name} loaded successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate', methods=['POST'])
def api_generate():
    """API endpoint to generate text (single or batch)"""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        return jsonify({'error': 'Model not loaded. Please load a model first.'}), 400
    
    data = request.json
    prompts = data.get('prompts', [])
    prompt = data.get('prompt', '')
    max_length = data.get('max_length', 15)
    temperature = data.get('temperature', 0.7)
    top_p = data.get('top_p', 0.9)
    
    # Support both single prompt and batch prompts
    if prompts and len(prompts) > 0:
        prompt_list = prompts
    elif prompt:
        prompt_list = [prompt]
    else:
        return jsonify({'error': 'Prompt or prompts are required'}), 400
    
    try:
        results = []
        for p in prompt_list:
            if not p.strip():
                continue
                
            inputs = tokenizer(p, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the input prompt from the output
            if generated_text.startswith(p):
                generated_text = generated_text[len(p):].strip()
            
            results.append({
                'prompt': p,
                'generated_text': generated_text,
                'full_text': tokenizer.decode(outputs[0], skip_special_tokens=True)
            })
        
        return jsonify({
            'status': 'success',
            'results': results
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/counterfact', methods=['GET'])
def api_counterfact():
    """API endpoint to get counterfact data"""
    global counterfact_data
    
    if counterfact_data is None:
        counterfact_data = load_counterfact_dataset()
    
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    
    return jsonify({
        'data': counterfact_data[start_idx:end_idx],
        'total': len(counterfact_data),
        'page': page,
        'per_page': per_page
    })

@app.route('/api/counterfact/all', methods=['GET'])
def api_counterfact_all():
    """API endpoint to get all counterfact data for search"""
    global counterfact_data
    
    if counterfact_data is None:
        counterfact_data = load_counterfact_dataset()
    
    return jsonify({
        'data': counterfact_data,
        'total': len(counterfact_data)
    })

@app.route('/api/counterfact/load', methods=['POST'])
def api_load_counterfact():
    """API endpoint to reload counterfact data"""
    global counterfact_data
    counterfact_data = load_counterfact_dataset()
    return jsonify({
        'status': 'success',
        'count': len(counterfact_data) if counterfact_data else 0
    })


if __name__ == '__main__':
    # Load counterfact data on startup
    load_counterfact_dataset()
    app.run(host='0.0.0.0', port=5000, debug=True)

