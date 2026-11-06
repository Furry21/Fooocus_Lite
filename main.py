import asyncio
import io
import json
import os
import glob
import time
import gc
import torch

# 1. Сначала удаляем ссылки на сам пайплайн
global pipeline
pipeline = None

# 2. Вызываем сборщик мусора Python, чтобы он стер остатки объектов из ОЗУ
gc.collect()

# 3. Полностью зачищаем кэш видеокарты
torch.cuda.empty_cache()

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
# Импортируем нужные классы для ControlNet
from diffusers import (
    StableDiffusionXLControlNetPipeline, 
    ControlNetModel, 
    EulerAncestralDiscreteScheduler
)
from controlnet_aux import OpenposeDetector
import torch
from PIL import Image

# Инициализируем наше алхимическое ядро
app = FastAPI(title="Fooocus-Lite // Avery Edition // ControlNet Pose")

# Размечаем пути к папкам проекта (теперь они строятся динамически от корня проекта)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# === ТОЧНЫЕ ПУТИ ПОД ТВОЮ СТРУКТУРУ (ИСПРАВЛЕНО) ===
# Шагаем внутрь папки AI_Baza, а затем в models
MODELS_DIR = os.path.join(BASE_DIR, "AI_Baza", "models", "checkpoints")
# Путь к Лоре внутри AI_Baza/models/loras/
LORA_PATH = os.path.join(BASE_DIR, "AI_Baza", "models", "loras", "sd_xl_offset_example-lora_1.0.safetensors")
# ===================================================

# Проверяем видеокарту
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32

# Глобальные переменные для удержания моделей в памяти GPU
pipe = None
pose_detector = None
current_model_path = ""

def get_available_models():
    """Сканируем папку и собираем все доступные чекпоинты"""
    search_path = os.path.join(MODELS_DIR, "*.safetensors")
    files = glob.glob(search_path)
    return [os.path.basename(f) for f in files]

def load_pipeline(model_name: str):
    global pipe, current_model_path, pose_detector
    
    # 1. Ленивая загрузка детектора позы (OpenPose)
    if pose_detector is None:
        print("[Core] Инициализация процессора OpenPose...")
        pose_detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators")

    if not model_name:
        available = get_available_models()
        model_name = available[0] if available else ""
    
    if not model_name:
        print("[Core] КРИТИЧЕСКАЯ ОШИБКА: Чекпоинты не найдены в папке!")
        return None
        
    target_path = os.path.join(MODELS_DIR, model_name)
    
    # Если модель еще не загружена или изменен чекпоинт
    if pipe is None or current_model_path != target_path:
        print(f"[Core] Загружаем ControlNet и модель: {target_path} на {device}...")
        if pipe is not None:
            del pipe
            pipe = None
        torch.cuda.empty_cache()
        
        # Загружаем официальный ControlNet OpenPose для SDXL
        controlnet = ControlNetModel.from_pretrained(
            "thibaud/controlnet-openpose-sdxl-1.0", 
            torch_dtype=torch_dtype
        )
        
        # Поднимаем Single File пайплайн с поддержкой ControlNet
        pipe = StableDiffusionXLControlNetPipeline.from_single_file(
            target_path,
            controlnet=controlnet,
            torch_dtype=torch_dtype,
            use_safetensors=False
        )
        
        # Наш хорни-костыль: отключаем фильтр цензуры, если он есть
        if hasattr(pipe, "safety_checker") and pipe.safety_checker is not None:
            pipe.safety_checker = None
            
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
        
        # Оптимизация памяти VRAM
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
        current_model_path = target_path
        
        # Загрузка LoRA (используем имя адаптера, которое прописано ниже при генерации)
        if os.path.exists(LORA_PATH):
            print(f"[Core] Вживляем LoRA адаптер: {LORA_PATH}...")
            pipe.load_lora_weights(LORA_PATH, adapter_name="sd_x1")
            pipe.set_adapters(["sd_x1"], adapter_weights=[1.0])
        else:
            print(f"[Core] ПРЕДУПРЕЖДЕНИЕ: Файл LoRA не найден по пути {LORA_PATH}")
            
        print("[Core] Пайплайн ControlNet успешно собран!")
    return pipe

async def sse_iterator(task, queue, timestamp):
    while not task.done() or not queue.empty():
        try:
            data = await asyncio.wait_for(queue.get(), timeout=0.2)
            yield data
        except asyncio.TimeoutError:
            continue
    await task 
    yield f"data: {json.dumps({'status': 'DONE', 'image_url': f'/static/result.png?t={timestamp}'})}\n\n"

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    models = get_available_models()
    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={"request": request, "models": models}
    )

# ВАЖНО: Переключаем эндпоинт на POST, так как передаем файл через Form-data
@app.post("/generate-stream")
async def generate(
    prompt: str = Form(...), 
    negative_prompt: str = Form(""), 
    model: str = Form(""), 
    aspect_ratio: str = Form("1:1"), 
    style: str = Form("anime"),
    guidance_scale: float = Form(7.0), 
    steps: int = Form(30),
    lora_scale: float = Form(0.75),
    control_image: UploadFile = File(None) # Принимаем изображение с позой
):
    torch.cuda.empty_cache()
    pipeline = load_pipeline(model)
    
    if pipeline and hasattr(pipeline, "text_encoder"):
        pipeline.text_encoder.to("cpu")
    if pipeline and hasattr(pipeline, "text_encoder_2"):
        pipeline.text_encoder_2.to("cpu")
    
    aspect_ratios = {"1:1": (1024, 1024), "7:9": (896, 1152), "9:7": (1152, 896), "9:16": (640, 1344)}
    width, height = aspect_ratios.get(aspect_ratio, (1024, 1024))

    # Обрабатываем загруженное изображение-референс
    pose_image = None
    if control_image:
        image_bytes = await control_image.read()
        init_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        init_img = init_img.resize((width, height))
        
        # Извлекаем скелет позы (черный фон + цветные палочки)
        print("[ControlNet] Извлекаем карту позы OpenPose...")
        pose_image = pose_detector(init_img)
        # Отладочное сохранение карты позы (можно посмотреть, как нейронка увидела скелет)
        pose_image.save(os.path.join(STATIC_DIR, "detected_pose.png"))
    else:
        # Если картинку не загрузили, создаем пустую черную карту, чтобы ControlNet не ломал генерацию
        pose_image = Image.new("RGB", (width, height), (0, 0, 0))

    queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    current_timestamp = int(time.time())

    # Стили
    style_prefix = ""
    style_negative = ""
    clean_style = style.rstrip("/")
    if clean_style == "flat":
        style_prefix = "score_9, score_8_up, source_anime, flat color, clean lineart, minimalist background, "
        style_negative = "shaded, 3d, realistic, render, volumetric lighting, "
    elif clean_style == "manga":
        style_prefix = "score_9, score_8_up, source_anime, monochrome, black and white, manga page, screentone, "

    full_prompt = f"{style_prefix}{prompt}"
    final_negative = negative_prompt if negative_prompt.strip() else "worst quality, low quality, blurry"
    full_negative_prompt = f"{style_negative}{final_negative}"

    current_step = [0]

    def latents_callback(*args, **kwargs):
        current_step[0] += 1
        step = current_step[0]  # Вытаскиваем чистое число из списка
        display_step = min(step, steps)
        progress = int((display_step / steps) * 100)
        
        payload = f"data: {json.dumps({'step': display_step, 'total': steps, 'progress': progress})}\n\n"
        loop.call_soon_threadsafe(queue.put_nowait, payload)
        
        if kwargs.get("callback_kwargs"):
            return kwargs["callback_kwargs"]
        return {}

    def run_generation_sync():
        # Динамически выставляем силу Лоры
        if os.path.exists(LORA_PATH) and pipeline:
            print(f"[Core] Выставляем динамический вес Лоры...")
            pipeline.set_adapters(["sd_x1"], adapter_weights=[lora_scale])

        # Хитрый перехват: заворачиваем промпты в аниме-стиль прямо перед генерацией!
        local_prompt = f"{full_prompt}, masterpiece, exquisite, highly detailed anime style, vibrant colors, soft lighting"
        local_negative_prompt = f"{full_negative_prompt}, photo, deformed, low quality, blurry, realism, 3d"

        # Теперь отдаем в пайплайн наши модифицированные локальные переменные
        image_result = pipeline(
            prompt=local_prompt,
            negative_prompt=local_negative_prompt,
            image=pose_image,
            num_inference_steps=steps,
            width=width,
            height=height,
            original_size=(width, height),
            target_size=(width, height),
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=0.8,
            callback_on_step_end=latents_callback,
            callback_on_step_end_tensor_inputs=["latents"]
        )
        
        image = image_result.images[0]
        filename = f"result_{current_timestamp}.png"
        image.save(os.path.join(STATIC_DIR, filename))
        image.save(os.path.join(STATIC_DIR, "result.png"))
        return filename

    task = asyncio.create_task(asyncio.to_thread(run_generation_sync))
    
