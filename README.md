# FunStay Concierge — Asistente Conversacional de Voz

Proyecto final de **Herramientas de IA Clásica** — Máster en IA, Cloud Computing y DevOps.

Asistente conversacional **de voz extremo a extremo** para FunStay Florida, gestor de propiedades vacacionales en el área de Disney (Orlando). Bilingüe **español + inglés**, con clasificación de intenciones por Dialogflow ES, validación de reservas, generación dinámica de códigos de cerradura inteligente, y consulta semántica (RAG) sobre los manuales de las propiedades.

> **Cumplimiento del enunciado:** Dialogflow recibe y devuelve **solo texto**. El reconocimiento de voz (STT) lo hace **OpenAI Whisper** y la síntesis (TTS) **AWS Polly** — nunca se usa el STT/TTS integrado de Dialogflow.

## Arquitectura

```
[ Usuario ]  🎙️ voz
   │
   ▼
 STT: OpenAI Whisper  (auto-detect ES/EN)
   │ texto
   ▼
[ Dialogflow ES Agent ]   ── NLU · intents · entidades · contextos
   │ webhook (texto)
   ▼
[ FastAPI Webhook ]  ──► Mock DB (reservas, códigos de puerta, tickets)
   │                 └─► RAG: ChromaDB ◄─ OpenAI embeddings + GPT-4o-mini (con guardrail)
   │ texto respuesta
   ▼
 TTS: AWS Polly  (neural: Lucia es-ES / Joanna en-US)
   │ 🔊 audio
   ▼
[ Altavoz ]

Ingesta offline:  docs/*.pdf → pytesseract (OCR) → chunk → embeddings → ChromaDB
```

Dos clientes de voz equivalentes:
- **Navegador web** (`/demo`): página con micrófono (MediaRecorder) — *recomendado para la demo*.
- **Terminal** (`scripts/voice_client_aws.py`): push-to-talk con `sounddevice`.

## Stack

| Bloque | Componente | Tecnología |
|---|---|---|
| A — Agente | Clasificación de intenciones + contextos | Dialogflow ES (bilingüe ES/EN) |
| B — STT | Reconocimiento de voz | **OpenAI Whisper** (`whisper-1`, auto-detect ES/EN) |
| B — TTS | Síntesis de voz | **AWS Polly** neural (Lucia es-ES / Joanna en-US) |
| C — OCR | Extracción de documentos | pytesseract (open source) — sustituible por Azure Document Intelligence |
| C — RAG | Búsqueda semántica | ChromaDB (coseno) + OpenAI `text-embedding-3-small` |
| C — LLM | Generación de respuesta | OpenAI GPT-4o-mini con guardrail anti-alucinación |
| Webhook / Web | API de fulfillment + UI de voz | FastAPI + uvicorn (expuesto con ngrok) |

## Requisitos previos

- **Python 3.10+**
- **Tesseract OCR**: `brew install tesseract tesseract-lang` (Mac) o `sudo apt install tesseract-ocr tesseract-ocr-spa tesseract-ocr-eng` (Linux)
- **Poppler** (para pdf2image): `brew install poppler` o `sudo apt install poppler-utils`
- **ngrok**: `brew install ngrok` (requiere un authtoken gratuito: `ngrok config add-authtoken <token>`)
- **gcloud CLI** (para ADC): [cloud.google.com/sdk](https://cloud.google.com/sdk)
- Cuentas activas: **OpenAI** (API key con saldo), **AWS** (Polly), **Google Cloud** (agente Dialogflow ES)

## Setup

### 1. Instalar dependencias

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> El pin `httpx==0.27.2` es necesario: `openai==1.54` pasa el argumento `proxies` que `httpx>=0.28` eliminó.

### 2. Configurar variables de entorno

```bash
cp .env.example .env
# Edita .env con tus valores
```

Carga el `.env` exportando las variables a la shell:

```bash
set -a; source .env; set +a
```

### 3. Autenticación de Google Cloud (ADC)

No se usa un service account JSON, sino **Application Default Credentials**:

```bash
gcloud auth application-default login          # inicia sesión con la cuenta dueña del agente
```

`GOOGLE_PROJECT_ID` y `GOOGLE_CLOUD_PROJECT` deben apuntar al proyecto que aloja el agente
(Dialogflow Console → ⚙️ Settings → *Project ID*). La cuenta necesita el rol **Dialogflow API Client**.

### 4. Importar el agente de Dialogflow

Dialogflow ES Console → ⚙️ Settings → *Export and Import* → **RESTORE FROM ZIP** → sube `funstay_agent.zip`.
Idioma por defecto **Spanish — es** (incluye `en`). Pulsa **Train**.

### 5. Permisos de AWS (Polly)

La identidad de AWS (perfil `default` o claves) necesita `polly:SynthesizeSpeech`
(política gestionada **AmazonPollyReadOnlyAccess**).

### 6. Generar la base de conocimiento RAG (offline)

```bash
python -m scripts.ingest
```

Rasteriza los 3 PDFs de `docs/`, hace OCR con pytesseract (eng+spa), chunkea por párrafo,
genera embeddings con OpenAI y persiste en `data/chroma/`. Debe terminar con `✓ Ingested N chunks`.

### 7. Arrancar el webhook + UI web

```bash
uvicorn app.main:app --port 8000
```

### 8. Exponer con ngrok (en otra terminal)

```bash
ngrok http 8000
```

Copia la URL `https://xxxx.ngrok-free.dev` y pégala en Dialogflow → **Fulfillment** →
URL: `https://xxxx.ngrok-free.dev/webhook` (¡con `/webhook` al final!) → **Save**.

### 9. Probar — dos opciones

**A) Navegador (recomendado):** abre **http://localhost:8000/demo**, permite el micrófono,
pulsa 🎙️ y habla.

**B) Terminal:** `python -m scripts.voice_client_aws` (push-to-talk: Enter para grabar/parar).

## Guion de prueba (di las frases en este orden)

El agente exige validar la reserva antes de responder otras preguntas (gestión de contexto).

1. **"Hola"** → saludo bilingüe
2. **"Mi código de reserva es FS uno dos tres cuatro cinco seis"** → "¡Bienvenido, Anto García!…"
3. **"¿Cuánto cuesta calentar la piscina en invierno?"** → RAG: *35 dólares al día*
4. **"¿La piscina tiene tobogán?"** → **guardrail**: declina sin inventar
5. **"Hay una avería en el aire acondicionado"** → abre ticket `TKT-XXXXX`
6. **"Sí, confirmo"** → intención de seguimiento confirma el aviso
7. **"Adiós"** → despedida
8. *(EN)* **"Hi"** → **"My booking code is FS two three four five six seven"** → **"How do I get to the pool?"**

> Consejo: deletrea los dígitos del código ("uno, dos, tres…"); Whisper los transcribe mejor.
> El código se normaliza a `FS123456` antes de enviarlo a Dialogflow (la entidad regexp `FS\d{6}` distingue mayúsculas).

## Estructura del repo

```
funstay_webhook/
├── app/
│   ├── main.py              # FastAPI: /webhook + /demo (UI) + /voice (pipeline) + routing
│   ├── handlers.py          # Lógica por intent (usa la pregunta real del huésped en RAG)
│   ├── data.py              # Mock DB de reservas + generación de códigos + tickets
│   ├── rag.py               # ChromaDB + embeddings + GPT-4o-mini con guardrail
│   ├── voice_pipeline.py    # Whisper → Dialogflow → Polly (para la UI web)
│   └── static/demo.html     # Cliente de voz en el navegador
├── scripts/
│   ├── build_docs.py        # Genera los 3 PDFs de la KB
│   ├── ingest.py            # OCR + chunk + embed → ChromaDB
│   ├── voice_client_aws.py  # Cliente de voz por terminal (Whisper + Polly)
│   └── build_report.py      # Genera el informe PDF
├── docs/                    # PDFs de la knowledge base
├── data/chroma/             # Vector store persistente (generado por ingest)
├── funstay_agent.zip        # Exportación del agente Dialogflow ES
├── requirements.txt
├── .env.example
└── README.md
```

## Decisiones técnicas

**OpenAI Whisper (STT) y AWS Polly (TTS), no los de Dialogflow.** El enunciado prohíbe el
STT/TTS integrado de Dialogflow (supone un 0 en el bloque). Whisper aporta detección de idioma
nativa; Polly aporta voces neurales naturales y se alinea con el stack productivo (AWS).

**pytesseract en lugar de Azure Document Intelligence.** Open source y sin coste por documento.
La capa de OCR está aislada en `scripts/ingest.py` (`ocr_pdf_to_text_per_page`) para migrar sin tocar el resto.

**ChromaDB.** Persistente, embebido, con metadatos para filtrar por `topic`. Sin servidor adicional.

**GPT-4o-mini.** ~60× más barato que GPT-4 con calidad suficiente para respuestas de voz cortas.

**Guardrail anti-alucinación.** Si la distancia coseno del mejor chunk supera 1.2, el sistema declina
sin llamar al LLM; además el prompt obliga a responder solo con el contexto recuperado. Los handlers
RAG usan la **pregunta real** del huésped, de modo que las preguntas fuera de dominio se rechazan con elegancia.

**Arquitectura híbrida (clásica + generativa).** Dialogflow ES gestiona el flujo (intents, contextos,
slot filling) y las respuestas estáticas; el webhook delega en el LLM solo para `pool_y_jacuzzi`,
`recomendaciones_disney_orlando` y `transporte_a_parques`. Baja latencia para FAQ, profundidad generativa para preguntas abiertas.
