"""
Generate the project report PDF: Informe_FunStay_Concierge.pdf

Covers the rubric's required sections:
  - Caso de uso, público objetivo y justificación tecnológica
  - Diseño y arquitectura con diagrama de flujo de datos end-to-end

Usage:
  python -m scripts.build_report
"""
import os
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Flowable,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "Informe_FunStay_Concierge.pdf"

AUTHOR = "Antonio Rodríguez"
COURSE = "Máster en IA, Cloud Computing y DevOps — Herramientas de IA Clásica"
DATE = "Junio 2026"

# ---------- palette ----------
NAVY = colors.HexColor("#1F3A93")
TEAL = colors.HexColor("#138D90")
AMBER = colors.HexColor("#E08E0B")
GREY = colors.HexColor("#5A5A5A")
LIGHT = colors.HexColor("#EAF0F6")

styles = getSampleStyleSheet()
title_style = ParagraphStyle("T", parent=styles["Title"], fontSize=26, textColor=NAVY, spaceAfter=6)
sub_style = ParagraphStyle("Sub", parent=styles["Normal"], fontSize=12, textColor=GREY, alignment=TA_CENTER, spaceAfter=4)
h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=15, textColor=NAVY, spaceBefore=12, spaceAfter=7)
h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12, textColor=TEAL, spaceBefore=8, spaceAfter=4)
body = ParagraphStyle("B", parent=styles["BodyText"], fontSize=10.3, leading=14.5, alignment=TA_JUSTIFY, spaceAfter=7)
bullet = ParagraphStyle("Bu", parent=body, leftIndent=12, spaceAfter=3)
small = ParagraphStyle("Sm", parent=styles["BodyText"], fontSize=8.5, textColor=GREY, leading=11)
cap = ParagraphStyle("Cap", parent=styles["Normal"], fontSize=8.5, textColor=GREY, alignment=TA_CENTER, spaceBefore=4)


# ---------- architecture diagram flowable ----------
class ArchDiagram(Flowable):
    """End-to-end data-flow diagram drawn with reportlab primitives."""
    def __init__(self, width=460, height=340):
        super().__init__()
        self.width = width
        self.height = height

    def _box(self, c, x, y, w, h, text, fill, fg=colors.white, fs=8.2, lines=None):
        c.setFillColor(fill)
        c.setStrokeColor(colors.white)
        c.roundRect(x, y, w, h, 5, fill=1, stroke=0)
        c.setFillColor(fg)
        rows = lines if lines else [text]
        c.setFont("Helvetica-Bold", fs)
        total = len(rows) * (fs + 2)
        ty = y + h / 2 + total / 2 - fs
        for r in rows:
            c.drawCentredString(x + w / 2, ty, r)
            ty -= (fs + 2)

    def _arrow(self, c, x1, y1, x2, y2, color=GREY, label=None):
        c.setStrokeColor(color)
        c.setLineWidth(1.4)
        c.line(x1, y1, x2, y2)
        # arrowhead
        import math
        ang = math.atan2(y2 - y1, x2 - x1)
        c.setFillColor(color)
        s = 5
        c.line(x2, y2, x2 - s * math.cos(ang - 0.4), y2 - s * math.sin(ang - 0.4))
        c.line(x2, y2, x2 - s * math.cos(ang + 0.4), y2 - s * math.sin(ang + 0.4))
        if label:
            c.setFillColor(color)
            c.setFont("Helvetica-Oblique", 7)
            c.drawCentredString((x1 + x2) / 2, (y1 + y2) / 2 + 3, label)

    def draw(self):
        c = self.canv
        W = self.width
        # --- Title band: RUNTIME (online) ---
        c.setFillColor(NAVY)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(0, 322, "TIEMPO REAL  (online)")

        bw, bh = 92, 40
        y1 = 270
        # row 1: mic -> whisper -> dialogflow -> webhook
        self._box(c, 0,   y1, 70, bh, "", TEAL, lines=["Huésped", "micrófono"])
        self._box(c, 95,  y1, bw, bh, "", AMBER, lines=["Whisper STT", "(OpenAI)", "ES/EN auto"])
        self._box(c, 205, y1, bw, bh, "", NAVY, lines=["Dialogflow ES", "NLU · intents", "entidades · contextos"])
        self._box(c, 315, y1, bw, bh, "", NAVY, lines=["Webhook", "FastAPI / uvicorn", "(ngrok)"])
        self._arrow(c, 70,  y1 + bh/2, 95,  y1 + bh/2, label="audio")
        self._arrow(c, 187, y1 + bh/2, 205, y1 + bh/2, label="texto")
        self._arrow(c, 297, y1 + bh/2, 315, y1 + bh/2, label="texto")

        # webhook -> backend (down)
        y2 = 195
        self._box(c, 250, y2, 70, 38, "", GREY, lines=["Mock DB", "reservas"])
        self._box(c, 330, y2, bw, 38, "", TEAL, lines=["RAG", "ChromaDB + LLM", "GPT-4o-mini"])
        self._arrow(c, 350, y1, 300, y2 + 38)        # webhook -> mockdb
        self._arrow(c, 365, y1, 380, y2 + 38)        # webhook -> rag

        # return path: dialogflow fulfillment -> polly -> speaker
        y3 = 120
        self._box(c, 205, y3, bw, bh, "", NAVY, lines=["Dialogflow", "fulfillment", "texto respuesta"])
        self._box(c, 95,  y3, bw, bh, "", AMBER, lines=["Polly TTS", "(AWS)", "Lucia / Joanna"])
        self._box(c, 0,   y3, 70, bh, "", TEAL, lines=["Huésped", "altavoz"])
        self._arrow(c, 360, y2, 297, y3 + bh/2)      # backend -> dialogflow fulfillment
        self._arrow(c, 205, y3 + bh/2, 187, y3 + bh/2, label="texto")
        self._arrow(c, 95,  y3 + bh/2, 70,  y3 + bh/2, label="audio")

        # compliance note
        c.setFillColor(AMBER)
        c.setFont("Helvetica-Oblique", 7.2)
        c.drawString(0, y3 - 14, "Dialogflow procesa SÓLO texto — STT (Whisper) y TTS (Polly) son externos (cumple la restricción del enunciado).")

        # --- Offline ingestion band ---
        c.setFillColor(NAVY)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(0, 78, "INGESTA DE CONOCIMIENTO  (offline)")
        y4 = 30
        ow = 80
        self._box(c, 0,   y4, ow, 36, "", GREY, lines=["3 PDFs", "manuales/guía"])
        self._box(c, 95,  y4, ow, 36, "", AMBER, lines=["OCR", "pytesseract", "eng+spa"])
        self._box(c, 190, y4, ow, 36, "", GREY, lines=["Chunking", "por párrafo"])
        self._box(c, 285, y4, ow, 36, "", AMBER, lines=["Embeddings", "text-embedding", "-3-small"])
        self._box(c, 380, y4, ow, 36, "", TEAL, lines=["ChromaDB", "vector store", "(coseno)"])
        for x1, x2 in [(80,95),(175,190),(270,285),(365,380)]:
            self._arrow(c, x1, y4 + 18, x2, y4 + 18)


def hr(color=LIGHT, h=2):
    class _HR(Flowable):
        def __init__(s):
            super().__init__(); s.width = 0; s.height = h
        def wrap(s, aw, ah):
            s.width = aw; return aw, h
        def draw(s):
            s.canv.setFillColor(color); s.canv.rect(0, 0, s.width, h, fill=1, stroke=0)
    return _HR()


def li(text):
    return Paragraph(f"•&nbsp;&nbsp;{text}", bullet)


def build():
    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4,
        topMargin=18 * mm, bottomMargin=16 * mm, leftMargin=20 * mm, rightMargin=20 * mm,
        title="Informe FunStay Concierge", author=AUTHOR,
    )
    s = []

    # ---------- COVER ----------
    s.append(Spacer(1, 40 * mm))
    s.append(Paragraph("FunStay Concierge", title_style))
    s.append(Paragraph("Asistente de voz multilingüe end-to-end para gestión de propiedades vacacionales", sub_style))
    s.append(Spacer(1, 6 * mm))
    s.append(hr(NAVY, 3))
    s.append(Spacer(1, 8 * mm))
    s.append(Paragraph("Evaluación Final · IA Conversacional", sub_style))
    s.append(Paragraph(COURSE, sub_style))
    s.append(Spacer(1, 30 * mm))
    s.append(Paragraph(f"<b>Autor:</b> {AUTHOR}", sub_style))
    s.append(Paragraph(DATE, sub_style))
    s.append(PageBreak())

    # ---------- 1. CASO DE USO ----------
    s.append(Paragraph("1. Caso de uso y público objetivo", h1))
    s.append(Paragraph(
        "<b>FunStay Concierge</b> es un conserje virtual de voz, bilingüe (español e inglés), para "
        "gestores de alquileres vacacionales de corta estancia en el área de Disney/Orlando (Florida). "
        "El cliente representativo es <b>FunStay Florida</b> (Mike Chen), que opera entre 30 y 80 villas "
        "en seis comunidades de resort —ChampionsGate, Reunion, Storey Lake, Windsor Hills, Solterra y "
        "Bear's Den—. Su personal recibe cada día decenas de llamadas repetitivas: contraseña del WiFi, "
        "cómo calentar la piscina, código de la puerta, horarios de check-in/out, transporte a los parques "
        "o incidencias de mantenimiento.", body))
    s.append(Paragraph(
        "El sistema automatiza esa primera línea de atención por voz, 24/7 y en el idioma del huésped "
        "(mayoritariamente turistas internacionales). El huésped llama, se identifica con su <b>código de "
        "reserva</b> y, una vez validado, obtiene respuestas inmediatas a sus preguntas o abre un ticket "
        "de incidencia —liberando al gestor para tareas de mayor valor y mejorando la experiencia del cliente.", body))

    s.append(Paragraph("Originalidad y valor", h2))
    s.append(li("<b>Dominio vertical real</b>: integra lógica de negocio (validación de reservas, generación "
                "de códigos de acceso de un solo uso estilo Seam.co/Nuki, tickets de mantenimiento), no es un FAQ genérico."))
    s.append(li("<b>Conocimiento propietario por documento</b>: cada villa tiene su manual; el RAG recupera "
                "información específica (p. ej. el coste exacto de calentar la piscina por temporada)."))
    s.append(li("<b>Experiencia puramente auditiva y bilingüe</b> con detección automática de idioma, "
                "pensada para uso telefónico/manos libres."))

    # ---------- 2. ARQUITECTURA ----------
    s.append(Paragraph("2. Diseño y arquitectura de la solución", h1))
    s.append(Paragraph(
        "La solución se organiza en cuatro bloques —Agente conversacional, Servicios de voz, Pipeline RAG "
        "y Lógica de negocio— unidos por un webhook. El siguiente diagrama muestra el flujo de datos "
        "completo, tanto en tiempo real como en la ingesta offline de conocimiento.", body))
    s.append(Spacer(1, 4))
    s.append(ArchDiagram())
    s.append(Paragraph("Figura 1 — Flujo de datos end-to-end (tiempo real + ingesta offline).", cap))
    s.append(Spacer(1, 6))
    s.append(Paragraph(
        "<b>Recorrido de una interacción:</b> el audio del huésped se captura por micrófono (cliente "
        "push-to-talk en terminal) y se transcribe con <b>OpenAI Whisper</b>, que además detecta el idioma. "
        "El texto se envía a <b>Dialogflow ES</b> vía <i>detect_intent</i>, que resuelve la intención, extrae "
        "entidades y gestiona el estado de la conversación mediante contextos. Las intenciones dinámicas "
        "invocan un <b>webhook (FastAPI)</b> expuesto con <b>ngrok</b>, que consulta la base de datos de "
        "reservas o el <b>pipeline RAG</b>. La respuesta en texto vuelve a Dialogflow y se sintetiza con "
        "<b>AWS Polly</b> (voz neural), reproduciéndose por el altavoz.", body))

    s.append(Paragraph("Bloque A — Agente conversacional (Dialogflow ES)", h2))
    s.append(li("<b>12 intenciones personalizadas</b> (sin contar Welcome/Fallback): mezcla de respuestas "
                "estáticas (wifi, check-in, normas, despedida) y dinámicas vía webhook (validar reserva, código "
                "de puerta, piscina/jacuzzi, recomendaciones, transporte, incidencias)."))
    s.append(li("<b>Entidades</b>: de sistema y personalizadas, incluyendo <b>Regexp</b> "
                "<font face='Courier'>FS\\d{6}</font> para el código de reserva y una entidad de lista "
                "<i>propiedad</i> con sinónimos para las seis comunidades."))
    s.append(li("<b>Gestión de contexto y multi-turno</b>: contextos <i>reserva-pendiente</i>, "
                "<i>reserva-validada</i> (lifespan 50) y <i>confirmar-incidencia</i>, con intenciones de "
                "seguimiento (sí/no) para confirmar el aviso al equipo. La mayoría de intenciones exigen "
                "<i>reserva-validada</i>, modelando un flujo seguro: identifícate antes de obtener datos del alojamiento."))

    s.append(Paragraph("Bloque B — Servicios de voz (STT + TTS)", h2))
    s.append(li("<b>STT: OpenAI Whisper</b> (<font face='Courier'>whisper-1</font>) con detección de idioma "
                "nativa (ES/EN) y salida en <i>verbose_json</i>."))
    s.append(li("<b>TTS: AWS Polly</b> motor <i>neural</i> — voz <b>Lucia</b> (es-ES) y <b>Joanna</b> (en-US); "
                "<i>fallback</i> automático a motor estándar si la voz neural no estuviera disponible."))
    s.append(li("<b>Cumplimiento del enunciado</b>: Dialogflow recibe y devuelve <b>solo texto</b> "
                "(<i>TextInput</i> / <i>fulfillment_text</i>). No se usa el STT ni el TTS integrados de "
                "Dialogflow en ningún punto, evitando el 0 del bloque."))

    s.append(Paragraph("Bloque C — Pipeline RAG y lógica de negocio", h2))
    s.append(li("<b>Ingesta (offline)</b>: OCR de 3 PDFs con <b>pytesseract</b> (eng+spa) → <i>chunking</i> "
                "por párrafo → <i>embeddings</i> con <b>text-embedding-3-small</b> (1536 dim) → "
                "<b>ChromaDB</b> persistente (distancia coseno), con metadatos de fuente, página y <i>topic</i>."))
    s.append(li("<b>Consulta (tiempo real)</b>: la pregunta se embebe y se recuperan los <i>top-k</i> "
                "fragmentos, filtrados por <i>topic</i> según la intención; se envían como contexto a "
                "<b>GPT-4o-mini</b> con instrucciones estrictas de fidelidad."))
    s.append(li("<b>Guardarraíl anti-alucinación</b>: si la distancia coseno del mejor resultado supera "
                "<b>1.2</b>, el sistema declina (\"no tengo esa información\") sin llamar al LLM; además el "
                "<i>prompt</i> obliga a responder solo con el contexto recuperado."))
    s.append(li("<b>Lógica de negocio</b>: validación de reservas contra BD mock, generación de códigos de "
                "puerta de un solo uso y apertura de tickets de incidencia (TKT-XXXXX)."))

    # ---------- 3. JUSTIFICACIÓN TECNOLÓGICA ----------
    s.append(Paragraph("3. Justificación tecnológica y decisiones", h1))
    # cell paragraph styles (so text wraps inside columns)
    cell = ParagraphStyle("cell", parent=styles["BodyText"], fontSize=8.2, leading=10.5, spaceAfter=0)
    cell_b = ParagraphStyle("cellb", parent=cell, fontName="Helvetica-Bold")
    cell_h = ParagraphStyle("cellh", parent=cell, fontName="Helvetica-Bold", textColor=colors.white)
    def P(txt, st):
        return Paragraph(txt, st)
    rows_raw = [
        ("Componente", "Tecnología elegida", "Justificación / alternativa descartada"),
        ("Agente NLU", "Dialogflow ES", "Requisito de la asignatura; gestión de contextos y entidades madura (CX sería el estándar industrial)."),
        ("STT", "OpenAI Whisper", "Detección de idioma nativa ES/EN; ya se usa OpenAI para el RAG. Alternativa: Azure Speech."),
        ("TTS", "AWS Polly (neural)", "Voces neurales naturales; alineado con el stack productivo previsto (AWS). Alternativa: Google/Azure TTS."),
        ("OCR", "pytesseract", "Coste cero para el piloto; capa aislada en ingest.py para migrar a Azure Document Intelligence sin tocar el resto."),
        ("Vector store", "ChromaDB", "Persistencia local sin servidor; metadatos para filtrar por topic. Alternativa: Pinecone/FAISS."),
        ("Embeddings", "text-embedding-3-small", "Buena relación calidad/coste para textos cortos conversacionales."),
        ("LLM", "GPT-4o-mini", "~60x más barato que GPT-4, calidad suficiente para respuestas breves de voz."),
        ("Webhook", "FastAPI + uvicorn", "Asíncrono, ligero y tipado; expuesto vía ngrok para conectar Dialogflow al backend local."),
    ]
    data = []
    for i, (a, b, cc) in enumerate(rows_raw):
        if i == 0:
            data.append([P(a, cell_h), P(b, cell_h), P(cc, cell_h)])
        else:
            data.append([P(a, cell_b), P(b, cell_b), P(cc, cell)])
    t = Table(data, colWidths=[70, 92, 278])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#C9D4E0")),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    s.append(t)
    s.append(Spacer(1, 8))

    s.append(Paragraph("Manejo de casos sin respuesta (UX)", h2))
    s.append(Paragraph(
        "El sistema gestiona la ausencia de información de forma explícita y honesta: el guardarraíl de "
        "distancia evita alucinar y, cuando no hay datos en el manual, ofrece abrir un ticket para que un "
        "humano (Mike) contacte al huésped. Esto prioriza la confianza sobre la cobertura, criterio clave "
        "en un asistente de atención al cliente.", body))

    s.append(Paragraph("Robustez y latencia", h2))
    s.append(Paragraph(
        "Las intenciones estáticas responden de forma instantánea; solo las dinámicas incurren en la "
        "latencia del webhook + LLM. El cliente de voz es push-to-talk para delimitar el turno con claridad. "
        "La autenticación a Google Cloud usa Application Default Credentials (ADC) con proyecto de cuota, y "
        "los servicios externos (Whisper, Polly) degradan con <i>fallbacks</i> controlados.", body))

    s.append(Spacer(1, 6))
    s.append(hr(LIGHT, 2))
    s.append(Paragraph(
        f"FunStay Concierge — {AUTHOR} — {DATE}. Entregables: informe (este documento), vídeo de "
        "demostración, código fuente con README y exportación del agente Dialogflow.", small))

    doc.build(s)
    print(f"✓ Report written to {OUT}  ({OUT.stat().st_size//1024} KB)")


if __name__ == "__main__":
    build()
