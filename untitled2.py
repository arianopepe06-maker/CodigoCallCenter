import os
import sys

# ---------------------------------------------------------
# 1. Carga e instalación automática del modelo de spaCy
# ---------------------------------------------------------
import spacy

try:
    nlp = spacy.load("es_core_news_sm")
except OSError:
    os.system("python -m spacy download es_core_news_sm")
    nlp = spacy.load("es_core_news_sm")

# ---------------------------------------------------------
# 2. Importación de las demás librerías
# ---------------------------------------------------------
import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pysentimiento import create_analyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Librerías para generación del reporte PDF
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Inicialización de analizador de sentimientos en español
analyzer = create_analyzer(task="sentiment", lang="es")

# ---------------------------------------------------------
# 3. Lógica principal de procesamiento
# ---------------------------------------------------------
def analizar_llamadas(archivo):
    if archivo is None:
        return "Por favor sube un archivo CSV válido.", None, None

    # Leer CSV
    df = pd.read_csv(archivo.name)
    
    if 'texto' not in df.columns:
        return "El archivo CSV debe contener una columna llamada 'texto'.", None, None

    textos = df['texto'].astype(str).tolist()

    # Análisis de Sentimiento con pysentimiento
    sentimientos = []
    for t in textos:
        res = analyzer.predict(t)
        sentimientos.append(res.output)
    
    df['sentimiento'] = sentimientos

    # Extracción de Palabras Clave y Agrupamiento (K-Means)
    vectorizer = TfidfVectorizer(max_features=10, stop_words='english')
    X = vectorizer.fit_transform(textos)
    
    n_clusters = min(3, len(textos))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['categoria'] = kmeans.fit_predict(X)

    # Conteo de Sentimientos
    conteo_sentimientos = df['sentimiento'].value_counts().to_dict()
    
    # Generación de resumen de texto
    resumen_txt = f"Total de llamadas analizadas: {len(df)}\n\n"
    resumen_txt += "Desglose de Sentimientos:\n"
    for k, v in conteo_sentimientos.items():
        resumen_txt += f"- {k.capitalize()}: {v}\n"

    # ---------------------------------------------------------
    # 4. Generación de Reporte PDF
    # ---------------------------------------------------------
    pdf_path = "reporte_call_center.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=12
    )

    story.append(Paragraph("Reporte de Análisis de Call Center", title_style))
    story.append(Spacer(1, 12))
    
    text_p = f"Se han procesado exitosamente {len(df)} registros de audio/texto."
    story.append(Paragraph(text_p, styles['Normal']))
    story.append(Spacer(1, 12))

    # Tabla de resultados
    data_tabla = [["Texto", "Sentimiento", "Categoría"]]
    for idx, row in df.iterrows():
        data_tabla.append([row['texto'][:40] + "...", row['sentimiento'], f"Grupo {row['categoria']}"])

    tabla = Table(data_tabla)
    tabla.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
    ]))
    
    story.append(tabla)
    doc.build(story)

    return resumen_txt, df[['texto', 'sentimiento', 'categoria']], pdf_path

# ---------------------------------------------------------
# 5. Interfaz de usuario con Gradio
# ---------------------------------------------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📞 Sistema de Análisis de Calidad en Call Center")
    gr.Markdown("Sube un archivo CSV con las transcripciones de las llamadas para analizar el sentimiento y generar reportes.")

    with gr.Row():
        archivo_input = gr.File(label="Subir archivo CSV", file_types=[".csv"])
        btn_procesar = gr.Button("Procesar Llamadas", variant="primary")

    with gr.Row():
        resumen_output = gr.Textbox(label="Resumen del Análisis", lines=6)
        tabla_output = gr.Dataframe(label="Detalle de Resultados")

    pdf_output = gr.File(label="Descargar Reporte PDF")

    btn_procesar.click(
        fn=analizar_llamadas,
        inputs=[archivo_input],
        outputs=[resumen_output, tabla_output, pdf_output]
    )

# ---------------------------------------------------------
# 6. Configuración de ejecución en servidor/cloud (Render)
# ---------------------------------------------------------
if __name__ == "__main__":
    # Lee el puerto asignado dinámicamente por la plataforma en la nube (Render)
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
