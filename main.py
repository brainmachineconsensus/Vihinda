import gradio as gr
import pandas as pd
import lightgbm as lgb
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import shap
from utils import  create_yield_chart,create_feature_importance_plot,create_shap_plot,predict_yield_per_env
# Load model and data
model = lgb.Booster(model_file='lightgbm_model.txt')
df = pd.read_csv('data.csv')


# Create Gradio interface
with gr.Blocks(theme=gr.themes.Citrus()) as demo:
    gr.Markdown("""
    # 🌾 **VIHINDA** - Système de Prédiction des Rendements
    Analyse des rendements à travers différents environnements et compréhension des facteurs d'influence.
    """)
    
    with gr.Row():
        hybrid_input = gr.Dropdown(
            choices=sorted(df['Hybrid'].unique().tolist()),
            label="Sélectionner un Hybride",
            info="Choisissez un hybride à analyser"
        )
        predict_btn = gr.Button("Prédire les Rendements", variant="primary")

    with gr.Tabs() as tabs:
        with gr.Tab("Prédictions de Rendement"):
            with gr.Row():
                error_output = gr.Textbox(label="Statut", visible=False)
            
            with gr.Row():
                predictions_output = gr.Textbox(
                    label="Rendements Prédits par Environnement",
                    lines=15,
                    max_lines=20
                )
            
            with gr.Row():
                plot_output = gr.Plot(label="Distribution des Rendements")

        with gr.Tab("Analyse des Caractéristiques"):
            with gr.Row():
                with gr.Column():
                    feature_importance_plot = gr.Plot(
                        label="Importance des Caractéristiques"
                    )
            with gr.Row():
                with gr.Column():
                    shap_plot = gr.Plot(
                        label="Analyse SHAP"
                    )

    def process_prediction(hybrid):
        result = predict_yield_per_env(hybrid)
        
        if result["error"]:
            error_output.visible = True
            return result["error"], None, None, None, None
        
        error_output.visible = False
        return None, result["predictions"], result["chart"], result["feature_importance"], result["shap_plot"]

    predict_btn.click(
        process_prediction,
        inputs=[hybrid_input],
        outputs=[error_output, predictions_output, plot_output, feature_importance_plot, shap_plot]
    )

    gr.Markdown("""
    ### 📊 Comment utiliser cet outil:
    1. Sélectionnez un hybride dans le menu déroulant
    2. Cliquez sur 'Prédire les Rendements' pour voir les prédictions
    3. Consultez l'onglet 'Analyse des Caractéristiques' pour comprendre les facteurs clés
    4. L'analyse inclut les rendements par environnement et l'influence des caractéristiques
    """)
print("running")
# Launch application
demo.launch()