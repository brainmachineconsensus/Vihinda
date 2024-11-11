
import lightgbm as lgb
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import shap
import pandas as pd

model = lgb.Booster(model_file='lightgbm_model.txt')
df = pd.read_csv('data.csv')


def create_yield_chart(predictions):
    """Create a bar chart for yield predictions"""
    envs, yields = zip(*predictions)
    
    fig = go.Figure(data=[
        go.Bar(
            x=[f"Env {env}" for env in envs],
            y=list(yields),
            marker_color='rgb(34, 139, 34)',
            marker=dict(
                line=dict(color='rgb(0, 100, 0)', width=1.5)
            )
        )
    ])
    
    fig.update_layout(
        title='Rendements Prédits par Environnement',
        xaxis_title='Environnement',
        yaxis_title='Rendement Prédit (Mg/ha)',
        template='plotly_white',
        plot_bgcolor='rgba(240, 255, 240, 0.5)'
    )
    
    return fig

def create_feature_importance_plot():
    """Create feature importance plot"""
    importance = model.feature_importance(importance_type='split')
    feature_names = model.feature_name()
    
    importance_df = pd.DataFrame({
        'Caractéristique': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=True).tail(20)
    
    fig = go.Figure(data=[
        go.Bar(
            y=importance_df['Caractéristique'],
            x=importance_df['Importance'],
            orientation='h',
            marker_color='rgb(34, 139, 34)'
        )
    ])
    
    fig.update_layout(
        title='Importance des Caractéristiques (Top 20)',
        xaxis_title='Score d\'Importance',
        yaxis_title='Caractéristique',
        template='plotly_white',
        height=800
    )
    
    return fig

def create_shap_plot(X_sample):
    """Create SHAP summary plot"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title('Analyse SHAP des Caractéristiques')
    return plt.gcf()

def predict_yield_per_env(hybrid_id):
    """Enhanced prediction function with SHAP analysis"""
    if hybrid_id not in df['Hybrid'].unique():
        return {
            "error": f"Erreur : L'hybride '{hybrid_id}' n'est pas présent dans les données.",
            "predictions": None,
            "chart": None,
            "feature_importance": None,
            "shap_plot": None
        }

    hybrid_data = df[df['Hybrid'] == hybrid_id]
    environments = hybrid_data['Env'].unique()
    yield_predictions = {}

    for env in environments:
        env_data = hybrid_data[hybrid_data['Env'] == env]
        X_env = env_data.drop(columns=['Yield_Mg_ha'])
        yield_pred = model.predict(X_env).mean()
        yield_predictions[env] = yield_pred

    sorted_predictions = sorted(yield_predictions.items(), key=lambda x: x[1], reverse=True)
    
    predictions_text = f"Rendements prédits pour l'hybride {hybrid_id} :\n\n"
    for env, yield_val in sorted_predictions:
        predictions_text += f"Environnement {env}: Rendement prédit = {yield_val:.2f} Mg/ha\n"

    # Create SHAP plot
    X_sample = df.drop(columns=['Yield_Mg_ha']).sample(min(100, len(df)))
    shap_plot = create_shap_plot(X_sample)

    return {
        "error": None,
        "predictions": predictions_text,
        "chart": create_yield_chart(sorted_predictions),
        "feature_importance": create_feature_importance_plot(),
        "shap_plot": shap_plot
    }
