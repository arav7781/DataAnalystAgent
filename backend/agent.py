from typing import Annotated, Tuple, List
from typing_extensions import TypedDict
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
import os
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import uuid
from io import StringIO
import sys
import plotly.express as px
import plotly.io as pio
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import warnings
import logging

# Setup
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
from flask_cors import CORS
CORS(app)

# Environment variables (use actual environment variables in production)
os.environ["GROQ_API_KEY"] = "your_groq_api_key"
os.environ["LANGSMITH_API_KEY"] = "your_langsmith_api_key"
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "DataAnalyticsPipeline"

from langchain.chat_models import init_chat_model
llm = init_chat_model("groq:llama3-8b-8192")

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    input_data: list[dict]
    current_variables: dict
    intermediate_outputs: list[dict]
    output_image_paths: list[str]
    analysis_results: dict
    csv_path: str

persistent_vars = {}
analysis_cache = {}

def safe_get_message_content(msg):
    return getattr(msg, 'content', str(msg))

def safe_get_message_role(msg):
    if isinstance(msg, dict):
        return msg.get('role', 'assistant')
    return {'HumanMessage': 'user', 'AIMessage': 'assistant', 'SystemMessage': 'system'}.get(msg.__class__.__name__, 'assistant')

def make_tool_graph():
    @tool
    def complete_python_task(thought: str, python_code: str, csv_path: str, task_type: str = "analysis") -> Tuple[str, dict]:
        """Execute custom Python code for data analysis."""
        logger.info(f"Executing task: {task_type} with CSV: {csv_path}")
        current_variables = persistent_vars.copy()
        
        if not os.path.exists(csv_path):
            return f"Error: CSV not found at {csv_path}", {}
        
        try:
            if 'df' not in current_variables:
                current_variables["df"] = pd.read_csv(csv_path)
            
            os.makedirs("static/outputs/images", exist_ok=True)
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            exec_globals = globals().copy()
            exec_globals.update(current_variables)
            exec_globals.update({
                "pd": pd, "np": np, "px": px, "plt": plt, "sns": sns,
                "StandardScaler": StandardScaler, "LabelEncoder": LabelEncoder,
                "KMeans": KMeans, "PCA": PCA, "RandomForestClassifier": RandomForestClassifier,
                "RandomForestRegressor": RandomForestRegressor, "train_test_split": train_test_split,
                "classification_report": classification_report, "mean_squared_error": mean_squared_error,
                "r2_score": r2_score, "plotly_figures": [], "analysis_results": {}
            })
            
            exec(python_code, exec_globals)
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
            
            persistent_vars.update({k: v for k, v in exec_globals.items() if k not in globals() and not k.startswith('__')})
            
            output_image_paths = []
            for fig in exec_globals.get("plotly_figures", []):
                pickle_filename = f"static/outputs/images/{uuid.uuid4()}.pickle"
                html_filename = f"static/outputs/images/{uuid.uuid4()}.html"
                with open(pickle_filename, 'wb') as f:
                    pickle.dump(fig, f)
                fig.write_html(html_filename)
                output_image_paths.extend([pickle_filename, html_filename])
            
            if plt.get_fignums():
                for fig_num in plt.get_fignums():
                    img_filename = f"static/outputs/images/{uuid.uuid4()}.png"
                    plt.figure(fig_num).savefig(img_filename, dpi=300, bbox_inches='tight')
                    output_image_paths.append(img_filename)
                plt.close('all')
            
            return output or "Task completed", {
                "intermediate_outputs": [{"thought": thought, "code": python_code, "output": output, "task_type": task_type}],
                "current_variables": {k: str(type(v)) for k, v in persistent_vars.items()},
                "analysis_results": exec_globals.get("analysis_results", {}),
                "output_image_paths": output_image_paths
            }
        except Exception as e:
            sys.stdout = old_stdout
            return f"Error: {str(e)}", {"intermediate_outputs": [{"thought": thought, "code": python_code, "output": str(e)}]}

    @tool
    def automated_eda_tool(csv_path: str, analysis_depth: str = "standard") -> Tuple[str, dict]:
        """Perform automated exploratory data analysis."""
        if not os.path.exists(csv_path):
            return f"Error: CSV not found at {csv_path}", {}
        
        df = pd.read_csv(csv_path)
        eda_results = {
            "dataset_info": {"shape": df.shape, "columns": list(df.columns), "dtypes": df.dtypes.to_dict(), "missing_values": df.isnull().sum().to_dict()},
            "statistical_summary": df.describe().to_dict(),
            "insights": []
        }
        
        output_images = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Correlation heatmap
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            eda_results["correlations"] = corr_matrix.to_dict()
            fig = px.imshow(corr_matrix, text_auto=True, title="Correlation Heatmap")
            html_filename = f"static/outputs/images/{uuid.uuid4()}.html"
            fig.write_html(html_filename)
            output_images.append(html_filename)
        
        # Distribution plots
        if analysis_depth == "standard" and numeric_cols.any():
            for col in numeric_cols[:3]:
                fig = px.histogram(df, x=col, title=f'Distribution of {col}')
                html_filename = f"static/outputs/images/{uuid.uuid4()}.html"
                fig.write_html(html_filename)
                output_images.append(html_filename)
        
        # Insights
        missing_pct = df.isnull().sum() / len(df) * 100
        eda_results["insights"].extend([f"High missing data in {col} ({pct:.1f}%)" for col, pct in missing_pct.items() if pct > 20])
        for col in numeric_cols:
            if abs(df[col].skew()) > 1:
                eda_results["insights"].append(f"Column '{col}' is highly skewed (skewness: {df[col].skew():.2f})")
        
        return "EDA completed", {"analysis_results": eda_results, "output_image_paths": output_images}

    @tool
    def smart_visualization_tool(csv_path: str, viz_type: str = "auto", columns: List[str] = None) -> Tuple[str, dict]:
        """Create intelligent visualizations."""
        if not os.path.exists(csv_path):
            return f"Error: CSV not found at {csv_path}", {}
        
        df = pd.read_csv(csv_path)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        output_paths = []
        
        plotly_figures = []
        if viz_type == "auto":
            if numeric_cols:
                fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0], title="Scatter Plot")
                plotly_figures.append(fig)
            if categorical_cols:
                fig = px.bar(df[categorical_cols[0]].value_counts().head(10), title=f"Top Values in {categorical_cols[0]}")
                plotly_figures.append(fig)
        elif viz_type == "scatter" and len(numeric_cols) >= 2:
            fig = px.scatter(df, x=columns[0] if columns else numeric_cols[0], y=columns[1] if columns else numeric_cols[1], title="Scatter Plot")
            plotly_figures.append(fig)
        elif viz_type == "bar" and categorical_cols:
            fig = px.bar(df[categorical_cols[0]].value_counts().head(10), title=f"Top Values in {categorical_cols[0]}")
            plotly_figures.append(fig)
        
        for fig in plotly_figures:
            html_filename = f"static/outputs/images/{uuid.uuid4()}.html"
            fig.write_html(html_filename)
            output_paths.append(html_filename)
        
        return f"Generated {len(plotly_figures)} visualizations", {"output_image_paths": output_paths}

    @tool
    def ml_modeling_tool(csv_path: str, target_column: str, model_type: str = "auto", test_size: float = 0.2) -> Tuple[str, dict]:
        """Build and evaluate ML models."""
        if not os.path.exists(csv_path):
            return f"Error: CSV not found at {csv_path}", {}
        
        df = pd.read_csv(csv_path)
        if target_column not in df.columns:
            return f"Error: Target column '{target_column}' not found", {}
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        X = X.fillna(X.mean(numeric_only=True) if X.select_dtypes(include=[np.number]).shape[1] > 0 else 0)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        model_results = {}
        if model_type == "auto":
            model_type = "classification" if y.dtype == 'object' or y.nunique() < 10 else "regression"
        
        if model_type == "classification":
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            model_results = {
                "model_type": "RandomForestClassifier",
                "accuracy": model.score(X_test, y_test),
                "report": classification_report(y_test, y_pred, output_dict=True)
            }
        else:
            model = RandomForestRegressor(random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            model_results = {
                "model_type": "RandomForestRegressor",
                "r2_score": r2_score(y_test, y_pred),
                "mse": mean_squared_error(y_test, y_pred)
            }
        
        fig = px.bar(pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_}).sort_values('importance', ascending=False).head(10),
                     x='importance', y='feature', orientation='h', title='Feature Importance')
        html_filename = f"static/outputs/images/{uuid.uuid4()}.html"
        fig.write_html(html_filename)
        
        return f"Model trained: {model_type}", {"analysis_results": model_results, "output_image_paths": [html_filename]}

    tools = [complete_python_task, automated_eda_tool, smart_visualization_tool, ml_modeling_tool]
    tool_node = ToolNode(tools)
    llm_with_tools = llm.bind_tools(tools)

    def call_llm_model(state: State):
        system_prompt = """
        You are an expert data scientist specializing in automated data analysis. Use provided tools to:
        - Perform EDA with automated_eda_tool
        - Create visualizations with smart_visualization_tool
        - Build ML models with ml_modeling_tool
        - Execute custom code with complete_python_task
        For "analyze my data" requests, start with automated_eda_tool, then use smart_visualization_tool.
        Use csv_path from state. Provide clear insights and visualizations.
        """
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        if state.get("csv_path"):
            for msg in messages:
                if isinstance(msg, HumanMessage) and "csv_path" not in msg.content:
                    msg.content += f"\nCSV path: {state['csv_path']}"
        
        try:
            response = llm_with_tools.invoke(messages)
            return {"messages": [response], "csv_path": state.get("csv_path", "")}
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return {"messages": [AIMessage(content=f"Error: {str(e)}")], "csv_path": state.get("csv_path", "")}

    builder = StateGraph(State)
    builder.add_node("tool_calling_llm", call_llm_model)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "tool_calling_llm")
    builder.add_conditional_edges("tool_calling_llm", tools_condition)
    builder.add_edge("tools", "tool_calling_llm")
    
    return builder.compile()

tool_agent = make_tool_graph()

@app.route('/upload', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if not file.filename.lower().endswith('.csv'):
        return jsonify({"error": "Only CSV files supported"}), 400
    
    upload_dir = "static/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    filename = os.path.join(upload_dir, f"{uuid.uuid4()}.csv")
    
    try:
        file.save(filename)
        df = pd.read_csv(filename)
        return jsonify({
            "csv_path": filename,
            "message": "CSV uploaded successfully",
            "info": {"shape": df.shape, "columns": list(df.columns)}
        })
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({"error": f"Failed to save CSV: {str(e)}"}), 500

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    csv_path = data.get('csv_path', '')
    
    if not user_message or not csv_path or not os.path.exists(csv_path):
        return jsonify({"error": "Invalid message or CSV path"}), 400
    
    initial_state = {
        "messages": [HumanMessage(content=user_message)],
        "input_data": [{"variable_name": "df", "data_type": "csv", "data_path": csv_path}],
        "current_variables": {}, "intermediate_outputs": [], "output_image_paths": [],
        "analysis_results": {}, "csv_path": csv_path
    }
    
    result = tool_agent.invoke(initial_state)
    
    response_data = {
        "message": "Analysis completed",
        "intermediate_outputs": result.get("intermediate_outputs", []),
        "analysis_results": result.get("analysis_results", {}),
        "output_images": [{"path": p, "type": "html", "url": f"/serve_file/{p.replace('static/', '')}"} for p in result.get("output_image_paths", []) if p.endswith('.html')]
    }
    
    for msg in result.get("messages", []):
        response_data["messages"].append({"role": safe_get_message_role(msg), "content": safe_get_message_content(msg)})
    
    return jsonify(response_data)

@app.route('/serve_file/<path:filename>')
def serve_file(filename):
    return send_file(os.path.join('static', filename))

if __name__ == '__main__':
    app.run(debug=True)