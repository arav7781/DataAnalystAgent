from typing import Annotated, Tuple, List, Optional
from typing_extensions import TypedDict
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
import os
import hashlib
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import uuid
from io import StringIO
import sys
import plotly.graph_objects as go
import plotly.express as px
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
warnings.filterwarnings('ignore')
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)



from langchain.chat_models import init_chat_model
llm = init_chat_model("groq:llama3-8b-8192")

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    input_data: list[dict]
    current_variables: dict
    intermediate_outputs: list[dict]
    output_image_paths: list[dict]
    analysis_results: dict
    csv_path: str
    analysis_history: list[dict]

# Global variables for persistent state
persistent_vars = {}
analysis_cache = {}
analysis_history = []
request_cache = set()  # To prevent duplicate requests

def safe_get_message_content(msg):
    if hasattr(msg, 'content'):
        return msg.content
    elif hasattr(msg, 'text'):
        return msg.text
    elif isinstance(msg, dict):
        return msg.get('content', str(msg))
    return str(msg)

def safe_get_message_role(msg):
    if hasattr(msg, 'role'):
        return msg.role
    elif hasattr(msg, '__class__'):
        class_name = msg.__class__.__name__.lower()
        if 'human' in class_name:
            return "user"
        elif 'ai' in class_name or 'assistant' in class_name:
            return "assistant"
        elif 'system' in class_name:
            return "system"
    elif isinstance(msg, dict):
        return msg.get('role', 'assistant')
    return "assistant"

def make_tool_graph():
    @tool
    def complete_python_task(
        thought: str,
        python_code: str,
        csv_path: str,
        task_type: str = "analysis"
    ) -> Tuple[str, dict]:
        """
        Advanced data analytics tool with pandas, plotly, and sklearn support.
        
        Args:
            thought (str): Reasoning behind the operation
            python_code (str): Python code for data processing/analysis
            csv_path (str): Path to input CSV file
            task_type (str): Type of task (analysis, visualization, modeling, etc.)
        
        Returns:
            Tuple[str, dict]: Execution output and updated state
        """
        # Normalize csv_path
        csv_path = os.path.normpath(csv_path)
        logger.debug(f"Processing CSV at path: {csv_path}")
        current_variables = persistent_vars.copy()

        if not os.path.exists(csv_path):
            error_msg = f"Error: CSV not found at {csv_path}"
            logger.error(error_msg)
            return error_msg, {
                "intermediate_outputs": [{"thought": thought, "code": python_code, "output": error_msg}]
            }

        try:
            if 'df' not in current_variables:
                current_variables["df"] = pd.read_csv(csv_path)
                logger.debug(f"Loaded CSV with shape: {current_variables['df'].shape}")

            os.makedirs("static/outputs/plotly_figures/pickle", exist_ok=True)
            os.makedirs("static/outputs/images", exist_ok=True)
            
            old_stdout = sys.stdout
            sys.stdout = StringIO()

            exec_globals = globals().copy()
            exec_globals.update(current_variables)
            exec_globals.update({
                "pd": pd,
                "np": np,
                "go": go,
                "px": px,
                "plt": plt,
                "sns": sns,
                "StandardScaler": StandardScaler,
                "LabelEncoder": LabelEncoder,
                "KMeans": KMeans,
                "PCA": PCA,
                "RandomForestClassifier": RandomForestClassifier,
                "RandomForestRegressor": RandomForestRegressor,
                "train_test_split": train_test_split,
                "classification_report": classification_report,
                "mean_squared_error": mean_squared_error,
                "r2_score": r2_score,
                "plotly_figures": [],
                "analysis_results": {},
                "model_metrics": {}
            })

            exec(python_code, exec_globals)

            output = sys.stdout.getvalue()
            sys.stdout = old_stdout

            persistent_vars.update({k: v for k, v in exec_globals.items() 
                                  if k not in globals() and not k.startswith('__')})

            output_image_paths = []
            if exec_globals.get("plotly_figures"):
                for fig in exec_globals["plotly_figures"]:
                    pickle_filename = os.path.normpath(f"static/outputs/plotly_figures/pickle/{uuid.uuid4()}.pickle")
                    html_filename = os.path.normpath(f"static/outputs/images/{uuid.uuid4()}.html")
                    with open(pickle_filename, 'wb') as f:
                        pickle.dump(fig, f)
                    fig.write_html(html_filename)
                    output_image_paths.append({
                        "path": pickle_filename,
                        "type": "pickle",
                        "url": f"/serve_file/{pickle_filename.replace('static/', '').replace(os.sep, '/')}"
                    })
                    output_image_paths.append({
                        "path": html_filename,
                        "type": "html",
                        "url": f"/serve_file/{html_filename.replace('static/', '').replace(os.sep, '/')}"
                    })

            if plt.get_fignums():
                for fig_num in plt.get_fignums():
                    fig = plt.figure(fig_num)
                    img_filename = os.path.normpath(f"static/outputs/images/{uuid.uuid4()}.png")
                    fig.savefig(img_filename, dpi=300, bbox_inches='tight')
                    output_image_paths.append({
                        "path": img_filename,
                        "type": "png",
                        "url": f"/serve_file/{img_filename.replace('static/', '').replace(os.sep, '/')}"
                    })
                plt.close('all')

            updated_state = {
                "intermediate_outputs": [{
                    "thought": thought,
                    "code": python_code,
                    "output": output or "Operation completed successfully",
                    "task_type": task_type,
                    "timestamp": datetime.now().isoformat()
                }],
                "current_variables": {k: str(type(v)) if not isinstance(v, (str, int, float, bool)) else v 
                                    for k, v in persistent_vars.items()},
                "analysis_results": exec_globals.get("analysis_results", {}),
                "model_metrics": exec_globals.get("model_metrics", {}),
                "output_image_paths": output_image_paths
            }

            analysis_history.append({
                "task_type": task_type,
                "thought": thought,
                "timestamp": datetime.now().isoformat(),
                "output_paths": [path["path"] for path in output_image_paths]
            })

            logger.debug(f"Data processing successful. paths: {[p['path'] for p in output_image_paths]}")
            return output or "Success", updated_state

        except Exception as e:
            sys.stdout = old_stdout
            error_msg = f"Error: {str(e)}"
            logger.error(error_msg)
            return error_msg, {
                "intermediate_outputs": [
                    {
                        "thought": thought,
                        "code": python_code,
                        "output": error_msg,
                        "task_type": task_type,
                        "timestamp": datetime.now().isoformat(),
                    }
                ]
            }

    @tool
    def automated_eda_tool(
        csv_path: str,
        analysis_level: str = "comprehensive"
    ) -> Tuple[str, dict]:
        """
        Automated Exploratory Data Analysis tool.
        
        Args:
            csv_path (str): Path to input CSV file
            analysis_level (str): Level of analysis (quick, standard, comprehensive)
        
        Returns:
            Tuple[str, dict]: EDA results and updated state
        """
        # Normalize csv_path
        csv_path = os.path.normpath(csv_path)
        try:
            if not os.path.exists(csv_path):
                error_msg = f"Error: CSV not found at {csv_path}"
                logger.error(error_msg)
                return error_msg, {}
                
            df = pd.read_csv(csv_path)
            logger.debug(f"Starting automated EDA for dataset with shape: {df.shape}")
            
            eda_results = {
                "dataset_info": {
                    "shape": df.shape,
                    "columns": list(df.columns),
                    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                    "memory_usage": df.memory_usage(deep=True).sum(),
                    "missing_values": df.isnull().sum().to_dict(),
                    "duplicate_rows": df.duplicated().sum(),
                    "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
                    "categorical_columns": df.select_dtypes(include=['object']).columns.tolist()
                },
                "statistical_summary": df.describe().to_dict(),
                "correlations": {},
                "insights": [],
                "outliers": {},
                "skewness": {}
            }
            
            output_image_paths = []
            plotly_figures = []

            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                eda_results["correlations"] = corr_matrix.to_dict()
                fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                              title="Correlation Heatmap", color_continuous_scale='RdBu_r')
                plotly_figures.append(fig)

            insights = []
            missing_pct = (df.isnull().sum() / len(df) * 100)
            high_missing = missing_pct[missing_pct > 20]
            if not high_missing.empty:
                insights.append(f"High missing data (>20%) in columns: {list(high_missing.index)}")
            
            for col in numeric_cols:
                skewness = df[col].skew()
                eda_results["skewness"][col] = skewness
                if abs(skewness) > 1:
                    insights.append(f"Column '{col}' is highly skewed (skewness: {skewness:.2f})")
                
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
                if len(outliers) > 0:
                    eda_results["outliers"][col] = len(outliers)
                    insights.append(f"Column '{col}' has {len(outliers)} potential outliers")
            
            eda_results["insights"] = insights
            
            if analysis_level in ["standard", "comprehensive"]:
                for col in numeric_cols[:4]:
                    fig = px.histogram(df, x=col, title=f'Distribution of {col}',
                                     nbins=50, marginal="box")
                    plotly_figures.append(fig)
                
                categorical_cols = df.select_dtypes(include=['object']).columns
                for col in categorical_cols[:3]:
                    value_counts = df[col].value_counts().head(10)
                    fig = px.bar(x=value_counts.index, y=value_counts.values,
                               title=f'Top 10 Values in {col}',
                               labels={'x': col, 'y': 'Count'})
                    plotly_figures.append(fig)
                
                if len(numeric_cols) >= 2 and analysis_level == "comprehensive":
                    fig = px.scatter_matrix(df[numeric_cols[:4]],
                                          title="Pair Plot of Numeric Variables",
                                          dimensions=numeric_cols[:4])
                    plotly_figures.append(fig)

            for fig in plotly_figures:
                pickle_filename = os.path.normpath(f"static/outputs/plotly_figures/pickle/{uuid.uuid4()}.pickle")
                html_filename = os.path.normpath(f"static/outputs/images/{uuid.uuid4()}.html")
                with open(pickle_filename, 'wb') as f:
                    pickle.dump(fig, f)
                fig.write_html(html_filename)
                output_image_paths.extend([
                    {"path": pickle_filename, "type": "pickle", 
                     "url": f"/serve_file/{pickle_filename.replace('static/', '').replace(os.sep, '/')}"},
                    {"path": html_filename, "type": "html", 
                     "url": f"/serve_file/{html_filename.replace('static/', '').replace(os.sep, '/')}"}
                ])

            analysis_history.append({
                "task_type": "eda",
                "thought": f"Automated EDA with {analysis_level} analysis",
                "timestamp": datetime.now().isoformat(),
                "output_paths": [path["path"] for path in output_image_paths]
            })

            logger.debug(f"Automated EDA completed. Generated {len(plotly_figures)} visualizations")
            return f"Automated EDA completed for {df.shape[0]} rows and {df.shape[1]} columns", {
                "analysis_results": eda_results,
                "output_image_paths": output_image_paths
            }
            
        except Exception as e:
            error_msg = f"EDA error: {str(e)}"
            logger.error(error_msg)
            return error_msg, {}

    @tool
    def smart_visualization_tool(
        csv_path: str,
        viz_type: str = "auto",
        columns: List[str] = None,
        max_visualizations: int = 5
    ) -> Tuple[str, dict]:
        """
        Smart visualization tool with automated chart selection.
        
        Args:
            csv_path (str): Path to input CSV file
            viz_type (str): Type of visualization (auto, scatter, bar, line, heatmap)
            columns (List[str]): Specific columns to visualize
            max_visualizations (int): Maximum number of visualizations
        
        Returns:
            Tuple[str, dict]: Visualization results and updated state
        """
        # Normalize csv_path
        csv_path = os.path.normpath(csv_path)
        try:
            if not os.path.exists(csv_path):
                error_msg = f"Error: CSV not found at {csv_path}"
                logger.error(error_msg)
                return error_msg, {}
                
            df = pd.read_csv(csv_path)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            plotly_figures = []
            output_image_paths = []
            
            if viz_type == "auto":
                for col in numeric_cols[:min(3, len(numeric_cols))]:
                    fig = px.histogram(df, x=col, title=f'Distribution of {col}',
                                     nbins=50, marginal="box",
                                     color_discrete_sequence=['#636EFA'])
                    plotly_figures.append(fig)
                
                if len(numeric_cols) > 1:
                    corr_matrix = df[numeric_cols].corr()
                    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                                  title="Correlation Heatmap", color_continuous_scale='RdBu_r')
                    plotly_figures.append(fig)
                
                if len(numeric_cols) >= 2:
                    fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1],
                                   title=f'{numeric_cols[0]} vs {numeric_cols[1]}',
                                   trendline="ols", color_discrete_sequence=['#EF553B'])
                    plotly_figures.append(fig)
                
                if len(categorical_cols) > 0:
                    for col in categorical_cols[:min(2, len(categorical_cols))]:
                        value_counts = df[col].value_counts().head(10)
                        fig = px.bar(x=value_counts.index, y=value_counts.values,
                                   title=f'Top 10 Values in {col}',
                                   labels={'x': col, 'y': 'Count'},
                                   color_discrete_sequence=['#00CC96'])
                        plotly_figures.append(fig)
            
            else:
                if viz_type == "scatter" and len(numeric_cols) >= 2:
                    x_col = columns[0] if columns and len(columns) > 0 else numeric_cols[0]
                    y_col = columns[1] if columns and len(columns) > 1 else numeric_cols[1]
                    fig = px.scatter(df, x=x_col, y=y_col, title=f'{x_col} vs {y_col}',
                                   trendline="ols", color_discrete_sequence=['#EF553B'])
                    plotly_figures.append(fig)
                
                elif viz_type == "bar" and len(categorical_cols) > 0:
                    col = columns[0] if columns and len(columns) > 0 else categorical_cols[0]
                    value_counts = df[col].value_counts().head(15)
                    fig = px.bar(x=value_counts.index, y=value_counts.values,
                               title=f'Distribution of {col}',
                               labels={'x': col, 'y': 'Count'},
                               color_discrete_sequence=['#00CC96'])
                    plotly_figures.append(fig)
                
                elif viz_type == "line" and len(numeric_cols) > 0:
                    col = columns[0] if columns and len(columns) > 0 else numeric_cols[0]
                    fig = px.line(df.reset_index(), x='index', y=col,
                                title=f'Trend of {col}',
                                color_discrete_sequence=['#636EFA'])
                    plotly_figures.append(fig)
                
                elif viz_type == "heatmap" and len(numeric_cols) > 1:
                    corr_matrix = df[numeric_cols].corr()
                    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                                  title="Correlation Heatmap", color_continuous_scale='RdBu_r')
                    plotly_figures.append(fig)

            plotly_figures = plotly_figures[:max_visualizations]
            
            for fig in plotly_figures:
                pickle_filename = os.path.normpath(f"static/outputs/plotly_figures/pickle/{uuid.uuid4()}.pickle")
                html_filename = os.path.normpath(f"static/outputs/images/{uuid.uuid4()}.html")
                
                with open(pickle_filename, 'wb') as f:
                    pickle.dump(fig, f)
                fig.write_html(html_filename)
                
                output_image_paths.extend([
                    {"path": pickle_filename, "type": "pickle", 
                     "url": f"/serve_file/{pickle_filename.replace('static/', '').replace(os.sep, '/')}"},
                    {"path": html_filename, "type": "html", 
                     "url": f"/serve_file/{html_filename.replace('static/', '').replace(os.sep, '/')}"}
                ])

            analysis_history.append({
                "task_type": "visualization",
                "thought": f"Generated {len(plotly_figures)} {viz_type} visualizations",
                "timestamp": datetime.now().isoformat(),
                "output_paths": [path["path"] for path in output_image_paths]
            })

            logger.debug(f"Smart visualization completed. Generated {len(plotly_figures)} charts")
            return f"Generated {len(plotly_figures)} visualizations", {
                "output_image_paths": output_image_paths
            }
            
        except Exception as e:
            error_msg = f"Visualization error: {str(e)}"
            logger.error(error_msg)
            return error_msg, {}

    @tool
    def ml_modeling_tool(
        csv_path: str,
        target_column: str,
        model_type: str = "auto",
        test_size: float = 0.2
    ) -> Tuple[str, dict]:
        """
        Machine learning modeling tool with automated model selection.
        
        Args:
            csv_path (str): Path to input CSV file
            target_column (str): Target column for prediction
            model_type (str): Type of ML model (auto, classification, regression)
            test_size (float): Proportion of data for testing
        
        Returns:
            Tuple[str, dict]: Model results and updated state
        """
        # Normalize csv_path
        csv_path = os.path.normpath(csv_path)
        try:
            if not os.path.exists(csv_path):
                error_msg = f"Error: CSV not found at {csv_path}"
                logger.error(error_msg)
                return error_msg, {}
                
            df = pd.read_csv(csv_path)
            
            if target_column not in df.columns:
                return f"Error: Target column '{target_column}' not found in dataset", {}
            
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            le_dict = {}
            for col in X.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                le_dict[col] = le
            
            X = X.fillna(X.mean() if X.select_dtypes(include=[np.number]).shape[1] > 0 else 0)
            
            if model_type == "auto":
                if y.dtype == 'object' or y.nunique() < 20:
                    problem_type = "classification"
                else:
                    problem_type = "regression"
            else:
                problem_type = model_type
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model_results = {}
            
            if problem_type == "classification":
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train_scaled, y_train)
                
                y_pred = model.predict(X_test_scaled)
                
                model_results = {
                    "model_type": "Random Forest Classifier",
                    "accuracy": model.score(X_test_scaled, y_test),
                    "classification_report": classification_report(y_test, y_pred, output_dict=True),
                    "feature_importance": dict(zip(X.columns, model.feature_importances_))
                }
                
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train_scaled, y_train)
                
                y_pred = model.predict(X_test_scaled)
                
                model_results = {
                    "model_type": "Random Forest Regressor",
                    "r2_score": r2_score(y_test, y_pred),
                    "mse": mean_squared_error(y_test, y_pred),
                    "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                    "feature_importance": dict(zip(X.columns, model.feature_importances_))
                }
            
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig = px.bar(importance_df.head(10), x='importance', y='feature',
                        orientation='h', title='Top 10 Feature Importance',
                        color_discrete_sequence=['#EF553B'])
            
            pickle_filename = os.path.normpath(f"static/outputs/plotly_figures/pickle/{uuid.uuid4()}.pickle")
            html_filename = os.path.normpath(f"static/outputs/images/{uuid.uuid4()}.html")
            
            with open(pickle_filename, 'wb') as f:
                pickle.dump(fig, f)
            fig.write_html(html_filename)
            
            output_image_paths = [
                {"path": pickle_filename, "type": "pickle", 
                 "url": f"/serve_file/{pickle_filename.replace('static/', '').replace(os.sep, '/')}"},
                {"path": html_filename, "type": "html", 
                 "url": f"/serve_file/{html_filename.replace('static/', '').replace(os.sep, '/')}"}
            ]

            analysis_history.append({
                "task_type": "modeling",
                "thought": f"Trained {problem_type} model on {target_column}",
                "timestamp": datetime.now().isoformat(),
                "output_paths": [path["path"] for path in output_image_paths]
            })

            logger.debug(f"ML modeling completed. Model type: {problem_type}")
            return f"ML model trained successfully. {problem_type.capitalize()} model performance evaluated.", {
                "analysis_results": model_results,
                "output_image_paths": output_image_paths
            }
            
        except Exception as e:
            error_msg = f"ML modeling error: {str(e)}"
            logger.error(error_msg)
            return error_msg, {}

    @tool
    def automated_pipeline(
        csv_path: str,
        target_column: Optional[str] = None,
        analysis_level: str = "comprehensive"
    ) -> Tuple[str, dict]:
        """
        Fully automated data analysis pipeline combining EDA, visualization, and modeling.
        
        Args:
            csv_path (str): Path to input CSV file
            target_column (Optional[str]): Target column for modeling (optional)
            analysis_level (str): Level of analysis (quick, standard, comprehensive)
        
        Returns:
            Tuple[str, dict]: Pipeline results and updated state
        """
        # Normalize csv_path
        csv_path = os.path.normpath(csv_path)
        try:
            if not os.path.exists(csv_path):
                error_msg = f"Error: CSV not found at {csv_path}"
                logger.error(error_msg)
                return error_msg, {}
                
            combined_results = {
                "eda_results": {},
                "visualization_results": {},
                "modeling_results": {},
                "output_image_paths": []
            }

            eda_output, eda_state = automated_eda_tool(csv_path, analysis_level=analysis_level)
            combined_results["eda_results"] = eda_state.get("analysis_results", {})
            combined_results["output_image_paths"].extend(eda_state.get("output_image_paths", []))
            
            viz_output, viz_state = smart_visualization_tool(csv_path, viz_type="auto",
                                                          max_visualizations=5 if analysis_level == "comprehensive" else 3)
            combined_results["visualization_results"] = viz_state
            combined_results["output_image_paths"].extend(viz_state.get("output_image_paths", []))
            
            if target_column:
                model_output, model_state = ml_modeling_tool(csv_path, target_column,
                                                          model_type="auto", test_size=0.2)
                combined_results["modeling_results"] = model_state.get("analysis_results", {})
                combined_results["output_image_paths"].extend(model_state.get("output_image_paths", []))

            analysis_history.append({
                "task_type": "automated_pipeline",
                "thought": f"Completed automated pipeline with {'EDA, visualization, and modeling' if target_column else 'EDA and visualization'}",
                "timestamp": datetime.now().isoformat(),
                "output_paths": [path["path"] for path in combined_results["output_image_paths"]]
            })

            logger.debug(f"Automated pipeline completed. Generated {len(combined_results['output_image_paths'])} visualizations")
            return (f"Automated pipeline completed successfully. Generated EDA, "
                   f"{len(combined_results['output_image_paths'])} visualizations"
                   f"{', and ML model' if target_column else ''}"), combined_results
            
        except Exception as e:
            error_msg = f"Automated pipeline error: {str(e)}"
            logger.error(error_msg)
            return error_msg, {}

    tools = [complete_python_task, automated_eda_tool, smart_visualization_tool, ml_modeling_tool, automated_pipeline]
    tool_node = ToolNode(tools)
    llm_with_tools = llm.bind_tools(tools)

    def call_llm_model(state: State):
        system_prompt = """
        You are an expert data scientist and analytics AI assistant specializing in automated data analysis pipelines.
        You have access to tools for:
        1. complete_python_task: Execute custom pandas/plotly/sklearn code
        2. automated_eda_tool: Perform comprehensive exploratory data analysis
        3. smart_visualization_tool: Generate intelligent visualizations
        4. ml_modeling_tool: Build and evaluate ML models
        5. automated_pipeline: Run complete analysis pipeline (EDA + visualization + optional modeling)

        For analysis requests:
        - Generic: Use automated_pipeline
        - Specific EDA: Use automated_eda_tool
        - Visualization: Use smart_visualization_tool
        - Modeling: Use ml_modeling_tool
        - Custom: Use complete_python_task

        Always:
        1. Use csv_path from state['csv_path'] with forward slashes (/)
        2. Validate csv_path exists before tool calls
        3. Explain approach in 'thought' parameter
        4. Append visualizations to output_image_paths
        5. Store results in analysis_results
        6. Handle missing data and categorical variables
        7. Provide actionable insights
        8. Cache results for efficiency
        9. Maintain analysis history
        """
        try:
            messages = state["messages"]
            csv_path = os.path.normpath(state.get("csv_path", "")).replace(os.sep, '/')
            
            if not csv_path or not os.path.exists(csv_path):
                error_msg = f"Invalid or missing CSV path: {csv_path}"
                logger.error(error_msg)
                return {
                    "messages": [AIMessage(content=error_msg)],
                    "csv_path": csv_path,
                    "analysis_history": analysis_history
                }

            if not any(safe_get_message_role(msg) == 'system' for msg in messages):
                messages = [SystemMessage(content=system_prompt)] + messages
                
            for msg in messages:
                if isinstance(msg, HumanMessage) and "csv_path" not in msg.content:
                    msg.content += f"\nCSV path: {csv_path}"
            
            cache_key = f"{csv_path}:{safe_get_message_content(messages[-1])}"
            if cache_key in analysis_cache:
                logger.debug("Returning cached result")
                return analysis_cache[cache_key]
            
            response = llm_with_tools.invoke(messages)
            
            analysis_cache[cache_key] = {
                "messages": [response],
                "csv_path": csv_path,
                "analysis_history": analysis_history
            }
            
            return {"messages": [response], "csv_path": csv_path,
                    "analysis_history": analysis_history}
        except Exception as e:
            logger.error(f"LLM call error: {str(e)}")
            return {
                "messages": [AIMessage(content=f"I encountered an error: {str(e)}. Please check the CSV path or try again.")],
                "csv_path": state.get("csv_path", ""),
                "analysis_history": analysis_history
            }

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
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not file.filename.lower().endswith('.csv'):
        return jsonify({"error": "Only CSV files are supported"}), 400
    
    upload_dir = r"static/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    
    filename = os.path.normpath(os.path.join(upload_dir, f"{uuid.uuid4()}.csv")).replace(os.sep, '/')
    
    try:
        file.save(filename)
        df = pd.read_csv(filename)
        logger.debug(f"CSV saved to {filename} with shape: {df.shape}")
        
        # Convert dtypes to string to ensure JSON serializability
        dtypes_dict = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        return jsonify({
            "csv_path": filename,
            "message": "CSV uploaded successfully",
            "info": {
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": dtypes_dict
            }
        })
    except Exception as e:
        logger.error(f"Failed to save CSV: {str(e)}")
        return jsonify({"error": f"Failed to save CSV: {str(e)}"}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        csv_path = os.path.normpath(data.get('csv_path', '')).replace(os.sep, '/')
        
        if not user_message:
            return jsonify({"error": "Message is required"}), 400
        
        if not csv_path or not os.path.exists(csv_path):
            error_msg = f"CSV file not found at {csv_path}"
            logger.error(error_msg)
            return jsonify({"error": error_msg}), 400

        # Generate a request hash to prevent duplicate processing
        request_hash = hashlib.sha256(f"{user_message}:{csv_path}".encode()).hexdigest()
        if request_hash in request_cache:
            logger.debug(f"Duplicate request detected: {request_hash}")
            return jsonify({"message": "Request already processed, please try a new query"}), 200
        request_cache.add(request_hash)

        logger.debug(f"Processing chat request with csv_path: {csv_path}, message: {user_message}")
        
        initial_state = {
            "messages": [HumanMessage(content=user_message)],
            "input_data": [{
                "variable_name": "df",
                "data_type": "csv",
                "data_path": csv_path
            }],
            "current_variables": persistent_vars,
            "intermediate_outputs": [],
            "output_image_paths": [],
            "analysis_results": {},
            "csv_path": csv_path,
            "analysis_history": analysis_history
        }
        
        result = tool_agent.invoke(initial_state)
        
        # Clear request cache after successful processing
        request_cache.discard(request_hash)

        response_data = {
            "message": "Analysis completed",
            "intermediate_outputs": result.get("intermediate_outputs", []),
            "analysis_results": result.get("analysis_results", {}),
            "model_metrics": result.get("model_metrics", {}),
            "current_variables": result.get("current_variables", {}),
            "messages": [],
            "output_images": result.get("output_image_paths", []),
            "analysis_history": result.get("analysis_history", [])
        }
        
        for msg in result.get("messages", []):
            try:
                role = safe_get_message_role(msg)
                content = safe_get_message_content(msg)
                response_data["messages"].append({
                    "role": role,
                    "content": content
                })
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                response_data["messages"].append({
                    "role": "assistant",
                    "content": "Analysis completed successfully"
                })
        
        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        request_cache.discard(request_hash)
        return jsonify({"error": f"Failed to process request: {str(e)}"}), 500

@app.route('/serve_file/<path:filename>')
def serve_file(filename):
    try:
        file_path = os.path.normpath(os.path.join('static', filename)).replace(os.sep, '/')
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 400
            
        if filename.lower().endswith('.png'):
            return send_file(file_path, mimetype='image/png')
        elif filename.lower().endswith('.html'):
            return send_file(file_path, mimetype='text/html')
        elif filename.lower().endswith('.pickle'):
            return send_file(file_path, mimetype='application/octet-stream')
        else:
            return jsonify({"error": "Unsupported file type"}), 400
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({"error": f"Failed to serve file: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)