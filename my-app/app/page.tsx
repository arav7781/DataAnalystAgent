"use client";
import type React from "react";
import { useState, useRef, useEffect } from "react";
import { Upload, Send, Zap, Brain, ImageIcon, Sparkles, Download, BarChart, FileText } from "lucide-react";

interface Message {
  role: string;
  content: string;
  images?: { path: string; type: string; url: string }[];
}

interface AnalysisResponse {
  message: string;
  intermediate_outputs: any[];
  analysis_results: any;
  model_metrics: any;
  current_variables: any;
  messages: { role: string; content: string }[];
  output_images: { path: string; type: string; url: string }[];
  analysis_history: any[];
}

const DataAnalyticsAI: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [csvPath, setCsvPath] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [analysisResults, setAnalysisResults] = useState<any>(null);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Scroll to bottom of chat container when messages update
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages]);

  // Handle CSV file upload
  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0] && !isUploading) {
      const file = event.target.files[0];
      if (!file.name.toLowerCase().endsWith(".csv")) {
        setError("Please upload a CSV file");
        return;
      }

      setIsUploading(true);
      setSelectedFile(file);
      setError(null);
      setAnalysisResults(null);
      setMessages([]);
      setCsvPath(null);

      const formData = new FormData();
      formData.append("file", file);

      try {
        setLoading(true);
        const response = await fetch("http://localhost:5000/upload", {
          method: "POST",
          body: formData,
        });

        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || "Upload failed");
        }

        setCsvPath(data.csv_path);
        setMessages([
          {
            role: "system",
            content: `CSV uploaded successfully to ${data.csv_path}. Initial analysis completed. Ask for specific analysis or visualizations.`,
            images: data.pipeline_results.output_image_paths || [],
          },
        ]);
        setAnalysisResults(data.pipeline_results);
      } catch (err: any) {
        setError(err.message || "Failed to upload CSV");
        console.error("Upload error:", err);
        if (fileInputRef.current) {
          fileInputRef.current.value = "";
        }
        setSelectedFile(null);
      } finally {
        setLoading(false);
        setIsUploading(false);
      }
    }
  };

  // Handle download of visualizations
  const handleDownloadFile = async (url: string, filename: string) => {
    try {
      const response = await fetch(url);
      const blob = await response.blob();
      const downloadUrl = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = downloadUrl;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(downloadUrl);
    } catch (error) {
      console.error("Download failed:", error);
      setError("Failed to download file");
    }
  };

  // Handle sending chat messages
  const handleSendMessage = async () => {
    if (!input.trim()) {
      setError("Please enter a message");
      return;
    }

    if (!csvPath) {
      setError("Please upload a CSV file first");
      return;
    }

    const userMessage: Message = { role: "user", content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);
    setError(null);

    try {
      const response = await fetch("http://localhost:5000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: input,
          csv_path: csvPath,
        }),
      });

      const data: AnalysisResponse = await response.json();
      if (!response.ok) {
        throw new Error(data.message || "Analysis failed");
      }

      const updatedMessages = [...messages, userMessage];

      if (data.messages && Array.isArray(data.messages)) {
        data.messages.forEach((msg) => {
          updatedMessages.push({
            role: msg.role,
            content: msg.content,
            images: data.output_images || [],
          });
        });
      }

      if (data.intermediate_outputs && Array.isArray(data.intermediate_outputs)) {
        data.intermediate_outputs.forEach((output) => {
          if (output.thought) {
            updatedMessages.push({
              role: "system",
              content: `💭 **Thought**: ${output.thought}`,
            });
          }
          if (output.code) {
            updatedMessages.push({
              role: "system",
              content: `🔧 Code Executed:\n\`\`\`python\n${output.code}\n\`\`\``,
            });
          }
          if (output.output) {
            updatedMessages.push({
              role: "system",
              content: `📋 **Output**: ${output.output}`,
              images: data.output_images || [],
            });
          }
        });
      }

      setMessages(updatedMessages);
      if (data.analysis_results && Object.keys(data.analysis_results).length > 0) {
        setAnalysisResults(data.analysis_results);
      }
    } catch (err: any) {
      console.error("Chat error:", err);
      setError(err.message || "An error occurred during analysis");
      setMessages((prev) => [
        ...prev,
        {
          role: "system",
          content: `❌ **Error**: ${err.message || "Analysis failed"}`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  // Render message content with support for markdown and code blocks
  const renderMessageContent = (message: Message) => {
    const { content } = message;

    if (content.includes("```")) {
      const parts = content.split("```");
      return parts.map((part, index) => {
        if (index % 2 === 1) {
          const lines = part.split("\n");
          const language = lines[0] || "";
          const code = lines.slice(1).join("\n");
          return (
            <pre key={index} className="bg-gray-900 p-4 rounded-lg mt-3 overflow-x-auto border border-gray-300">
              <code className="text-green-400 text-sm font-mono">{code}</code>
            </pre>
          );
        }
        return <span key={index}>{part}</span>;
      });
    }

    if (content.includes("**")) {
      const parts = content.split("**");
      return parts.map((part, index) =>
        index % 2 === 1 ? (
          <strong key={index} className="font-semibold text-green-700">{part}</strong>
        ) : (
          <span key={index}>{part}</span>
        ),
      );
    }

    return content;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 via-emerald-50 to-green-100 text-gray-800 relative overflow-hidden">
      {/* Background Elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-gradient-to-br from-green-200 to-emerald-200 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-pulse"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-gradient-to-br from-emerald-200 to-green-200 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-pulse delay-1000"></div>
      </div>

      <div className="relative z-10 flex flex-col items-center p-6">
        {/* Header */}
        <div className="text-center mb-8 opacity-0 animate-fade-in">
          <div className="flex items-center justify-center gap-4 mb-6">
            <div className="p-4 bg-gradient-to-br from-green-500 to-emerald-600 rounded-2xl shadow-lg">
              <Brain className="w-10 h-10 text-white" />
            </div>
            <h1 className="text-6xl font-bold bg-gradient-to-r from-green-600 via-emerald-600 to-green-700 bg-clip-text text-transparent">
              DataAnalyticsAI
            </h1>
            <div className="p-4 bg-gradient-to-br from-emerald-500 to-green-600 rounded-2xl shadow-lg">
              <BarChart className="w-10 h-10 text-white" />
            </div>
          </div>
          <p className="text-2xl text-green-700 font-light">Advanced Data Analysis Platform</p>
          <div className="w-24 h-1 bg-gradient-to-r from-green-500 to-emerald-500 mx-auto mt-4 rounded-full"></div>
        </div>

        {/* CSV Upload Section */}
        <div className="w-full max-w-6xl mb-8">
          <div className="bg-white/80 backdrop-blur-sm border border-green-200 rounded-2xl p-8 shadow-lg hover:shadow-xl transition-all duration-300">
            <div className="flex items-center gap-3 mb-6">
              <div className="p-2 bg-green-100 rounded-lg">
                <Upload className="w-6 h-6 text-green-600" />
              </div>
              <h2 className="text-2xl font-semibold text-gray-800">Upload CSV</h2>
            </div>
            <div className="relative">
              <input
                ref={fileInputRef}
                type="file"
                accept=".csv"
                onChange={handleFileChange}
                className="w-full p-6 border-2 border-dashed border-green-300 rounded-xl bg-green-50/50 text-gray-700 file:mr-4 file:py-3 file:px-6 file:rounded-lg file:border-0 file:bg-green-500 file:text-white hover:file:bg-green-600 transition-all duration-300 hover:border-green-400 hover:bg-green-50 focus:outline-none focus:border-green-500"
                disabled={loading || isUploading}
              />
              {(loading || isUploading) && (
                <div className="absolute inset-0 bg-white/70 backdrop-blur-sm rounded-xl flex items-center justify-center">
                  <div className="flex items-center gap-3">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-green-500"></div>
                    <span className="text-green-700 font-medium">{isUploading ? "Uploading..." : "Analyzing..."}</span>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Dataset Info */}
          {selectedFile && csvPath && (
            <div className="bg-white/80 backdrop-blur-sm border border-green-200 rounded-2xl p-8 shadow-lg mt-6 opacity-0 animate-slide-up hover:shadow-xl transition-all duration-300">
              <div className="flex items-center gap-3 mb-6">
                <div className="p-2 bg-green-100 rounded-lg">
                  <FileText className="w-6 h-6 text-green-600" />
                </div>
                <h2 className="text-2xl font-semibold text-gray-800">Dataset Info</h2>
              </div>
              <div className="text-gray-700">
                <p><strong>File:</strong> {selectedFile.name}</p>
                <p><strong>Path:</strong> {csvPath}</p>
                {analysisResults?.info && (
                  <>
                    <p><strong>Shape:</strong> {analysisResults.info.shape[0]} rows, {analysisResults.info.shape[1]} columns</p>
                    <p><strong>Columns:</strong> {analysisResults.info.columns.join(", ")}</p>
                  </>
                )}
              </div>
            </div>
          )}

          {/* Analysis Results */}
          {analysisResults && Object.keys(analysisResults).length > 0 && (
            <div className="bg-white/80 backdrop-blur-sm border border-green-200 rounded-2xl p-8 shadow-lg mt-6 opacity-0 animate-slide-up hover:shadow-xl transition-all duration-300">
              <div className="flex items-center gap-3 mb-6">
                <div className="p-2 bg-green-100 rounded-lg">
                  <Zap className="w-6 h-6 text-green-600" />
                </div>
                <h2 className="text-2xl font-semibold text-gray-800">Analysis Results</h2>
              </div>
              <pre className="bg-green-50 p-6 rounded-xl text-sm overflow-x-auto border border-green-200">
                <code className="text-green-800 font-mono">{JSON.stringify(analysisResults, null, 2)}</code>
              </pre>
            </div>
          )}
        </div>

        {/* Chat Section */}
        <div className="w-full max-w-6xl bg-white/80 backdrop-blur-sm border border-green-200 rounded-2xl shadow-lg p-8 mb-6 hover:shadow-xl transition-all duration-300">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 bg-green-100 rounded-lg">
              <Brain className="w-6 h-6 text-green-600" />
            </div>
            <h2 className="text-2xl font-semibold text-gray-800">Data Analysis Assistant</h2>
          </div>

          <div
            ref={chatContainerRef}
            className="h-96 overflow-y-auto mb-6 p-6 bg-gradient-to-br from-green-50/50 to-emerald-50/50 rounded-xl border border-green-200"
          >
            {messages.map((msg, index) => (
              <div
                key={index}
                className={`mb-4 p-6 rounded-xl opacity-0 animate-message-slide shadow-sm ${
                  msg.role === "user"
                    ? "bg-green-100 border border-green-200 ml-8"
                    : msg.role === "system"
                    ? "bg-blue-50 border border-blue-200"
                    : "bg-white border border-gray-200 mr-8"
                }`}
                style={{ animationDelay: `${index * 100}ms` }}
              >
                <div className="font-semibold mb-3 flex items-center gap-2">
                  {msg.role === "user" ? (
                    <>
                      <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                      <span className="text-green-700">You</span>
                    </>
                  ) : msg.role === "system" ? (
                    <>
                      <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                      <span className="text-blue-700">System</span>
                    </>
                  ) : (
                    <>
                      <div className="w-3 h-3 bg-emerald-500 rounded-full"></div>
                      <span className="text-gray-700">AI Assistant</span>
                    </>
                  )}
                </div>
                <div className="whitespace-pre-wrap text-gray-700 leading-relaxed">{renderMessageContent(msg)}</div>
                {msg.images && msg.images.length > 0 && (
                  <div className="mt-6">
                    <div className="flex items-center justify-between mb-3">
                      <p className="text-sm font-semibold text-gray-700">Visualizations:</p>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {msg.images.map((img, imgIndex) => (
                        img.type === "html" && (
                          <div key={imgIndex} className="relative group">
                            <iframe
                              src={img.url}
                              className="w-full h-64 rounded-xl shadow-lg border border-green-200 group-hover:shadow-xl transition-all duration-300"
                              title={`Visualization ${imgIndex + 1}`}
                            />
                            <button
                              onClick={() => handleDownloadFile(img.url, `visualization-${imgIndex + 1}.html`)}
                              className="absolute top-2 right-2 flex items-center gap-1 text-xs text-green-600 hover:text-green-700 bg-green-50 hover:bg-green-100 px-3 py-2 rounded-lg transition-all duration-200 font-medium"
                            >
                              <Download className="w-3 h-3" />
                              Download
                            </button>
                          </div>
                        )
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ))}
            {loading && (
              <div className="text-center text-green-600 py-8">
                <div className="flex items-center justify-center gap-3">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-green-500"></div>
                  <span className="text-lg font-medium">Analyzing...</span>
                </div>
              </div>
            )}
          </div>

          {/* Input Section */}
          <div className="flex gap-4">
            <div className="flex-1 relative">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={(e) => e.key === "Enter" && !loading && handleSendMessage()}
                className="w-full p-4 rounded-xl text-gray-700 border bg-white border-green-300 focus:outline-none focus:border-green-500 focus:ring-2 focus:ring-green-200 transition-all duration-300 placeholder-gray-400"
                placeholder="Enter analysis instructions (e.g., 'create a scatter plot', 'run regression', 'show correlations')..."
                disabled={loading}
              />
              <div className="absolute right-4 top-1/2 transform -translate-y-1/2">
                <Sparkles className="w-5 h-5 text-green-500" />
              </div>
            </div>
            <button
              onClick={handleSendMessage}
              disabled={loading || !input.trim() || !csvPath}
              className="bg-green-500 text-white px-8 py-4 rounded-xl disabled:bg-gray-400 disabled:cursor-not-allowed hover:bg-green-600 transition-all duration-300 flex items-center gap-2 shadow-md hover:shadow-lg font-medium"
            >
              {loading ? (
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
              ) : (
                <Send className="w-5 h-5" />
              )}
              <span>Send</span>
            </button>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="w-full max-w-6xl bg-red-50 border border-red-200 text-red-700 px-6 py-4 rounded-xl mb-6 shadow-sm opacity-0 animate-shake">
            <div className="flex items-center gap-3">
              <div className="w-2 h-2 bg-red-500 rounded-full"></div>
              <strong>Error:</strong> {error}
            </div>
          </div>
        )}

        {/* Quick Actions */}
        <div className="w-full max-w-6xl bg-white/80 backdrop-blur-sm border border-green-200 rounded-2xl p-8 shadow-lg hover:shadow-xl transition-all duration-300">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 bg-green-100 rounded-lg">
              <Zap className="w-6 h-6 text-green-600" />
            </div>
            <h3 className="text-2xl font-semibold text-gray-800">Quick Actions</h3>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              "Run comprehensive EDA",
              "Create scatter plots",
              "Generate correlation heatmap",
              "Build predictive model",
            ].map((action, index) => (
              <button
                key={action}
                onClick={() => !loading && setInput(action)}
                disabled={!csvPath || loading}
                className="bg-green-50 hover:bg-green-100 disabled:bg-gray-100 disabled:text-gray-400 text-green-700 px-6 py-4 rounded-xl text-sm transition-all duration-300 border border-green-200 hover:border-green-300 opacity-0 animate-fade-in font-medium hover:shadow-md"
                style={{ animationDelay: `${index * 100}ms` }}
              >
                {action}
              </button>
            ))}
          </div>
        </div>
      </div>

      <style jsx>{`
        @keyframes fade-in {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        @keyframes slide-up {
          from {
            opacity: 0;
            transform: translateY(30px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        @keyframes message-slide {
          from {
            opacity: 0;
            transform: translateX(-20px);
          }
          to {
            opacity: 1;
            transform: translateX(0);
          }
        }
        @keyframes shake {
          0%, 100% {
            transform: translateX(0);
          }
          25% {
            transform: translateX(-5px);
          }
          75% {
            transform: translateX(5px);
          }
        }
        .animate-fade-in {
          animation: fade-in 0.6s ease-out forwards;
        }
        .animate-slide-up {
          animation: slide-up 0.5s ease-out forwards;
        }
        .animate-message-slide {
          animation: message-slide 0.4s ease-out forwards;
        }
        .animate-shake {
          animation: shake 0.5s ease-in-out forwards;
        }
        .delay-1000 {
          animation-delay: 1s;
        }
        .delay-2000 {
          animation-delay: 2s;
        }
      `}</style>
    </div>
  );
};

export default DataAnalyticsAI;