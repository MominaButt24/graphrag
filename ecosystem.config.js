module.exports = {
  apps: [
    {
      name: "graphrag-api",
      script: "/usr/local/bin/python3",
      args: "-m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload",
      cwd: __dirname,
    },
    {
      name: "graphrag-gradio",
      script: "/usr/local/bin/python3",
      args: "gradio_app.py",
      cwd: __dirname,
    },
    {
      name: "chainlit-app",
      script: "uv",
      args: "run chainlit run chainlit_app.py -w --port 8001",
      cwd: __dirname,
    },

  ],
};