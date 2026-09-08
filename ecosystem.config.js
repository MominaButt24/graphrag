module.exports = {
  apps: [
    {
      name: "graphrag-api",
      script: "./env/bin/python",
      args: "-m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload",
      cwd: __dirname,
    },

    {
      name: "graphrag-gradio",
      script: "./env/bin/python",
      args: "gradio_app.py",
      cwd: __dirname,
    },

    // {
    //   name: "chainlit-app",
    //   script: "./env/bin/python",
    //   args: "-m chainlit run chainlit_app.py -w --port 8001",
    //   cwd: __dirname,
    // },
  ],
};