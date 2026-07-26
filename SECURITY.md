# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Reporting a Vulnerability

ExplainX AI is designed to run entirely on local infrastructure — the FastAPI
backend binds to `127.0.0.1` by default and makes no outbound network calls at
runtime. That said, if you discover a security vulnerability (for example, in
how the dashboard communicates with the API, or in dependency handling),
please report it responsibly:

1. **Do not** open a public issue describing the vulnerability in detail.
2. Email the maintainer directly at **mirajgaonkarshaunak@gmail.com** with:
   - A description of the vulnerability
   - Steps to reproduce
   - Potential impact
3. You should receive an acknowledgment within a reasonable timeframe.
   Once the issue is confirmed, a fix will be prioritized and a new release
   will be published.

## Security Considerations for Deployers

If you deploy ExplainX AI beyond local use (e.g. on a shared network):

- The FastAPI backend has permissive CORS (`allow_origins=["*"]`) by default
  for local development — restrict this before exposing the API beyond
  `localhost`.
- The SQLite database (`explainx.db`) contains model metadata, explanation
  logs, and fairness audit results — treat it as sensitive if your models
  were trained on real (non-synthetic) data.
- No authentication is implemented by default. Do not expose the API or
  dashboard directly to the public internet without adding an auth layer.
