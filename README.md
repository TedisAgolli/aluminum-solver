# Aluminum Cutting Calculator

A web app that calculates the optimal way to cut aluminum bars to minimize waste.

## Local Development

### Prerequisites
- Python 3.11+
- Vercel CLI

### Setup

1. Install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Install Vercel CLI (if not already installed):
   ```bash
   npm i -g vercel
   ```

3. Run the dev server:
   ```bash
   vercel dev
   ```

4. Open http://localhost:3000

## Deployment

Push to main branch or run:
```bash
vercel --prod
```
