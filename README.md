# Aluminum Cutting Calculator

A web application that calculates the optimal cutting plan for aluminum bars using linear programming.

## Features

- Calculate minimum number of bars needed
- Optimal cutting plan for each bar
- Waste calculation and efficiency metrics
- Simple, easy-to-use interface

## Local Development

1. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Run the development server:
```bash
python3 main.py
```

4. Open http://localhost:8000 in your browser

## Deployment

1. Install Vercel CLI (one time):
```bash
npm i -g vercel
```

2. Deploy to Vercel:
```bash
vercel --prod
```

## How It Works

1. Enter the length of your stock aluminum bars (default: 6500mm)
2. Add the pieces you need (length and quantity)
3. Click "Calculate Cutting Plan"
4. See how many bars to buy and how to cut them

## Technology Stack

- **Backend**: FastAPI (Python)
- **Frontend**: Vanilla HTML/CSS/JavaScript
- **Solver**: PuLP (linear programming)
- **Deployment**: Vercel
