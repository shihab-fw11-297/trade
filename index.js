const axios = require("axios");
const express = require("express");
const tf = require("@tensorflow/tfjs-node"); // Import TensorFlow
const { Sequential, tensor } = tf;
const TechnicalIndicators = require("technicalindicators");
const app = express();

// Finazon API configuration
const FINAZON_API_KEY = "18535cbd97e2400d93f96802097d83c9af";
const BASE_URL = "https://api.finazon.io/latest/finazon/forex/time_series";

// Fetch Forex Data from Finazon API
async function fetchForexData(
  ticker = "EUR/USD",
  interval = "1m",
  pageSize = 60
) {
  const url = `${BASE_URL}?ticker=${ticker}&interval=${interval}&page=0&page_size=${pageSize}&apikey=${FINAZON_API_KEY}`;
  try {
    const response = await axios.get(url);
    return response.data.data;
  } catch (error) {
    console.error("Error fetching forex data:", error);
    return null;
  }
}

// Calculate Technical Indicators
function calculateTechnicalIndicators(data) {
  const closePrices = data.map((item) => parseFloat(item.c));
  const highPrices = data.map((item) => parseFloat(item.h));
  const lowPrices = data.map((item) => parseFloat(item.l));

  // Default indicators object
  const indicators = {
    rsi: null,
    ema: null,
    atr: null,
    macd: null,
    stochastic: null,
    upperBand: null,
    lowerBand: null,
    fibonacciLevels: null,
  };

  // RSI
  if (closePrices.length >= 14) {
    indicators.rsi = TechnicalIndicators.RSI.calculate({
      values: closePrices,
      period: 14,
    }).slice(-1)[0];
  }

  // EMA
  if (closePrices.length >= 14) {
    indicators.ema = TechnicalIndicators.EMA.calculate({
      values: closePrices,
      period: 14,
    }).slice(-1)[0];
  }

  // ATR
  if (closePrices.length >= 14) {
    indicators.atr = TechnicalIndicators.ATR.calculate({
      high: highPrices,
      low: lowPrices,
      close: closePrices,
      period: 14,
    }).slice(-1)[0];
  }

  // MACD
  if (closePrices.length >= 26) {
    const macd = TechnicalIndicators.MACD.calculate({
      values: closePrices,
      fastPeriod: 12,
      slowPeriod: 26,
      signalPeriod: 9,
    });
    indicators.macd = macd.length ? macd[macd.length - 1].MACD : null;
  }

  // Stochastic
  if (closePrices.length >= 14) {
    const stochastic = TechnicalIndicators.Stochastic.calculate({
      high: highPrices,
      low: lowPrices,
      close: closePrices,
      period: 14,
      signalPeriod: 3,
    });

    indicators.stochastic = stochastic.length
      ? stochastic[stochastic.length - 1].k
      : null;
  }

  // Calculate Fibonacci Retracement Levels
  const { fibonacciLevels } = calculateFibonacciRetracement(data);
  indicators.fibonacciLevels = fibonacciLevels;

  // Calculate Bollinger Bands
  const { upperBand, lowerBand } = calculateBollingerBands(closePrices);
  indicators.upperBand = upperBand;
  indicators.lowerBand = lowerBand;

  return indicators;
}

// Calculate Fibonacci Retracement Levels
function calculateFibonacciRetracement(data) {
  const closePrices = data.map((item) => parseFloat(item.c));
  const maxPrice = Math.max(...closePrices);
  const minPrice = Math.min(...closePrices);
  const difference = maxPrice - minPrice;

  const firstLevel = maxPrice - difference * 0.236;
  const secondLevel = maxPrice - difference * 0.382;
  const thirdLevel = maxPrice - difference * 0.618;

  return { fibonacciLevels: [firstLevel, secondLevel, thirdLevel] };
}

// Function to calculate standard deviation
function calculateStandardDeviation(values, period) {
  const mean =
    values.slice(-period).reduce((acc, val) => acc + val, 0) / period;
  const variance =
    values
      .slice(-period)
      .reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / period;
  return Math.sqrt(variance);
}

// Calculate Bollinger Bands
function calculateBollingerBands(
  closePrices,
  period = 20,
  stdDevMultiplier = 2
) {
  const middleBand = TechnicalIndicators.SMA.calculate({
    values: closePrices,
    period,
  });

  // Calculate standard deviation for the latest period
  const stdDev = calculateStandardDeviation(closePrices, period);

  // Calculate the latest upper and lower bands
  const upperBand =
    middleBand[middleBand.length - 1] + stdDevMultiplier * stdDev;
  const lowerBand =
    middleBand[middleBand.length - 1] - stdDevMultiplier * stdDev;

  return { upperBand, lowerBand };
}

// Custom Trading Strategies
function customTradingStrategy(data) {
  const { rsi, ema, macd, stochastic, upperBand, lowerBand } =
    calculateTechnicalIndicators(data);
  const closePrices = data.map((item) => parseFloat(item.c));
  const volume = calculateVolume(data);


  console.log( calculateTechnicalIndicators(data));
  const lastClose = closePrices[closePrices.length - 1];
  const lastVolume = volume[volume.length - 1];

  // Confidence thresholds can be adjusted based on backtesting
  const confidenceThreshold = 0.7;

  let buySignalConfidence = 0;
  let sellSignalConfidence = 0;

  // Buy Condition
  if (
    rsi < 30 &&
    lastClose < ema &&
    macd?.macd > macd?.signal &&
    lastClose < lowerBand
  ) {
    buySignalConfidence = 1; // Signal for buy
  }

  // Sell Condition
  if (
    rsi > 70 &&
    lastClose > ema &&
    macd?.macd < macd?.signal &&
    lastClose > upperBand
  ) {
    sellSignalConfidence = 1; // Signal for sell
  }

  // Return trade decision based on confidence
  if (buySignalConfidence >= confidenceThreshold) {
    return "up"; // Buy signal
  } else if (sellSignalConfidence >= confidenceThreshold) {
    return "down"; // Sell signal
  } else {
    return "flat"; // Hold signal
  }
}

// Additional Strategy: Price Action Analysis
function priceActionStrategy(data) {
  const lastClose = parseFloat(data[data.length - 1].c);
  const previousClose = parseFloat(data[data.length - 2].c);

  if (lastClose > previousClose) {
    return "up"; // Bullish signal
  } else if (lastClose < previousClose) {
    return "down"; // Bearish signal
  } else {
    return "flat"; // No movement
  }
}

function analyzeCandlestickPatterns(data) {
  if (data.length < 2) {
    return "Not enough data";
  }

  const lastCandle = data[data.length - 1];
  const previousCandle = data[data.length - 2];

  // Check for Bullish Engulfing
  const isBullishEngulfing =
    previousCandle.close < previousCandle.open && // Previous candle is bearish
    lastCandle.close > lastCandle.open && // Last candle is bullish
    lastCandle.open < previousCandle.close && // Last candle opens lower
    lastCandle.close > previousCandle.open; // Last candle closes higher

  // Check for Bearish Engulfing
  const isBearishEngulfing =
    previousCandle.close > previousCandle.open && // Previous candle is bullish
    lastCandle.close < lastCandle.open && // Last candle is bearish
    lastCandle.open > previousCandle.close && // Last candle opens higher
    lastCandle.close < previousCandle.open; // Last candle closes lower

  // Check for Hammer
  const isHammer =
    lastCandle.close > lastCandle.open && // Bullish candle
    lastCandle.high - lastCandle.close <=
      (lastCandle.close - lastCandle.open) * 2 && // Small upper shadow
    lastCandle.open - lastCandle.low >=
      (lastCandle.close - lastCandle.open) * 2; // Long lower shadow

  // Check for Shooting Star
  const isShootingStar =
    lastCandle.close < lastCandle.open && // Bearish candle
    lastCandle.high - lastCandle.open <=
      (lastCandle.open - lastCandle.close) * 2 && // Small upper shadow
    lastCandle.open - lastCandle.low >=
      (lastCandle.open - lastCandle.close) * 2; // Long lower shadow

  // Predict direction based on patterns
  if (isBullishEngulfing) {
    return "up";
  } else if (isBearishEngulfing) {
    return "down";
  } else if (isHammer) {
    return "may up";
  } else if (isShootingStar) {
    return "may down";
  } else {
    return "flat";
  }
}

// Additional Strategy: Moving Average Crossover
function movingAverageCrossover(data) {
  const closePrices = data.map((item) => parseFloat(item.c));
  const shortMA = TechnicalIndicators.SMA.calculate({
    values: closePrices,
    period: 5,
  }).slice(-1)[0];
  const longMA = TechnicalIndicators.SMA.calculate({
    values: closePrices,
    period: 20,
  }).slice(-1)[0];

  if (shortMA > longMA) {
    return "up"; // Buy signal
  } else if (shortMA < longMA) {
    return "down"; // Sell signal
  } else {
    return "flat"; // Hold signal
  }
}

// Machine Learning Model
async function createMLModel() {
  const model = new Sequential();
  model.add(
    tf.layers.dense({ units: 32, activation: "relu", inputShape: [8] })
  ); // Adjusted input shape
  model.add(tf.layers.dense({ units: 16, activation: "relu" }));
  model.add(tf.layers.dense({ units: 3, activation: "softmax" })); // Up, Down, Flat
  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });
  return model;
}

// Prepare Data for ML Model
function prepareData(data) {
  const features = data.map((item) => {
    const indicators = calculateTechnicalIndicators(data);
    return [
      parseFloat(item.c),
      parseFloat(item.h),
      parseFloat(item.l),
      parseFloat(item.o),
      indicators.rsi,
      indicators.ema,
      indicators.upperBand,
      indicators.lowerBand,
    ];
  });

  return tensor(features);
}

// Train ML Model
async function trainMLModel(model, trainingData) {
  const xs = prepareData(trainingData);
  const ys = tensor(
    trainingData.map((item) => {
      if (item.c > item.o) return [1, 0, 0]; // Up
      if (item.c < item.o) return [0, 1, 0]; // Down
      return [0, 0, 1]; // Flat
    })
  );

  await model.fit(xs, ys, { epochs: 50 });
  return model;
}

// Predict with ML Model
async function predictWithModel(model, data) {
  const inputData = prepareData(data);
  const prediction = model.predict(inputData).arraySync();
  const predictions = prediction.map((pred) =>
    pred[0] > pred[1] ? "up" : pred[1] > pred[2] ? "down" : "flat"
  );
  return predictions[predictions.length - 1];
}

// Custom Analysis Function
function customAnalysis(data) {
  const lastClose = parseFloat(data[data.length - 1].c);
  const previousClose = parseFloat(data[data.length - 2].c);

  // Example of a custom condition
  if (lastClose > previousClose * 1.01) {
    // 1% increase
    return "bullish"; // Indicating bullish behavior
  } else if (lastClose < previousClose * 0.99) {
    // 1% decrease
    return "bearish"; // Indicating bearish behavior
  } else {
    return "neutral"; // No significant movement
  }
}

// Calculate Volume Indicator
function calculateVolume(data) {
  return data.map((item) => parseFloat(item.v)); // Assuming 'v' is the volume field in your data
}

// Main Prediction Flow
app.get("/predict", async (req, res) => {
  try {
    // Fetching forex data
    const forexData = await fetchForexData("EUR/USD", "1m", 300);

    // Ensure data has enough points for the calculation
    if (forexData.length < 30) {
      return res.status(400).json({ error: "Not enough data for prediction" });
    }

    // Machine Learning Model
    const model = await createMLModel();
    await trainMLModel(model, forexData);
    const mlPrediction = await predictWithModel(model, forexData);

    // Applying strategies
    const technicalSignal = customTradingStrategy(forexData);
    const priceActionSignal = priceActionStrategy(forexData);
    const maCrossoverSignal = movingAverageCrossover(forexData);
    const analysisSignal = customAnalysis(forexData);
    const checkCandles = analyzeCandlestickPatterns(forexData);

    // Combine signals
    const finalSignal = combineAllStrategies({
      technical: technicalSignal,
      priceAction: priceActionSignal,
      movingAverage: maCrossoverSignal,
      ml: mlPrediction,
      analysis: analysisSignal
    });

    // Improved response with confidence levels
    const tradeDecision = {
      prediction: finalSignal,
      analysis: analysisSignal,
      ml: mlPrediction,
      strategies: {
        technical: technicalSignal,
        priceAction: priceActionSignal,
        maCrossover: maCrossoverSignal,
        candles: checkCandles,
      },
    };

    res.json(tradeDecision);
  } catch (error) {
    console.error("Error in prediction:", error);
    res.status(500).send("Internal Server Error");
  }
});

// Combine Signals from All Strategies
function combineAllStrategies(signals) {
  const { technical, priceAction, movingAverage, ml, analysis } = signals;

  // Define weights for each strategy
  const weights = {
    technical: 0.2,
    priceAction: 0.2,
    movingAverage: 0.2,
    candles:0.2,
    ml: 0.4,
  };

  // Initialize a counter for buy/sell/flat signals
  const signalCounts = {
    up: 0,
    down: 0,
    flat: 0,
  };

  // Count signals based on weights
  signalCounts[technical] += weights.technical;
  signalCounts[priceAction] += weights.priceAction;
  signalCounts[movingAverage] += weights.movingAverage;
  signalCounts[ml] += weights.ml;

  // Add analysis signal
  if (analysis === "bullish") {
    signalCounts.up += 0.1; // Adjust the weight for analysis signal
  } else if (analysis === "bearish") {
    signalCounts.down += 0.1; // Adjust the weight for analysis signal
  }

  // Determine the final decision based on the highest count
  let finalDecision = "flat"; // Default to flat
  if (signalCounts.up > signalCounts.down) {
    finalDecision = "up";
  } else if (signalCounts.down > signalCounts.up) {
    finalDecision = "down";
  }

  return finalDecision;
}

// Start the server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});