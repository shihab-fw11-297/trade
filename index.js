ML Model
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